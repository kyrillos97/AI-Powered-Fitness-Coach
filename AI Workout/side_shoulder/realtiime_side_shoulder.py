#!/usr/bin/env python3
"""
realtime_side_shoulder_tf.py  –  Real-time side shoulder raise tracker (TensorFlow version).
All modifications applied: Hip-anchored dynamic scaling, corrected state machine boundaries.
"""

import json, time, collections
from enum import Enum
from typing import List

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  GLOBAL CONFIGURATION PARAMETERS (Modify easily here)
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG_PATH = "vae_config_side_shoulder.json"
MODEL_PATH  = "st_gcvae_side_shoulder.weights.h5"

FPS_TARGET      = 10          # Target frame rate for tracking loop
PRE_BUFFER_SIZE = 10          # Frames captured before movement triggers recording
STATIC_CONFIRM  = 5           # Frames required in STATIC zone to finalize/save a rep

# Geometry & Range Adjustments
MID_LINE_FACTOR = 0.5         # 0.5 = exactly halfway between rest wrist and elbow (Static/Low boundary)
OVER_LINE_OFFSET_TORSO = 0.2  # Distance above shoulder line for OVER boundary (in fractions of torso length)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD MODEL CONFIG CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

with open(CONFIG_PATH) as f:
    CFG = json.load(f)

NUM_LANDMARKS   = CFG["num_landmarks"]          # 33
FEAT_PER_LM     = CFG["features_per_landmark"]  # 4
TEMPORAL_LENGTH = CFG["temporal_length"]        # 20
LATENT_DIM      = CFG["latent_dim"]             # 32
HIDDEN_C        = CFG["hidden_channels"]        # 64
THRESHOLD       = 200            # Rejection threshold from training (~2.3889)

VAE_REJECTION_THRESHOLD = THRESHOLD  
GLOBAL_MEAN     = np.array(CFG["global_mean"], dtype=np.float32)   
GLOBAL_STD      = np.array(CFG["global_std"],  dtype=np.float32)   
EDGES           = [tuple(e) for e in CFG["adjacency_edges"]]

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL DEFINITION (TensorFlow / Keras GCN-VAE)
# ═══════════════════════════════════════════════════════════════════════════════

def build_adjacency(edges, num_nodes):
    A = np.eye(num_nodes, dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = np.diag(A.sum(axis=1) ** -0.5)
    return tf.convert_to_tensor(D @ A @ D, dtype=tf.float32)


class GraphConv(tf.keras.layers.Layer):
    def __init__(self, out_f, **kwargs):
        super().__init__(**kwargs)
        self.out_f = out_f

    def build(self, input_shape):
        in_f = input_shape[-1]
        self.W = self.add_weight(shape=(in_f, self.out_f), initializer="glorot_uniform", trainable=True, name="W")
        self.b = self.add_weight(shape=(self.out_f,), initializer="zeros", trainable=True, name="b")

    def call(self, x, A):
        xw = tf.matmul(x, self.W) + self.b
        return tf.matmul(A, xw)


class STGCBlock(tf.keras.layers.Layer):
    def __init__(self, out_c, tk=3, **kwargs):
        super().__init__(**kwargs)
        self.out_c = out_c
        self.tk = tk

    def build(self, input_shape):
        in_c = input_shape[-1]
        self.gcn = GraphConv(self.out_c, name="gcn")
        self.tcn = tf.keras.layers.Conv1D(filters=self.out_c, kernel_size=self.tk, padding='same', name="tcn")
        self.bn  = tf.keras.layers.BatchNormalization(axis=-1, name="bn")
        if in_c != self.out_c:
            self.residual = tf.keras.layers.Dense(self.out_c, name="residual")
        else:
            self.residual = tf.keras.layers.Lambda(lambda x: x)

    def call(self, x, A):
        shape = tf.shape(x)
        B, T, N, C = shape[0], shape[1], shape[2], shape[3]
        
        x_gcn = tf.reshape(x, (B * T, N, C))
        h = tf.nn.relu(self.gcn(x_gcn, A))
        
        h = tf.reshape(h, (B, T, N, self.out_c))
        h = tf.transpose(h, perm=[0, 2, 1, 3]) 
        h = tf.reshape(h, (B * N, T, self.out_c))
        
        h = self.tcn(h)
        h = self.bn(h)
        
        h = tf.reshape(h, (B, N, T, self.out_c))
        h = tf.transpose(h, perm=[0, 2, 1, 3])
        
        return tf.nn.relu(h + self.residual(x))


class Encoder(tf.keras.layers.Layer):
    def __init__(self, hid, lat, **kwargs):
        super().__init__(**kwargs)
        self.stgc1 = STGCBlock(hid, name="stgc1")
        self.stgc2 = STGCBlock(hid, name="stgc2")
        self.stgc3 = STGCBlock(hid // 2, name="stgc3")
        self.fc_mu     = tf.keras.layers.Dense(lat, name="fc_mu")
        self.fc_logvar = tf.keras.layers.Dense(lat, name="fc_logvar")

    def call(self, x, A):
        h = self.stgc1(x, A)
        h = self.stgc2(h, A)
        h = self.stgc3(h, A)
        h = tf.reshape(h, (tf.shape(h)[0], -1))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, hid, out_c, T, N, **kwargs):
        super().__init__(**kwargs)
        self.T, self.N = T, N
        self.hid = hid
        self.out_c = out_c
        self.fc = tf.keras.layers.Dense(T * N * (hid // 2), name="fc")
        self.stgc1 = STGCBlock(hid, name="stgc1")
        self.stgc2 = STGCBlock(hid, name="stgc2")
        self.gcn_out = GraphConv(out_c, name="gcn_out")

    def call(self, z, A):
        h = tf.nn.relu(self.fc(z))
        half = self.hid // 2
        B = tf.shape(z)[0]
        h = tf.reshape(h, (B, self.T, self.N, half))
        h = self.stgc1(h, A)
        h = self.stgc2(h, A)
        
        shape_h = tf.shape(h)
        h_gcn = tf.reshape(h, (shape_h[0] * shape_h[1], shape_h[2], shape_h[3]))
        out = self.gcn_out(h_gcn, A)
        return tf.reshape(out, (shape_h[0], shape_h[1], shape_h[2], self.out_c))


class STGCVAE(tf.keras.Model):
    def __init__(self, in_c, hid, lat, T, N, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(hid, lat, name="encoder")
        self.decoder = Decoder(hid, in_c, T, N, name="decoder")

    def call(self, x, A):
        mu, lv = self.encoder(x, A)
        eps = tf.random.normal(shape=tf.shape(lv))
        z = mu + tf.exp(0.5 * lv) * eps
        return self.decoder(z, A), mu, lv

    def reconstruction_error(self, x, A):
        mu, _ = self.encoder(x, A)
        xr = self.decoder(mu, A)
        return tf.reduce_mean(tf.square(x - xr), axis=[1, 2, 3])


# Initialize and load model structure
A_HAT = build_adjacency(EDGES, NUM_LANDMARKS)
model = STGCVAE(FEAT_PER_LM, HIDDEN_C, LATENT_DIM, TEMPORAL_LENGTH, NUM_LANDMARKS)
_ = model(tf.zeros((1, TEMPORAL_LENGTH, NUM_LANDMARKS, FEAT_PER_LM)), A_HAT)

try:
    model.load_weights(MODEL_PATH)
    print("TensorFlow model layers and weights initialized successfully.")
except Exception as e:
    print(f"Weight load notification: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PROCESSING PIPELINE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def landmarks_to_array(landmarks) -> np.ndarray:
    arr = np.zeros((NUM_LANDMARKS, FEAT_PER_LM), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        arr[i] = [lm.x, lm.y, lm.z, lm.visibility]
    return arr


def normalise_skeleton_frame(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    hip_mid = (out[23, :3] + out[24, :3]) / 2.0
    sho_mid = (out[11, :3] + out[12, :3]) / 2.0
    torso = np.linalg.norm(sho_mid - hip_mid) + 1e-6
    out[:, :3] = (out[:, :3] - hip_mid) / torso
    return out


def resample_temporal(frames: np.ndarray, target: int) -> np.ndarray:
    T = frames.shape[0]
    if T == target: return frames
    if T < 2: return np.tile(frames, (target, 1, 1))
    flat = frames.reshape(T, -1)
    f = interp1d(np.linspace(0, 1, T), flat, axis=0, kind="linear")
    return f(np.linspace(0, 1, target)).reshape(target, NUM_LANDMARKS, FEAT_PER_LM).astype(np.float32)


def z_normalise(frames: np.ndarray) -> np.ndarray:
    return (frames - GLOBAL_MEAN) / GLOBAL_STD


def validate_rep_with_vae(frames_list: List[np.ndarray]) -> (bool, float):
    seq = np.stack(frames_list)                        
    for t in range(seq.shape[0]):
        seq[t] = normalise_skeleton_frame(seq[t])
    seq = resample_temporal(seq, TEMPORAL_LENGTH)       
    seq = z_normalise(seq)

    tensor = tf.expand_dims(tf.convert_to_tensor(seq, dtype=tf.float32), axis=0)
    error_tensor = model.reconstruction_error(tensor, A_HAT)
    error = float(error_tensor.numpy()[0])
    return error <= VAE_REJECTION_THRESHOLD, error

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  HIP-LINKED DYNAMIC LINE REGION CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class Region(Enum):
    STATIC = 0       
    LOW    = 1       
    PERFECT = 2      
    OVER   = 3       


class DynamicBodyLines:
    def __init__(self):
        self.calibrated = False
        # Anchor offsets relative to the hip midpoint, normalized by torso height
        self.offset_rest_wrist = None
        self.offset_elbow = None
        self.offset_shoulder = None
        self.offset_over = None

    def calibrate(self, frame: np.ndarray):
        hip_mid_y = (frame[23, 1] + frame[24, 1]) / 2.0
        sho_mid_y = (frame[11, 1] + frame[12, 1]) / 2.0
        torso_y = abs(hip_mid_y - sho_mid_y) + 1e-6

        avg_wrist_y = (frame[15, 1] + frame[16, 1]) / 2.0
        avg_elbow_y = (frame[13, 1] + frame[14, 1]) / 2.0

        # Save relative directional scale factors from the hip anchor
        self.offset_rest_wrist = (avg_wrist_y - hip_mid_y) / torso_y
        self.offset_elbow      = (avg_elbow_y - hip_mid_y) / torso_y
        self.offset_shoulder   = (sho_mid_y - hip_mid_y) / torso_y
        # Place the over-extension boundary directly above the shoulder line
        self.offset_over       = self.offset_shoulder - OVER_LINE_OFFSET_TORSO
        self.calibrated = True

    def get_live_lines(self, frame: np.ndarray):
        """Calculates exact absolute positions mapped dynamically to current hip and depth scale."""
        hip_mid_y = (frame[23, 1] + frame[24, 1]) / 2.0
        sho_mid_y = (frame[11, 1] + frame[12, 1]) / 2.0
        torso_y = abs(hip_mid_y - sho_mid_y) + 1e-6

        # Project lines forward from the structural hip anchor location
        rest_y  = hip_mid_y + self.offset_rest_wrist * torso_y
        elbow_y = hip_mid_y + self.offset_elbow * torso_y
        sho_y   = hip_mid_y + self.offset_shoulder * torso_y
        over_y  = hip_mid_y + self.offset_over * torso_y

        # Midpoint definition between hanging wrist position and elbow height
        mid_y = elbow_y + MID_LINE_FACTOR * (rest_y - elbow_y)

        return {"mid": mid_y, "shoulder": sho_y, "over": over_y}

    def classify_region(self, frame: np.ndarray) -> Region:
        if not self.calibrated: 
            return Region.STATIC
        
        lines = self.get_live_lines(frame)
        wrist_y = (frame[15, 1] + frame[16, 1]) / 2.0

        # Note: Screens space 0.0 is top, 1.0 is bottom. Lower value = higher hand raise.
        if wrist_y >= lines["mid"]:
            return Region.STATIC
        elif wrist_y >= lines["shoulder"]:
            return Region.LOW
        elif wrist_y >= lines["over"]:
            return Region.PERFECT
        else:
            return Region.OVER


class RepTracker:
    def __init__(self):
        self.is_recording = False
        self.pre_buffer = collections.deque(maxlen=PRE_BUFFER_SIZE)
        self.recorded_frames: List[np.ndarray] = []
        self.rep_count = 0
        self.reached_perfect = False
        self.static_frame_counter = 0

    def update(self, frame: np.ndarray, region: Region):
        msg = None
        
        if not self.is_recording:
            self.pre_buffer.append(frame.copy())
            # Triggers recording immediately when entering the LOW movement phase
            if region in (Region.LOW, Region.PERFECT, Region.OVER):
                self.is_recording = True
                self.recorded_frames = list(self.pre_buffer)
                self.recorded_frames.append(frame.copy())
                self.reached_perfect = (region == Region.PERFECT)
                self.static_frame_counter = 0
        else:
            self.recorded_frames.append(frame.copy())
            if region == Region.PERFECT:
                self.reached_perfect = True

            if region == Region.STATIC:
                self.static_frame_counter += 1
                if self.static_frame_counter >= STATIC_CONFIRM:
                    self.is_recording = False
                    self.pre_buffer.clear()

                    if self.reached_perfect:
                        is_valid, error = validate_rep_with_vae(self.recorded_frames)
                        if is_valid:
                            self.rep_count += 1
                            msg = f"✓ Rep {self.rep_count} counted! (err={error:.4f})"
                        else:
                            msg = f"✗ Rep rejected by VAE Anomaly Detection (err={error:.4f})"
                    else:
                        msg = "✗ Rep dropped: Did not reach full perfect height extension"
                    
                    self.recorded_frames = []
                    self.reached_perfect = False
            else:
                self.static_frame_counter = 0
        return msg

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  VISUAL DISPLAY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def draw_visual_overlays(img, lines: dict, region: Region, rep_count: int, is_recording: bool, h: int, w: int):
    colors = {"mid": (0, 165, 255), "shoulder": (0, 255, 0), "over": (0, 0, 255)}
    labels = {"mid": "Start Recording Line (Mid Wrist-Elbow)", "shoulder": "Perfect Target Line", "over": "Over-Extension Limit Line"}
    
    for name, y_norm in lines.items():
        y_px = max(0, min(h - 1, int(y_norm * h)))
        cv2.line(img, (0, y_px), (w, y_px), colors[name], 2)
        cv2.putText(img, labels[name], (10, y_px - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[name], 1)

    # State Status Box
    reg_colors = {Region.STATIC: (160, 160, 160), Region.LOW: (0, 165, 255), Region.PERFECT: (0, 255, 0), Region.OVER: (0, 0, 255)}
    reg_texts  = {Region.STATIC: "STATIC (REST)", Region.LOW: "LOW (RECORDING...)", Region.PERFECT: "PERFECT ZONE", Region.OVER: "OVER EXTENSION!"}
    
    cv2.rectangle(img, (w - 280, 10), (w - 10, 50), reg_colors[region], -1)
    cv2.putText(img, reg_texts[region], (w - 270, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Core Counter Info
    cv2.putText(img, f"Reps: {rep_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    status_str = "RECORDING ACTIVE" if is_recording else "AWAITING MOTION"
    cv2.putText(img, f"System: {status_str}", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN RUNTIME LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    body_lines = DynamicBodyLines()
    tracker = RepTracker()

    calibration_frames = 0
    CALIBRATION_COUNT = 30    
    calibration_history = []

    print("Stand straight with your arms completely resting at your sides for system calibration...")

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            t_start = time.time()
            ret, frame_bgr = cap.read()
            if not ret: break

            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                lm_array = landmarks_to_array(results.pose_landmarks.landmark)

                if not body_lines.calibrated:
                    calibration_history.append(lm_array)
                    calibration_frames += 1
                    cv2.putText(frame_bgr, f"Calibrating Anchor Base... {CALIBRATION_COUNT - calibration_frames} frames left", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    if calibration_frames >= CALIBRATION_COUNT:
                        mean_calibration_frame = np.mean(np.stack(calibration_history), axis=0)
                        body_lines.calibrate(mean_calibration_frame)
                        print("System Anchored! Live scaling boundaries active.")
                else:
                    region = body_lines.classify_region(lm_array)
                    lines = body_lines.get_live_lines(lm_array)

                    msg = tracker.update(lm_array, region)
                    if msg: print(msg)

                    draw_visual_overlays(frame_bgr, lines, region, tracker.rep_count, tracker.is_recording, h, w)

                mp_draw.draw_landmarks(frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("TF Balanced Lateral Raise Tracker", frame_bgr)
            elapsed = time.time() - t_start
            wait = max(1, int((1.0 / FPS_TARGET - elapsed) * 1000))
            if cv2.waitKey(wait) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()