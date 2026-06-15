import numpy as np
import tensorflow as tf
import json
from scipy.interpolate import interp1d
from core.feedback_engine import FeedbackType
from .base_engine import BaseWorkoutEngine, FrameResult
from enum import Enum

class Region(Enum):
    STATIC = 0       
    LOW    = 1       
    PERFECT = 2      
    OVER   = 3       

MID_LINE_FACTOR = 0.5
OVER_LINE_OFFSET_TORSO = 0.2

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

class DynamicBodyLines:
    def __init__(self):
        self.calibrated = False
        self.offset_rest_wrist = None
        self.offset_elbow = None
        self.offset_shoulder = None
        self.offset_over = None
        self.calibration_history = []

    def calibrate(self, frame: np.ndarray):
        hip_mid_y = (frame[23, 1] + frame[24, 1]) / 2.0
        sho_mid_y = (frame[11, 1] + frame[12, 1]) / 2.0
        torso_y = abs(hip_mid_y - sho_mid_y) + 1e-6

        avg_wrist_y = (frame[15, 1] + frame[16, 1]) / 2.0
        avg_elbow_y = (frame[13, 1] + frame[14, 1]) / 2.0

        self.offset_rest_wrist = (avg_wrist_y - hip_mid_y) / torso_y
        self.offset_elbow      = (avg_elbow_y - hip_mid_y) / torso_y
        self.offset_shoulder   = (sho_mid_y - hip_mid_y) / torso_y
        self.offset_over       = self.offset_shoulder - OVER_LINE_OFFSET_TORSO
        self.calibrated = True

    def get_live_lines(self, frame: np.ndarray):
        hip_mid_y = (frame[23, 1] + frame[24, 1]) / 2.0
        sho_mid_y = (frame[11, 1] + frame[12, 1]) / 2.0
        torso_y = abs(hip_mid_y - sho_mid_y) + 1e-6

        rest_y  = hip_mid_y + self.offset_rest_wrist * torso_y
        elbow_y = hip_mid_y + self.offset_elbow * torso_y
        sho_y   = hip_mid_y + self.offset_shoulder * torso_y
        over_y  = hip_mid_y + self.offset_over * torso_y

        mid_y = elbow_y + MID_LINE_FACTOR * (rest_y - elbow_y)
        return {"mid": mid_y, "shoulder": sho_y, "over": over_y}

    def classify_region(self, frame: np.ndarray) -> Region:
        if not self.calibrated: 
            return Region.STATIC
        
        lines = self.get_live_lines(frame)
        wrist_y = (frame[15, 1] + frame[16, 1]) / 2.0

        if wrist_y >= lines["mid"]:
            return Region.STATIC
        elif wrist_y >= lines["shoulder"]:
            return Region.LOW
        elif wrist_y >= lines["over"]:
            return Region.PERFECT
        else:
            return Region.OVER

class SideShoulderEngine(BaseWorkoutEngine):
    def initialize(self) -> None:
        with open(self.config.get("config_path")) as f:
            self.cfg = json.load(f)
            
        self.NUM_LANDMARKS = self.cfg["num_landmarks"]
        self.FEAT_PER_LM = self.cfg["features_per_landmark"]
        self.TEMPORAL_LENGTH = self.cfg["temporal_length"]
        self.LATENT_DIM = self.cfg["latent_dim"]
        self.HIDDEN_C = self.cfg["hidden_channels"]
        self.THRESHOLD = 6.8 # Fixed as per user script
        
        self.GLOBAL_MEAN = np.array(self.cfg["global_mean"], dtype=np.float32)
        self.GLOBAL_STD = np.array(self.cfg["global_std"], dtype=np.float32)
        self.EDGES = [tuple(e) for e in self.cfg["adjacency_edges"]]
        
        self.A_HAT = build_adjacency(self.EDGES, self.NUM_LANDMARKS)
        self.model = STGCVAE(self.FEAT_PER_LM, self.HIDDEN_C, self.LATENT_DIM, self.TEMPORAL_LENGTH, self.NUM_LANDMARKS)
        
        # Initialize model structure
        _ = self.model(tf.zeros((1, self.TEMPORAL_LENGTH, self.NUM_LANDMARKS, self.FEAT_PER_LM)), self.A_HAT)
        try:
            self.model.load_weights(self.config.get("model_path"), by_name=True, skip_mismatch=True)
            print("Side Shoulder STGCVAE Model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load weights from {self.config.get('model_path')}. Error: {e}")
        
        self.buffer_kp = []
        self.state = 0 # 0=WAITING, 1=ACTIVE
        self.body_lines = DynamicBodyLines()
        self.current_region = Region.STATIC
        
        import collections
        self.PRE_BUFFER_SIZE = 10
        self.STATIC_CONFIRM = 5
        self.pre_buffer = collections.deque(maxlen=self.PRE_BUFFER_SIZE)
        self.reached_perfect = False
        self.static_frame_counter = 0

    def landmarks_to_array(self, landmarks) -> np.ndarray:
        arr = np.zeros((self.NUM_LANDMARKS, self.FEAT_PER_LM), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            # We assume mediapipe format or simple 4D arrays
            if hasattr(lm, 'x'):
                arr[i] = [lm.x, lm.y, lm.z, lm.visibility]
            else:
                arr[i] = [lm[0], lm[1], lm[2], lm[3]]
        return arr

    def normalise_skeleton_frame(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        hip_mid = (out[23, :3] + out[24, :3]) / 2.0
        sho_mid = (out[11, :3] + out[12, :3]) / 2.0
        torso = np.linalg.norm(sho_mid - hip_mid) + 1e-6
        out[:, :3] = (out[:, :3] - hip_mid) / torso
        return out

    def resample_temporal(self, frames: np.ndarray, target: int) -> np.ndarray:
        T = frames.shape[0]
        if T == target: return frames
        if T < 2: return np.tile(frames, (target, 1, 1))
        flat = frames.reshape(T, -1)
        f = interp1d(np.linspace(0, 1, T), flat, axis=0, kind="linear")
        return f(np.linspace(0, 1, target)).reshape(target, self.NUM_LANDMARKS, self.FEAT_PER_LM).astype(np.float32)

    def z_normalise(self, frames: np.ndarray) -> np.ndarray:
        return (frames - self.GLOBAL_MEAN) / self.GLOBAL_STD

    def validate_rep_with_vae(self, frames_list) -> (bool, float):
        seq = np.stack(frames_list)                        
        for t in range(seq.shape[0]):
            seq[t] = self.normalise_skeleton_frame(seq[t])
        seq = self.resample_temporal(seq, self.TEMPORAL_LENGTH)       
        seq = self.z_normalise(seq)

        tensor = tf.expand_dims(tf.convert_to_tensor(seq, dtype=tf.float32), axis=0)
        error_tensor = self.model.reconstruction_error(tensor, self.A_HAT)
        error = float(error_tensor.numpy()[0])
        return error <= self.THRESHOLD, error

    def process_frame(self, landmarks: list) -> FrameResult:
        lm_array = self.landmarks_to_array(landmarks)
        
        def _angle_3pt(a, b, c):
            a = np.asarray(a, np.float32); b = np.asarray(b, np.float32); c = np.asarray(c, np.float32)
            ba = a - b; bc = c - b
            cos = (ba @ bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            
        R = _angle_3pt(lm_array[24][:3], lm_array[12][:3], lm_array[14][:3])
        L = _angle_3pt(lm_array[23][:3], lm_array[11][:3], lm_array[13][:3])
        cur_M = (R + L) * 0.5
        
        # Handle Dynamic Body Lines calibration and tracking
        if not self.body_lines.calibrated:
            self.body_lines.calibration_history.append(lm_array)
            if len(self.body_lines.calibration_history) >= 30:
                mean_frame = np.mean(np.stack(self.body_lines.calibration_history), axis=0)
                self.body_lines.calibrate(mean_frame)
            self.current_region = Region.STATIC
        else:
            self.current_region = self.body_lines.classify_region(lm_array)
        
        feedback = FeedbackType.NONE
        conf = 0.0
        
        if self.body_lines.calibrated:
            if self.state == 0: # WAITING
                self.pre_buffer.append(lm_array)
                # Start recording when user enters LOW, PERFECT, or OVER region
                if self.current_region in (Region.LOW, Region.PERFECT, Region.OVER):
                    self.state = 1
                    self.buffer_kp = list(self.pre_buffer)
                    self.buffer_kp.append(lm_array)
                    self.reached_perfect = (self.current_region == Region.PERFECT)
                    self.static_frame_counter = 0
            
            elif self.state == 1: # RECORDING
                self.buffer_kp.append(lm_array)
                
                if self.current_region == Region.PERFECT:
                    self.reached_perfect = True
                elif self.current_region == Region.OVER:
                    # Real-time warning: arms too high
                    feedback = FeedbackType.OVER_RANGE
                
                # End recording when user returns to STATIC for a few frames
                if self.current_region == Region.STATIC:
                    self.static_frame_counter += 1
                    if self.static_frame_counter >= self.STATIC_CONFIRM:
                        # Rep finished, evaluate it!
                        if self.reached_perfect:
                            is_valid, error = self.validate_rep_with_vae(self.buffer_kp)
                            self.last_vae_error = float(error)
                            if is_valid:
                                self.rep_count_internal += 1
                                feedback = FeedbackType.PERFECT
                            else:
                                feedback = FeedbackType.REJECTED_BY_VAE
                        else:
                            feedback = FeedbackType.LOWER_RANGE

                        # Reset state
                        self.state = 0
                        self.buffer_kp = []
                        self.pre_buffer.clear()
                        self.reached_perfect = False
                        self.static_frame_counter = 0
                else:
                    self.static_frame_counter = 0

        details = {"angle": float(cur_M)}
        if hasattr(self, "last_vae_error"):
            details["vae_error"] = self.last_vae_error
            details["threshold"] = self.THRESHOLD
            if self.state == 0 and feedback == FeedbackType.NONE:
                delattr(self, "last_vae_error")

        return FrameResult(
            rep_count=self.rep_count_internal,
            feedback=feedback,
            confidence=conf,
            is_recording=(self.state == 1),
            details=details
        )

    def draw_custom_visuals(self, render_frame, landmarks: list) -> None:
        import cv2
        h, w, _ = render_frame.shape
        
        lm_array = self.landmarks_to_array(landmarks)
        
        def lm_px(idx):
            return (int(lm_array[idx][0] * w), int(lm_array[idx][1] * h))
            
        def _angle_3pt(a, b, c):
            a = np.asarray(a, np.float32); b = np.asarray(b, np.float32); c = np.asarray(c, np.float32)
            ba = a - b; bc = c - b
            cos = (ba @ bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            
        try:
            # Side shoulder tracks shoulder abduction
            R = _angle_3pt(lm_array[24][:3], lm_array[12][:3], lm_array[14][:3])
            L = _angle_3pt(lm_array[23][:3], lm_array[11][:3], lm_array[13][:3])
            
            r_sho_px = lm_px(12)
            l_sho_px = lm_px(11)
            
            color = (255, 165, 0) # Orange
            cv2.putText(render_frame, f"{R:.0f}", (r_sho_px[0] + 10, r_sho_px[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(render_frame, f"{R:.0f}", (r_sho_px[0] + 10, r_sho_px[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)
            
            cv2.putText(render_frame, f"{L:.0f}", (l_sho_px[0] - 40, l_sho_px[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(render_frame, f"{L:.0f}", (l_sho_px[0] - 40, l_sho_px[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)
            
            # Draw Dynamic Body Lines UI
            if hasattr(self, 'body_lines') and self.body_lines.calibrated:
                lines = self.body_lines.get_live_lines(lm_array)
                region = getattr(self, 'current_region', Region.STATIC)
                
                colors = {"mid": (0, 165, 255), "shoulder": (0, 255, 0), "over": (0, 0, 255)}
                labels = {"mid": "Start Recording Line (Mid Wrist-Elbow)", "shoulder": "Perfect Target Line", "over": "Over-Extension Limit Line"}
                
                for name, y_norm in lines.items():
                    y_px = max(0, min(h - 1, int(y_norm * h)))
                    cv2.line(render_frame, (0, y_px), (w, y_px), colors[name], 2)
                    cv2.putText(render_frame, labels[name], (10, y_px - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[name], 1)

                # State Status Box
                reg_colors = {Region.STATIC: (160, 160, 160), Region.LOW: (0, 165, 255), Region.PERFECT: (0, 255, 0), Region.OVER: (0, 0, 255)}
                reg_texts  = {Region.STATIC: "STATIC (REST)", Region.LOW: "LOW (RECORDING...)", Region.PERFECT: "PERFECT ZONE", Region.OVER: "OVER EXTENSION!"}
                
                cv2.rectangle(render_frame, (w - 280, 10), (w - 10, 50), reg_colors[region], -1)
                cv2.putText(render_frame, reg_texts[region], (w - 270, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            elif hasattr(self, 'body_lines') and not self.body_lines.calibrated:
                frames_left = 30 - len(self.body_lines.calibration_history)
                cv2.putText(render_frame, f"Calibrating Anchor Base... {frames_left} frames left", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        except Exception as e:
            print("Error drawing custom side shoulder visuals:", e)
