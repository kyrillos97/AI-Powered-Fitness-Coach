#!/usr/bin/env python3
"""
realtime_front_shoulder.py  –  Real-time front shoulder raise tracker.

Updated Version:
- Reference lines are anchored to the Nose keypoint after calibration.
- Line 1: 25% from hands to elbow (Low range boundary).
- Line 2: 50% from elbow to shoulder (Perfect range boundary / Recording trigger).
- Line 3: Above shoulders (Over range boundary).
- Configuration parameters moved to the top.
"""

import json, time, collections
from enum import Enum
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION (Adjustable parameters)
# ═══════════════════════════════════════════════════════════════════════════════

# File Paths
CONFIG_PATH = "vae_config_front_shoulder.json"
MODEL_PATH  = "st_gcvae_front_shoulder.pt"

# Range Configuration
LOW_RANGE_PCT = 0.25          # 25% from hands to elbow (Line 1)
PERFECT_RANGE_PCT = 0.50      # 50% from elbow to shoulder (Line 2 / Perfect Area Start)
OVER_SHOULDER_OFFSET = 0.05   # Offset above shoulder for Over Range (Line 3)

# VAE & Recording
VAE_REJECTION_THRESHOLD = 120 # Strictness for VAE rejection
PRE_BUFFER_SIZE = 10          # Frames kept before region trigger
FPS_TARGET = 10               # Target processing FPS
STATIC_CONFIRM = 5            # Frames of STATIC needed to confirm rep end
CALIBRATION_COUNT = 30        # Frames at rest for calibration

# Load VAE Config to get constants
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

NUM_LANDMARKS   = CFG["num_landmarks"]       # 33
FEAT_PER_LM     = CFG["features_per_landmark"]  # 4
TEMPORAL_LENGTH  = CFG["temporal_length"]     # 20
LATENT_DIM      = CFG["latent_dim"]          # 32
HIDDEN_C        = CFG["hidden_channels"]     # 64
THRESHOLD       = CFG["threshold"]

GLOBAL_MEAN     = np.array(CFG["global_mean"], dtype=np.float32)   # (33, 4)
GLOBAL_STD      = np.array(CFG["global_std"],  dtype=np.float32)   # (33, 4)
EDGES           = [tuple(e) for e in CFG["adjacency_edges"]]

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

def build_adjacency(edges, num_nodes):
    A = np.eye(num_nodes, dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = np.diag(A.sum(axis=1) ** -0.5)
    return torch.tensor(D @ A @ D, dtype=torch.float32)

class GraphConv(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.W = nn.Linear(in_f, out_f, bias=True)

    def forward(self, x, A):
        return torch.matmul(A, self.W(x))

class STGCBlock(nn.Module):
    def __init__(self, in_c, out_c, tk=3):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c)
        self.tcn = nn.Conv1d(out_c, out_c, tk, padding=tk // 2)
        self.bn  = nn.BatchNorm1d(out_c)
        self.residual = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()

    def forward(self, x, A):
        B, T, N, C = x.shape
        h = F.relu(self.gcn(x.reshape(B*T, N, C), A))
        Co = h.shape[-1]
        h = h.reshape(B, T, N, Co).permute(0, 2, 3, 1).reshape(B*N, Co, T)
        h = self.bn(self.tcn(h))
        h = h.reshape(B, N, Co, T).permute(0, 3, 1, 2)
        return F.relu(h + self.residual(x))

class Encoder(nn.Module):
    def __init__(self, in_c, hid, lat, T, N):
        super().__init__()
        self.stgc1 = STGCBlock(in_c, hid)
        self.stgc2 = STGCBlock(hid, hid)
        self.stgc3 = STGCBlock(hid, hid // 2)
        flat = T * N * (hid // 2)
        self.fc_mu     = nn.Linear(flat, lat)
        self.fc_logvar = nn.Linear(flat, lat)

    def forward(self, x, A):
        h = self.stgc3(self.stgc2(self.stgc1(x, A), A), A)
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, lat, hid, out_c, T, N):
        super().__init__()
        self.T, self.N = T, N
        half = hid // 2
        self.fc = nn.Linear(lat, T * N * half)
        self.stgc1 = STGCBlock(half, hid)
        self.stgc2 = STGCBlock(hid, hid)
        self.gcn_out = GraphConv(hid, out_c)

    def forward(self, z, A):
        h = F.relu(self.fc(z))
        half = h.shape[-1] // (self.T * self.N)
        h = h.reshape(-1, self.T, self.N, half)
        h = self.stgc2(self.stgc1(h, A), A)
        B, T, N, C = h.shape
        out = self.gcn_out(h.reshape(B*T, N, C), A)
        return out.reshape(B, T, N, -1)

class STGCVAE(nn.Module):
    def __init__(self, in_c, hid, lat, T, N):
        super().__init__()
        self.encoder = Encoder(in_c, hid, lat, T, N)
        self.decoder = Decoder(lat, hid, in_c, T, N)

    def forward(self, x, A):
        mu, lv = self.encoder(x, A)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(lv)
        return self.decoder(z, A), mu, lv

    def reconstruction_error(self, x, A):
        with torch.no_grad():
            mu, _ = self.encoder(x, A)
            xr = self.decoder(mu, A)
            return ((x - xr) ** 2).mean(dim=(1, 2, 3))

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  LOAD TRAINED MODEL
# ═══════════════════════════════════════════════════════════════════════════════

device = torch.device("cpu")
A_HAT = build_adjacency(EDGES, NUM_LANDMARKS).to(device)

model = STGCVAE(FEAT_PER_LM, HIDDEN_C, LATENT_DIM, TEMPORAL_LENGTH, NUM_LANDMARKS).to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded.")
except Exception as e:
    print(f"Warning: Could not load model from {MODEL_PATH}. Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  HELPER FUNCTIONS
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
    if T == target:
        return frames
    if T < 2:
        return np.tile(frames, (target, 1, 1))
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
    tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    error = model.reconstruction_error(tensor, A_HAT).item()
    return error <= VAE_REJECTION_THRESHOLD, error

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  REGION CLASSIFICATION (Nose-Anchored Lines)
# ═══════════════════════════════════════════════════════════════════════════════

class Region(Enum):
    STATIC  = 0  # Below Line 1
    LOW     = 1  # Between Line 1 and Line 2
    PERFECT = 2  # Between Line 2 and Line 3
    OVER    = 3  # Above Line 3

class BodyLines:
    """
    Computes reference lines anchored to the Nose keypoint.
    Line 1: 25% from wrist to elbow.
    Line 2: 50% from elbow to shoulder (Perfect Area Start).
    Line 3: Over shoulders.
    """
    def __init__(self):
        self.calibrated = False
        self.offsets = {
            "low": None,      # Line 1 offset from Nose
            "perfect": None,  # Line 2 offset from Nose
            "over": None      # Line 3 offset from Nose
        }

    def calibrate(self, nose_y, sho_y, elb_y, wri_y):
        """Calculate and store offsets relative to the nose during calibration."""
        # Line 1: 25% from wrist to elbow
        dist_we = wri_y - elb_y
        line1_y = wri_y - (LOW_RANGE_PCT * dist_we)
        
        # Line 2: 50% from elbow to shoulder
        dist_es = elb_y - sho_y
        line2_y = elb_y - (PERFECT_RANGE_PCT * dist_es)
        
        # Line 3: Over shoulders
        line3_y = sho_y - OVER_SHOULDER_OFFSET
        
        # Store offsets: (Target_Y - Nose_Y)
        self.offsets["low"] = line1_y - nose_y
        self.offsets["perfect"] = line2_y - nose_y
        self.offsets["over"] = line3_y - nose_y
        self.calibrated = True

    def get_lines(self, current_nose_y):
        """Returns absolute Y coordinates based on current nose position."""
        if not self.calibrated:
            return None
        return {
            "low": current_nose_y + self.offsets["low"],
            "perfect": current_nose_y + self.offsets["perfect"],
            "over": current_nose_y + self.offsets["over"]
        }

    def classify_region(self, frame: np.ndarray) -> Region:
        """Classify current wrist position into a region using nose-anchored lines."""
        if not self.calibrated:
            return Region.STATIC
            
        nose_y = frame[0, 1]
        wrist_y = (frame[15, 1] + frame[16, 1]) / 2.0
        lines = self.get_lines(nose_y)

        # MediaPipe Y increases downward: lower Y = higher on screen
        if wrist_y > lines["low"]:
            return Region.STATIC
        elif wrist_y > lines["perfect"]:
            return Region.LOW
        elif wrist_y > lines["over"]:
            return Region.PERFECT
        else:
            return Region.OVER

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  REP STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

class RepState(Enum):
    IDLE          = 0
    RECORDING     = 1

class RepTracker:
    def __init__(self):
        self.state = RepState.IDLE
        self.pre_buffer = collections.deque(maxlen=PRE_BUFFER_SIZE)
        self.recorded_frames: List[np.ndarray] = []
        self.rep_count = 0
        self.reached_perfect = False
        self.frames_in_static = 0

    def update(self, frame: np.ndarray, region: Region):
        msg = None

        if self.state == RepState.IDLE:
            self.pre_buffer.append(frame.copy())
            # Start recording when entering PERFECT region (as per "third line" request)
            if region in (Region.PERFECT, Region.OVER):
                self.state = RepState.RECORDING
                self.recorded_frames = list(self.pre_buffer)
                self.recorded_frames.append(frame.copy())
                self.reached_perfect = (region == Region.PERFECT)
                self.frames_in_static = 0

        elif self.state == RepState.RECORDING:
            self.recorded_frames.append(frame.copy())
            if region == Region.PERFECT:
                self.reached_perfect = True

            if region == Region.STATIC:
                self.frames_in_static += 1
                if self.frames_in_static >= STATIC_CONFIRM:
                    self.state = RepState.IDLE
                    self.pre_buffer.clear()
                    if self.reached_perfect:
                        is_valid, error = validate_rep_with_vae(self.recorded_frames)
                        if is_valid:
                            self.rep_count += 1
                            msg = f"✓ Rep {self.rep_count} counted! (err={error:.4f})"
                        else:
                            msg = f"✗ Motion rejected by VAE (err={error:.4f})"
                    else:
                        msg = "✗ Did not reach perfect range"
                    self.recorded_frames = []
                    self.reached_perfect = False
            else:
                self.frames_in_static = 0

        return msg

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_lines(img, lines: dict, h: int, w: int):
    colors = {
        "low":      (0,   165, 255),  # Orange
        "perfect":  (0,   255,   0),  # Green
        "over":     (0,   0,   255),  # Red
    }
    labels = {
        "low":      "Line 1: Low Range Start",
        "perfect":  "Line 2: Perfect Start / Recording",
        "over":     "Line 3: Over Range",
    }
    for name, y_norm in lines.items():
        y_px = int(y_norm * h)
        y_px = max(0, min(h - 1, y_px))
        cv2.line(img, (0, y_px), (w, y_px), colors[name], 2)
        cv2.putText(img, labels[name], (10, y_px - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 1)

def draw_region_indicator(img, region: Region, h: int, w: int):
    color_map = {
        Region.STATIC:  (180, 180, 180),
        Region.LOW:     (0, 165, 255),
        Region.PERFECT: (0, 255, 0),
        Region.OVER:    (0, 0, 255),
    }
    text_map = {
        Region.STATIC:  "STATIC",
        Region.LOW:     "LOW - Raise more!",
        Region.PERFECT: "PERFECT",
        Region.OVER:    "OVER - Lower down!",
    }
    color = color_map[region]
    cv2.rectangle(img, (w - 300, 10), (w - 10, 50), color, -1)
    cv2.putText(img, text_map[region], (w - 290, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    body_lines = BodyLines()
    tracker = RepTracker()

    calibration_frames = 0
    calibration_data = []

    print("Stand still with arms at your sides for calibration…")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as pose:

        while cap.isOpened():
            t_start = time.time()
            ret, frame_bgr = cap.read()
            if not ret:
                break

            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                lm_array = landmarks_to_array(results.pose_landmarks.landmark)

                if not body_lines.calibrated:
                    # Calibration: collect nose, shoulder, elbow, and wrist positions
                    nose_y = lm_array[0, 1]
                    sho_y = (lm_array[11, 1] + lm_array[12, 1]) / 2.0
                    elb_y = (lm_array[13, 1] + lm_array[14, 1]) / 2.0
                    wri_y = (lm_array[15, 1] + lm_array[16, 1]) / 2.0
                    
                    calibration_data.append([nose_y, sho_y, elb_y, wri_y])
                    calibration_frames += 1
                    cv2.putText(frame_bgr,
                                f"Calibrating… {CALIBRATION_COUNT - calibration_frames} frames left",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    if calibration_frames >= CALIBRATION_COUNT:
                        avg = np.mean(calibration_data, axis=0)
                        body_lines.calibrate(avg[0], avg[1], avg[2], avg[3])
                        print(f"Calibrated! Lines anchored to nose.")

                else:
                    # Normal tracking
                    region = body_lines.classify_region(lm_array)
                    lines = body_lines.get_lines(lm_array[0, 1])

                    msg = tracker.update(lm_array, region)
                    if msg:
                        print(msg)

                    draw_lines(frame_bgr, lines, h, w)
                    draw_region_indicator(frame_bgr, region, h, w)

                    cv2.putText(frame_bgr, f"Reps: {tracker.rep_count}",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    cv2.putText(frame_bgr, f"State: {tracker.state.name}",
                                (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                mp_draw.draw_landmarks(frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Front Shoulder Raise Tracker", frame_bgr)

            elapsed = time.time() - t_start
            wait = max(1, int((1.0 / FPS_TARGET - elapsed) * 1000))
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Total valid reps: {tracker.rep_count}")

if __name__ == "__main__":
    main()
