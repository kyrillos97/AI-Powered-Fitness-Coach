"""
=============================================================================
REAL-TIME FRONT SHOULDER RAISE FORM ASSESSMENT
=============================================================================
TSM (Temporal Self-Similarity Matrix) based rep detection using VAE latents.

Architecture:
  Camera → MediaPipe → 8 biomechanical features → VAE encoder → latent z_t
  → Rolling TSM buffer → Cosine similarity signal → Autocorrelation → Period
  → Peak detection on similarity signal → REP DETECTED
  → Extract window → Conv1D classifier → Form feedback

NO state machine. Detection is purely from the temporal self-similarity 
structure in VAE latent space.

Tested with:
  Python 3.10.19 | mediapipe 0.10.32 | onnxruntime 1.18.0

Files needed (same folder):
  - vae_model.onnx
  - conv1d_classifier.onnx
  - mobile_config.json
  - pose_landmarker_full.task  (auto-downloaded if missing)
=============================================================================
"""

import cv2
import numpy as np
import json
import time
import os
import sys
import urllib.request
from collections import deque
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

import onnxruntime as ort

# =============================================================================
# PATHS & CONSTANTS
# =============================================================================

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) if '__file__' in dir() else Path('.')

CONFIG_PATH = SCRIPT_DIR / 'mobile_config.json'
VAE_ONNX_PATH = SCRIPT_DIR / 'vae_model.onnx'
CONV1D_ONNX_PATH = SCRIPT_DIR / 'conv1d_classifier.onnx'
POSE_MODEL_PATH = SCRIPT_DIR / 'pose_landmarker_full.task'

CAMERA_INDEX = 1
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

COLORS = {
    'perfect':    (0, 200, 0),
    'over_range': (0, 0, 255),
    'low':        (0, 165, 255),
    'bent_elbow': (255, 0, 200),
    'idle':       (200, 200, 200),
    'unknown':    (128, 128, 128),
}

FEEDBACK_MESSAGES = {
    'perfect':    "Perfect Form!",
    'over_range': "Too High! Lower your arm",
    'low':        "Go Higher! Raise to shoulder level",
    'bent_elbow': "Straighten Your Elbow!",
    'unknown':    "Could not classify",
}

# =============================================================================
# DOWNLOAD POSE MODEL
# =============================================================================

def download_pose_model():
    if POSE_MODEL_PATH.exists():
        return
    print(f"  Downloading pose landmarker model...")
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "pose_landmarker/pose_landmarker_full/float16/latest/"
           "pose_landmarker_full.task")
    urllib.request.urlretrieve(url, str(POSE_MODEL_PATH))
    print(f"  ✓ Downloaded ({POSE_MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

# =============================================================================
# LOAD CONFIG
# =============================================================================

print("=" * 70)
print("  LOADING CONFIGURATION")
print("=" * 70)

if not CONFIG_PATH.exists():
    print(f"  ✗ Config not found: {CONFIG_PATH}")
    sys.exit(1)

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

LANDMARKS = config['landmarks']
RS = LANDMARKS['right_shoulder']
RE = LANDMARKS['right_elbow']
RW = LANDMARKS['right_wrist']
RH = LANDMARKS['right_hip']
LS = LANDMARKS['left_shoulder']
LH = LANDMARKS['left_hip']

VAE_MEANS = np.array(config['vae_feature_means'], dtype=np.float32)
VAE_STDS = np.array(config['vae_feature_stds'], dtype=np.float32)
VAE_LATENT_DIM = config['vae_latent_dim']

CONV1D_WINDOW = config['conv1d_window_size']
CONV1D_CHANNELS = config['conv1d_total_channels']
CONV1D_MEANS = np.array(config['conv1d_channel_means'], dtype=np.float32)
CONV1D_STDS = np.array(config['conv1d_channel_stds'], dtype=np.float32)

ANOMALY_THRESHOLD = config['anomaly_threshold']
CLASS_LABELS = config['class_labels']

# TSM-specific config
rep_cfg = config['rep_detection']
MIN_REP_FRAMES = rep_cfg['min_rep_frames']
MAX_REP_FRAMES = rep_cfg['max_rep_frames']
COOLDOWN_FRAMES = rep_cfg['cooldown_frames']

# TSM parameters
TSM_BUFFER_SIZE = 300          # rolling buffer of latent vectors
TSM_SIMILARITY_SMOOTH = 5     # smooth the similarity signal
TSM_PERIOD_MIN = MIN_REP_FRAMES
TSM_PERIOD_MAX = MAX_REP_FRAMES
TSM_AUTOCORR_UPDATE_INTERVAL = 15  # re-estimate period every N frames
TSM_PEAK_PROMINENCE = 0.15    # minimum prominence for similarity peak
TSM_PEAK_MIN_HEIGHT = 0.3     # minimum height of similarity peak to count as rep boundary
TSM_CONFIDENCE_THRESHOLD = 0.08  # minimum autocorrelation confidence

print(f"  ✓ Config loaded")
print(f"    Classes: {CLASS_LABELS}")
print(f"    VAE latent dim: {VAE_LATENT_DIM}")
print(f"    TSM buffer: {TSM_BUFFER_SIZE} frames")
print(f"    Period range: [{TSM_PERIOD_MIN}, {TSM_PERIOD_MAX}] frames")

# =============================================================================
# LOAD ONNX MODELS
# =============================================================================

print("\n  Loading ONNX models...")

sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.intra_op_num_threads = 4

vae_session = None
if VAE_ONNX_PATH.exists():
    vae_session = ort.InferenceSession(str(VAE_ONNX_PATH), sess_opts,
                                        providers=['CPUExecutionProvider'])
    VAE_INPUT_NAME = vae_session.get_inputs()[0].name
    print(f"  ✓ VAE ONNX loaded (input: '{VAE_INPUT_NAME}')")
else:
    print(f"  ✗ VAE not found: {VAE_ONNX_PATH}")
    sys.exit(1)

conv1d_session = None
if CONV1D_ONNX_PATH.exists():
    conv1d_session = ort.InferenceSession(str(CONV1D_ONNX_PATH), sess_opts,
                                           providers=['CPUExecutionProvider'])
    CONV1D_INPUT_NAME = conv1d_session.get_inputs()[0].name
    print(f"  ✓ Conv1D ONNX loaded (input: '{CONV1D_INPUT_NAME}')")
else:
    print(f"  ✗ Conv1D not found: {CONV1D_ONNX_PATH}")
    sys.exit(1)

# =============================================================================
# DOWNLOAD MEDIAPIPE MODEL
# =============================================================================

print("\n  Setting up MediaPipe...")
download_pose_model()

# =============================================================================
# BIOMECHANICAL ANGLE COMPUTATION
# =============================================================================

def get_landmark_xyz(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def angle_at_b(a, b, c):
    """Angle at point b in degrees."""
    ba = a - b
    bc = c - b
    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_val = np.dot(ba, bc) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def compute_angles(landmarks):
    """Compute 4 biomechanical angles from pose landmarks."""
    shoulder = get_landmark_xyz(landmarks, RS)
    elbow = get_landmark_xyz(landmarks, RE)
    wrist = get_landmark_xyz(landmarks, RW)
    hip = get_landmark_xyz(landmarks, RH)
    l_shoulder = get_landmark_xyz(landmarks, LS)

    arm_elev = angle_at_b(hip, shoulder, wrist)
    elbow_ang = angle_at_b(shoulder, elbow, wrist)

    vert = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    tv = shoulder - hip
    n_tv = np.linalg.norm(tv)
    torso = float(np.degrees(np.arccos(np.clip(
        np.dot(vert, tv) / (n_tv + 1e-8), -1.0, 1.0)))) if n_tv > 1e-8 else 0.0

    s_diff = float(l_shoulder[1] - shoulder[1])

    return {
        'primary_arm_elevation': arm_elev,
        'primary_elbow_angle': elbow_ang,
        'torso_lean': torso,
        'shoulder_height_diff': s_diff,
    }


# =============================================================================
# VAE INFERENCE
# =============================================================================

def run_vae(features_8d):
    """Run VAE: returns (latent_mu, reconstruction_error)."""
    scaled = ((features_8d - VAE_MEANS) / VAE_STDS).reshape(1, -1).astype(np.float32)
    outputs = vae_session.run(None, {VAE_INPUT_NAME: scaled})
    recon = outputs[0][0]
    mu = outputs[1][0]
    recon_error = float(np.mean((recon - scaled[0]) ** 2))
    return mu.astype(np.float32), recon_error


# =============================================================================
# CONV1D CLASSIFIER
# =============================================================================

def pad_or_truncate(seq, target_len):
    n = len(seq)
    if n >= target_len:
        start = (n - target_len) // 2
        return seq[start:start + target_len]
    else:
        pad = np.tile(seq[-1:], (target_len - n, 1))
        return np.vstack([seq, pad])


def classify_rep_conv1d(window_data):
    """
    Classify rep from temporal window.
    window_data: (T, channels) array of [feat8 + latents]
    Returns: (class_name, confidence, probs)
    """
    if window_data.shape[0] < 3:
        return 'unknown', 0.0, np.zeros(len(CLASS_LABELS), dtype=np.float32)

    padded = pad_or_truncate(window_data, CONV1D_WINDOW)
    normalized = ((padded - CONV1D_MEANS) / CONV1D_STDS).astype(np.float32)
    batch = normalized.reshape(1, CONV1D_WINDOW, CONV1D_CHANNELS)

    logits = conv1d_session.run(None, {CONV1D_INPUT_NAME: batch})[0][0]

    shifted = logits - np.max(logits)
    exp_l = np.exp(shifted)
    probs = exp_l / (exp_l.sum() + 1e-8)

    idx = int(np.argmax(probs))
    cls = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else 'unknown'
    return cls, float(probs[idx]), probs


# =============================================================================
# TSM REP DETECTOR — The core of the architecture
# =============================================================================

class TSMRepDetector:
    """
    Temporal Self-Similarity Matrix based rep detector.
    
    How it works (per frame):
    1. VAE encodes frame → latent z_t
    2. z_t is added to rolling buffer of latent vectors
    3. Compute cosine similarity: s(k) = sim(z_t, z_{t-k}) for all k in buffer
       This gives us ONE ROW of the self-similarity matrix
    4. Periodically run autocorrelation on the accumulated similarity signal
       to estimate the rep period P
    5. Once P is known, monitor the similarity signal at lag P:
       When sim(z_t, z_{t-P}) peaks → current frame is at same phase as P frames ago
       → A full cycle (rep) has completed
    6. Detect peaks in the lag-P similarity signal → each peak = one rep boundary
    
    This is a direct online implementation of the RepNet (Dwibedi et al. 2020)
    concept using VAE latents instead of learned embeddings.
    """
    
    def __init__(self):
        # Rolling buffer of latent vectors
        self.latent_buffer = deque(maxlen=TSM_BUFFER_SIZE)
        
        # Rolling similarity signals at various lags
                # sim_at_lag[k] = history of cosine_similarity(z_t, z_{t-k})
        self.sim_at_estimated_period = deque(maxlen=TSM_BUFFER_SIZE)
        
        # Full similarity row for current frame (for autocorrelation)
        self.similarity_rows = deque(maxlen=TSM_BUFFER_SIZE)
        
        # Period estimation
        self.estimated_period = None
        self.period_confidence = 0.0
        self.period_history = deque(maxlen=10)  # recent period estimates for stability
        
        # Peak detection on similarity signal
        self.rep_count = 0
        self.last_rep_frame = -999
        self.frame_count = 0
        
        # Data buffers for classification (keep raw features + latents)
        self.feature8_buffer = deque(maxlen=TSM_BUFFER_SIZE)
        self.latent_for_class_buffer = deque(maxlen=TSM_BUFFER_SIZE)
        
        # Track the similarity signal for peak detection
        self.sim_signal_smooth = deque(maxlen=TSM_BUFFER_SIZE)
        
        # For detecting rising/falling in similarity signal
        self.sim_was_above_threshold = False
        self.sim_peak_val = 0.0
        self.sim_peak_frame = 0
        self.sim_trough_since_last_rep = 0.0
        self.saw_trough = False
        
        # Previous angles for velocity
        self.prev_angles = None
        
        # Recon error for anomaly gate
        self.last_recon_error = 0.0
        
        # History
        self.rep_history = []
        
        # Debug info
        self.debug_info = {
            'period': None,
            'confidence': 0.0,
            'sim_value': 0.0,
            'sim_smooth': 0.0,
            'buffer_len': 0,
        }
    
    def _cosine_similarity(self, a, b):
        """Cosine similarity between two vectors."""
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(dot / (na * nb))
    
    def _compute_similarity_row(self, z_t):
        """
        Compute similarity between z_t and all previous latents in buffer.
        Returns array where s[k] = cosine_sim(z_t, z_{t-k})
        s[0] = 1.0 (self-similarity)
        """
        n = len(self.latent_buffer)
        if n == 0:
            return np.array([1.0])
        
        # Vectorized cosine similarity
        buffer_arr = np.array(self.latent_buffer)  # (n, latent_dim)
        z_t_norm = z_t / (np.linalg.norm(z_t) + 1e-8)
        norms = np.linalg.norm(buffer_arr, axis=1, keepdims=True) + 1e-8
        buffer_normed = buffer_arr / norms
        
        sims = buffer_normed @ z_t_norm  # (n,)
        
        # Reverse so index 0 = most recent (lag 0), index n-1 = oldest
        sims = sims[::-1]
        return sims.astype(np.float32)
    
    def _estimate_period_autocorrelation(self):
        """
        Run autocorrelation on accumulated similarity rows to find the dominant period.
        
        We take the mean similarity at each lag across recent frames,
        then find peaks in the autocorrelation.
        """
        if len(self.similarity_rows) < TSM_PERIOD_MIN * 2:
            return None, 0.0
        
        # Build mean similarity profile across lags
        # Each row in similarity_rows has different length (grows over time)
        # Use the last N rows and compute mean sim at each lag
        recent_rows = list(self.similarity_rows)[-TSM_BUFFER_SIZE:]
        
        max_lag = min(TSM_PERIOD_MAX + 20, min(len(r) for r in recent_rows[-50:]) if len(recent_rows) >= 50 else min(len(r) for r in recent_rows))
        
        if max_lag < TSM_PERIOD_MIN:
            return None, 0.0
        
        # Compute mean similarity at each lag from recent rows
        n_rows = min(60, len(recent_rows))
        mean_sim = np.zeros(max_lag)
        count = np.zeros(max_lag)
        
        for row in recent_rows[-n_rows:]:
            usable = min(len(row), max_lag)
            mean_sim[:usable] += row[:usable]
            count[:usable] += 1
        
        count[count == 0] = 1
        mean_sim = mean_sim / count
        
        # Autocorrelation of the mean similarity profile
        if len(mean_sim) < TSM_PERIOD_MIN + 5:
            return None, 0.0
        
        # Normalize
        mean_sim_centered = mean_sim - mean_sim.mean()
        energy = np.sum(mean_sim_centered ** 2)
        if energy < 1e-8:
            return None, 0.0
        
        # Compute autocorrelation via direct method (faster for short signals)
        autocorr = np.correlate(mean_sim_centered, mean_sim_centered, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / (autocorr[0] + 1e-8)
        
        # Find peaks in the valid period range
        search_start = TSM_PERIOD_MIN
        search_end = min(TSM_PERIOD_MAX, len(autocorr) - 1)
        
        if search_end <= search_start:
            return None, 0.0
        
        segment = autocorr[search_start:search_end]
        
        # Find local maxima
        peaks = []
        for i in range(1, len(segment) - 1):
            if segment[i] > segment[i-1] and segment[i] > segment[i+1]:
                if segment[i] > TSM_CONFIDENCE_THRESHOLD:
                    peaks.append((i + search_start, segment[i]))
        
        if len(peaks) == 0:
            return None, 0.0
        
        # Take the peak with highest autocorrelation value
        best_peak = max(peaks, key=lambda x: x[1])
        period = best_peak[0]
        confidence = best_peak[1]
        
        return period, confidence
    
    def _smooth_value(self, val, buffer, window=TSM_SIMILARITY_SMOOTH):
        """Add value to buffer and return smoothed (moving average)."""
        buffer.append(val)
        if len(buffer) < window:
            return float(np.mean(list(buffer)))
        return float(np.mean(list(buffer)[-window:]))
    
    def _detect_rep_from_similarity(self, sim_smooth, frame):
        """
        Detect if a rep just completed based on the similarity-at-estimated-period signal.
        
        Logic: The similarity at lag P oscillates. It peaks when the current frame
        is at the same phase as P frames ago (= one full cycle completed).
        
        We detect a rep when:
        1. The smoothed similarity signal rises above a threshold (entering "similar" zone)
        2. Then falls below (leaving "similar" zone → passed through the peak)
        3. The peak value was high enough
        4. We saw a trough before the peak (confirming a full cycle)
        """
        # Adaptive threshold: use 60% of the running mean of peak values
        if len(self.sim_signal_smooth) > 20:
            recent = np.array(list(self.sim_signal_smooth)[-60:])
            threshold = np.median(recent) + 0.15 * (np.max(recent) - np.median(recent))
            threshold = max(threshold, TSM_PEAK_MIN_HEIGHT)
        else:
            threshold = TSM_PEAK_MIN_HEIGHT
        
        # Cooldown
        if (frame - self.last_rep_frame) < COOLDOWN_FRAMES:
            return False
        
        # Track if signal crossed above threshold
        is_above = sim_smooth > threshold
        
        if is_above:
            if sim_smooth > self.sim_peak_val:
                self.sim_peak_val = sim_smooth
                self.sim_peak_frame = frame
            self.sim_was_above_threshold = True
        else:
            # Signal dropped below threshold
            if self.sim_was_above_threshold and self.saw_trough:
                # We had a peak and now fell — rep completed
                if self.sim_peak_val > TSM_PEAK_MIN_HEIGHT:
                    # Valid rep!
                    self.sim_was_above_threshold = False
                    self.sim_peak_val = 0.0
                    self.saw_trough = False
                    self.sim_trough_since_last_rep = sim_smooth
                    return True
            
            self.sim_was_above_threshold = False
            self.sim_peak_val = 0.0
            
            # Track trough (need to see a trough between reps)
            if sim_smooth < threshold * 0.6:
                self.saw_trough = True
                self.sim_trough_since_last_rep = min(
                    self.sim_trough_since_last_rep, sim_smooth)
        
        return False
    
    def process_frame(self, landmarks):
        """
        Process one frame through the TSM pipeline.
        
        Returns: (rep_result_or_None, angles_dict, debug_info_dict)
        """
        # ── 1. Compute angles ──
        angles = compute_angles(landmarks)
        
        # ── 2. Compute velocities ──
        if self.prev_angles is not None:
            vels = {f'{k}_vel': angles[k] - self.prev_angles[k] for k in angles}
        else:
            vels = {f'{k}_vel': 0.0 for k in angles}
        self.prev_angles = dict(angles)
        
        # ── 3. Build 8-feature vector ──
        feat8 = np.array([
            angles['primary_arm_elevation'],
            angles['primary_elbow_angle'],
            angles['torso_lean'],
            angles['shoulder_height_diff'],
            vels['primary_arm_elevation_vel'],
            vels['primary_elbow_angle_vel'],
            vels['torso_lean_vel'],
            vels['shoulder_height_diff_vel'],
        ], dtype=np.float32)
        
        # ── 4. VAE forward pass ──
        latent, recon_error = run_vae(feat8)
        self.last_recon_error = recon_error
        
        # ── 5. Compute similarity row (one row of the TSM) ──
        sim_row = self._compute_similarity_row(latent)
        
        # ── 6. Add to buffers ──
        self.latent_buffer.append(latent.copy())
        self.similarity_rows.append(sim_row.copy())
        self.feature8_buffer.append(feat8.copy())
        self.latent_for_class_buffer.append(latent.copy())
        
        frame = self.frame_count
        self.frame_count += 1
        
        # ── 7. Periodically estimate period from autocorrelation ──
        if frame % TSM_AUTOCORR_UPDATE_INTERVAL == 0 and frame > TSM_PERIOD_MIN * 3:
            period, confidence = self._estimate_period_autocorrelation()
            
            if period is not None and confidence > TSM_CONFIDENCE_THRESHOLD:
                self.period_history.append(period)
                
                # Use median of recent estimates for stability
                if len(self.period_history) >= 3:
                    self.estimated_period = int(np.median(list(self.period_history)))
                else:
                    self.estimated_period = period
                
                self.period_confidence = confidence
        
        # ── 8. If we have a period estimate, track similarity at that lag ──
        rep_result = None
        sim_at_period = 0.0
        sim_smooth = 0.0
        
        if self.estimated_period is not None and len(self.latent_buffer) > self.estimated_period:
            # Get latent from P frames ago
            lag_idx = len(self.latent_buffer) - 1 - self.estimated_period
            if lag_idx >= 0:
                z_past = self.latent_buffer[lag_idx]
                sim_at_period = self._cosine_similarity(latent, z_past)
            
            # Smooth the similarity signal
            sim_smooth = self._smooth_value(sim_at_period, self.sim_signal_smooth)
            
            # ── 9. Detect rep from similarity peak ──
            rep_detected = self._detect_rep_from_similarity(sim_smooth, frame)
            
            if rep_detected:
                self.rep_count += 1
                self.last_rep_frame = frame
                
                # ── 10. Classify the detected rep ──
                rep_result = self._classify_detected_rep(frame, recon_error)
                self.rep_history.append(rep_result)
        
        # Update debug info
        self.debug_info = {
            'period': self.estimated_period,
            'confidence': self.period_confidence,
            'sim_value': sim_at_period,
            'sim_smooth': sim_smooth,
            'buffer_len': len(self.latent_buffer),
            'elevation': angles['primary_arm_elevation'],
        }
        
        return rep_result, angles, self.debug_info
    
    def _classify_detected_rep(self, end_frame, recon_error):
        """Extract window around detected rep and classify it."""
        
        result = {
            'rep_id': self.rep_count,
            'end_frame': end_frame,
            'duration': self.estimated_period if self.estimated_period else 0,
            'peak_elevation': 0.0,
            'recon_error': recon_error,
            'anomaly': recon_error > ANOMALY_THRESHOLD,
            'class': 'unknown',
            'confidence': 0.0,
            'probs': np.zeros(len(CLASS_LABELS), dtype=np.float32),
        }
        
        if result['anomaly']:
            return result
        
        # Extract one period of data ending at current frame
        period = self.estimated_period if self.estimated_period else 70
        buf_len = len(self.feature8_buffer)
        
        window_size = min(period, buf_len)
        start_idx = buf_len - window_size
        end_idx = buf_len
        
        feat8_window = np.array(list(self.feature8_buffer)[start_idx:end_idx], dtype=np.float32)
        latent_window = np.array(list(self.latent_for_class_buffer)[start_idx:end_idx], dtype=np.float32)
        
        if len(feat8_window) < 3:
            return result
        
        # Peak elevation within window (first column is arm_elevation)
        result['peak_elevation'] = float(np.max(feat8_window[:, 0]))
        result['start_frame'] = end_frame - window_size
        
        # Build classifier input: concat feat8 + latents
        window = np.hstack([feat8_window, latent_window])
        
        cls, conf, probs = classify_rep_conv1d(window)
        result['class'] = cls
        result['confidence'] = conf
        result['probs'] = probs
        
        return result


# =============================================================================
# OVERLAY RENDERER
# =============================================================================

class OverlayRenderer:
    def __init__(self):
        self.feedback_text = ""
        self.feedback_color = COLORS['idle']
        self.feedback_timer = 0
        self.feedback_duration = 90
        
        # Similarity signal visualization
        self.sim_history = deque(maxlen=200)
        self.threshold_history = deque(maxlen=200)

    def set_feedback(self, rep_result):
        cls = rep_result['class']
        conf = rep_result['confidence']
        rid = rep_result['rep_id']
        msg = FEEDBACK_MESSAGES.get(cls, "")
        self.feedback_text = f"Rep {rid}: {msg} ({conf:.0%})"
        self.feedback_color = COLORS.get(cls, COLORS['idle'])
        self.feedback_timer = self.feedback_duration

    def draw(self, frame, detector, angles, debug_info, fps):
        h, w = frame.shape[:2]

        # ── Stats panel (top-left) ──
        panel_w = 380
        panel_h = 230
        ov = frame.copy()
        cv2.rectangle(ov, (8, 8), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

        y = 30
        cv2.putText(frame, f"FPS: {fps:.0f}", (18, y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        y += 28
        cv2.putText(frame, f"Reps: {detector.rep_count}", (18, y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        
        # TSM-specific info
        y += 30
        elev = debug_info.get('elevation', 0)
        cv2.putText(frame, f"Elevation: {elev:.0f} deg", (18, y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        y += 24
        period = debug_info.get('period', None)
        conf = debug_info.get('confidence', 0)
        if period is not None:
            cv2.putText(frame, f"Period: {period} frames ({conf:.2f})", (18, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "Period: estimating...", (18, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        y += 24
        sim_val = debug_info.get('sim_smooth', 0)
        sim_color = (0, 255, 0) if sim_val > TSM_PEAK_MIN_HEIGHT else (150, 150, 150)
        cv2.putText(frame, f"Similarity: {sim_val:.3f}", (18, y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, sim_color, 1)
        
        y += 24
        buf_len = debug_info.get('buffer_len', 0)
        cv2.putText(frame, f"Buffer: {buf_len}/{TSM_BUFFER_SIZE}", (18, y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        # ── Similarity signal graph (bottom-left) ──
        self.sim_history.append(debug_info.get('sim_smooth', 0))
        
        graph_x = 10
        graph_y = h - 130
        graph_w = 400
        graph_h = 110
        
        ov2 = frame.copy()
        cv2.rectangle(ov2, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h),
                       (0, 0, 0), -1)
        cv2.addWeighted(ov2, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "TSM Similarity Signal", (graph_x + 5, graph_y + 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        if len(self.sim_history) > 2:
            sig = np.array(self.sim_history)
            n = len(sig)
            
            # Draw threshold line
            thresh_y_px = int(graph_y + graph_h - TSM_PEAK_MIN_HEIGHT * graph_h)
            cv2.line(frame, (graph_x, thresh_y_px), (graph_x + graph_w, thresh_y_px),
                     (0, 0, 200), 1)
            cv2.putText(frame, "thresh", (graph_x + graph_w - 50, thresh_y_px - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)
            
            # Draw signal
            # Map signal values [−0.5, 1.0] to pixel y coordinates
            for i in range(1, n):
                x1 = graph_x + int((i - 1) * graph_w / max(n - 1, 1))
                x2 = graph_x + int(i * graph_w / max(n - 1, 1))
                
                v1 = np.clip(sig[i-1], -0.5, 1.0)
                v2 = np.clip(sig[i], -0.5, 1.0)
                
                y1 = int(graph_y + graph_h - (v1 + 0.5) / 1.5 * graph_h)
                y2 = int(graph_y + graph_h - (v2 + 0.5) / 1.5 * graph_h)
                
                color = (0, 255, 0) if sig[i] > TSM_PEAK_MIN_HEIGHT else (0, 200, 200)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        # ── Elevation bar ──
        bar_x = 18
        bar_y = panel_h + 8
        bar_w = panel_w - 30
        bar_h = 18
        fill_pct = np.clip(elev / 180.0, 0, 1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                       (50, 50, 50), -1)
        fill_w = int(bar_w * fill_pct)
        bar_col = (0, 200, 0) if 80 < elev < 150 else (0, 0, 255) if elev > 150 else (0, 165, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

        # ── Center feedback ──
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
            alpha = min(1.0, self.feedback_timer / 20.0)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            ts = cv2.getTextSize(self.feedback_text, font, 1.1, 3)[0]
            tx = (w - ts[0]) // 2
            ty = h - 160
            
            ov3 = frame.copy()
            cv2.rectangle(ov3, (tx - 20, ty - ts[1] - 20),
                          (tx + ts[0] + 20, ty + 20), (0, 0, 0), -1)
            cv2.addWeighted(ov3, 0.65 * alpha, frame, 1.0 - 0.65 * alpha, 0, frame)
            
            col = tuple(int(c * alpha) for c in self.feedback_color)
            cv2.putText(frame, self.feedback_text, (tx, ty), font, 1.1, col, 3)

        # ── Rep history (right side) ──
        if len(detector.rep_history) > 0:
            n_show = min(10, len(detector.rep_history))
            hx = w - 300
            hy = 12
            
            ov4 = frame.copy()
            cv2.rectangle(ov4, (hx - 10, hy), (w - 8, hy + 28 + n_show * 25),
                          (0, 0, 0), -1)
            cv2.addWeighted(ov4, 0.65, frame, 0.35, 0, frame)
            
            cv2.putText(frame, "Rep History", (hx, hy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            
            for i, rep in enumerate(detector.rep_history[-n_show:]):
                ry = hy + 42 + i * 25
                cls = rep['class']
                conf = rep['confidence']
                rid = rep['rep_id']
                col = COLORS.get(cls, COLORS['idle'])
                cv2.putText(frame, f"#{rid}: {cls} ({conf:.0%})",
                            (hx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.47, col, 1)

        # ── Instructions ──
        cv2.putText(frame, "Q/ESC=Quit  R=Reset", (20, h - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        return frame


# =============================================================================
# DRAW SKELETON
# =============================================================================

def draw_skeleton(frame, landmarks, fw, fh):
    key_pts = [RS, RE, RW, RH, LS, LH]
    conns = [(RS, RE), (RE, RW), (RS, RH), (LS, LH), (RS, LS), (RH, LH)]

    def px(idx):
        lm = landmarks[idx]
        return int(lm.x * fw), int(lm.y * fh)

    for p1, p2 in conns:
        try:
            cv2.line(frame, px(p1), px(p2), (0, 255, 255), 2)
        except (IndexError, AttributeError):
            pass

    for idx in key_pts:
        try:
            x, y = px(idx)
            cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)
            cv2.circle(frame, (x, y), 9, (255, 255, 255), 2)
        except (IndexError, AttributeError):
            pass

    try:
        wx, wy = px(RW)
        cv2.putText(frame, "R.WRIST", (wx + 10, wy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    except (IndexError, AttributeError):
        pass

    return frame


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  FRONT SHOULDER RAISE — TSM Real-Time Form Assessment")
    print("=" * 70)
    print(f"  Camera: index {CAMERA_INDEX}")
    print(f"  Detection: Temporal Self-Similarity Matrix (VAE latent space)")
    print(f"  Classification: Conv1D (ONNX)")
    print(f"  Controls: Q/ESC=Quit, R=Reset")
    print("=" * 70)

    # ── MediaPipe Tasks API ──
    base_options = mp_tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    print("  ✓ MediaPipe PoseLandmarker (VIDEO mode)")

    # ── Camera ──
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"  ⚠ Camera {CAMERA_INDEX} failed, trying 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  ✗ No camera available!")
            landmarker.close()
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    print(f"  ✓ Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # ── Components ──
    detector = TSMRepDetector()
    renderer = OverlayRenderer()

    fps_deque = deque(maxlen=30)
    prev_time = time.time()
    total_frames = 0
    timestamp_ms = 0

    print("\n  🏋️ Ready! Start doing front shoulder raises.")
    print("  (TSM needs ~3-4 reps to lock onto your period)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            total_frames += 1
            timestamp_ms += 33

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps_deque.append(1.0 / dt)
            fps = float(np.mean(fps_deque)) if fps_deque else 0.0

            # ── Pose ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            try:
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception:
                result = None

            landmarks = None
            if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]

            debug_info = {'period': None, 'confidence': 0, 'sim_smooth': 0,
                          'sim_value': 0, 'buffer_len': 0, 'elevation': 0}

            if landmarks is not None and len(landmarks) >= 33:
                frame = draw_skeleton(frame, landmarks, fw, fh)

                rep_result, angles, debug_info = detector.process_frame(landmarks)

                if rep_result is not None:
                    renderer.set_feedback(rep_result)
                    cls = rep_result['class']
                    conf = rep_result['confidence']
                    peak = rep_result.get('peak_elevation', 0)
                    dur = rep_result.get('duration', 0)
                    rid = rep_result['rep_id']
                    period = debug_info.get('period', '?')
                    print(f"  Rep {rid:2d}: {cls:12s} ({conf:.0%}) "
                          f"| Peak: {peak:.0f}° | Period: {period} frames")
            else:
                msg = "Stand in frame - no pose detected"
                ts = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.putText(frame, msg, ((fw - ts[0]) // 2, fh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            frame = renderer.draw(frame, detector, angles, debug_info, fps)

            cv2.imshow('Front Shoulder Raise - TSM Assessment', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                detector = TSMRepDetector()
                renderer = OverlayRenderer()
                print("\n  🔄 Reset!\n")

    except KeyboardInterrupt:
        print("\n  Stopped.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

        print("\n" + "=" * 70)
        print("  SESSION SUMMARY")
        print("=" * 70)
        print(f"  Frames: {total_frames}")
        print(f"  Reps:   {detector.rep_count}")

        if detector.rep_history:
            from collections import Counter
            counts = Counter(r['class'] for r in detector.rep_history)
            
            print(f"\n  Breakdown:")
            for cls in CLASS_LABELS:
                n = counts.get(cls, 0)
                pct = n / len(detector.rep_history) * 100
                bar = '█' * int(pct / 5)
                print(f"    {cls:15s}: {n:3d} ({pct:4.0f}%) {bar}")

            confs = [r['confidence'] for r in detector.rep_history]
            print(f"\n  Confidence: avg={np.mean(confs):.0%}, min={np.min(confs):.0%}")

            print(f"\n  {'#':>3s}  {'Class':12s}  {'Conf':>5s}  {'Peak':>5s}  {'Dur':>4s}")
            print(f"  {'-'*38}")
            for r in detector.rep_history:
                print(f"  {r['rep_id']:3d}  {r['class']:12s}  {r['confidence']:5.0%}  "
                      f"{r.get('peak_elevation',0):5.0f}°  {r.get('duration',0):4d}f")
        else:
            print("\n  No reps detected.")
            print("  Tip: TSM needs ~3-4 reps to estimate the period.")
            print("  Try doing 5+ continuous reps at a steady pace.")

        print("=" * 70)


if __name__ == '__main__':
    main()