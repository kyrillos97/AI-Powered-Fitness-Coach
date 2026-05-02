"""
FormBall Real-Time Exercise Form Assessment
============================================
Uses webcam + MediaPipe BlazePose + TFLite model
to classify exercise form in real-time.

Requirements:
  - tensorflow==2.15.0
  - mediapipe==0.10.9
  - opencv-python
  - numpy

Files needed in same directory:
  - formball_model.tflite
  - formball_config.json
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import os
import sys
import time
from collections import deque

# ============================================================
# CONFIG
# ============================================================

CAMERA_INDEX = 1
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH = os.path.join(SCRIPT_DIR, "formball_model.tflite")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "formball_config.json")

# Display settings
WINDOW_NAME = "FormBall - Real-Time Exercise Assessment"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Ball smoothing
SMOOTHING_ALPHA = 0.3  # 0 = very smooth, 1 = no smoothing


# ============================================================
# LOAD MODEL & CONFIG
# ============================================================

def load_model_and_config():
    """Load TFLite model and JSON config."""
    # Check files exist
    if not os.path.exists(TFLITE_PATH):
        print(f"❌ TFLite model not found: {TFLITE_PATH}")
        sys.exit(1)
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config not found: {CONFIG_PATH}")
        sys.exit(1)

    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    print(f"✅ Config loaded: {config['exercise']} (v{config['model_version']})")

    # Load TFLite
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"✅ TFLite loaded: {os.path.getsize(TFLITE_PATH)/1024:.1f} KB")
    print(f"   Input:  {input_details[0]['shape']}")
    for i, od in enumerate(output_details):
        print(f"   Output[{i}]: {od['shape']} ({od['name']})")

    return interpreter, input_details, output_details, config


# ============================================================
# ANGLE CALCULATION
# ============================================================

def calculate_angle_3d(a, b, c):
    """Calculate angle at point b given three 3D points."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def extract_angles_from_landmarks(landmarks):
    """
    Extract 8 joint angles from MediaPipe landmarks.
    Returns dict of angles or None if landmarks are not visible enough.
    """
    # Check visibility of critical landmarks
    critical_indices = [11, 12, 13, 14, 15, 16, 23, 24]
    for idx in critical_indices:
        if landmarks[idx].visibility < 0.5:
            return None

    # Extract 3D coordinates
    def lm(idx):
        l = landmarks[idx]
        return np.array([l.x, l.y, l.z])

    ls = lm(11);  rs = lm(12)   # shoulders
    le = lm(13);  re = lm(14)   # elbows
    lw = lm(15);  rw = lm(16)   # wrists
    lh = lm(23);  rh = lm(24)   # hips

    angles = {
        'l_elbow_angle': calculate_angle_3d(lw, le, ls),
        'r_elbow_angle': calculate_angle_3d(rw, re, rs),
        'l_shoulder_angle': calculate_angle_3d(le, ls, lh),
        'r_shoulder_angle': calculate_angle_3d(re, rs, rh),
        'l_torso_angle': calculate_angle_3d(ls, lh, lh + np.array([0, 0.1, 0])),
        'r_torso_angle': calculate_angle_3d(rs, rh, rh + np.array([0, 0.1, 0])),
        'elbow_symmetry': abs(calculate_angle_3d(lw, le, ls) - calculate_angle_3d(rw, re, rs)),
        'shoulder_symmetry': abs(calculate_angle_3d(le, ls, lh) - calculate_angle_3d(re, rs, rh)),
    }
    return angles


# ============================================================
# REP TRACKER (Phase Features + Rep Counting)
# ============================================================

class RepTracker:
    """
    Tracks exercise repetitions and computes phase features in real-time.

    Phase features require knowing where we are within a rep.
    We detect reps by monitoring shoulder angle:
      - Rep starts when shoulder angle rises above a threshold
      - Rep ends when it drops back below the threshold
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # Rep counting
        self.rep_count = 0
        self.in_rep = False
        self.rep_start_threshold = 30.0   # Shoulder angle to start rep
        self.rep_end_threshold = 25.0     # Shoulder angle to end rep

        # Current rep tracking
        self.rep_frame_index = 0
        self.rep_shoulder_min = 999.0
        self.rep_shoulder_max = 0.0

        # History for velocity/acceleration
        self.prev_shoulder_angle = None
        self.prev_shoulder_velocity = None
        self.prev_elbow_angle = None

        # Smoothing history
        self.angle_history = deque(maxlen=5)

        # Rep length estimation (for phase calculation)
        self.recent_rep_lengths = deque(maxlen=5)
        self.estimated_rep_length = 70  # Default ~70 frames at 30fps

    def update(self, angles):
        """
        Update rep tracker with new angles.
        Returns 8 phase features or None if not ready.
        """
        shoulder = angles['l_shoulder_angle']
        elbow = angles['l_elbow_angle']

        # --- Rep detection ---
        if not self.in_rep:
            if shoulder > self.rep_start_threshold:
                # Rep started!
                self.in_rep = True
                self.rep_frame_index = 0
                self.rep_shoulder_min = shoulder
                self.rep_shoulder_max = shoulder
                self.prev_shoulder_angle = shoulder
                self.prev_shoulder_velocity = 0.0
                self.prev_elbow_angle = elbow
        else:
            self.rep_frame_index += 1

            # Update min/max within this rep
            self.rep_shoulder_min = min(self.rep_shoulder_min, shoulder)
            self.rep_shoulder_max = max(self.rep_shoulder_max, shoulder)

            # Check if rep ended
            if shoulder < self.rep_end_threshold and self.rep_frame_index > 15:
                # Rep completed!
                self.rep_count += 1
                self.recent_rep_lengths.append(self.rep_frame_index)
                self.estimated_rep_length = int(np.mean(self.recent_rep_lengths))
                self.in_rep = False

        # --- Compute phase features ---

        # Phase position (0 to 1 within estimated rep)
        if self.in_rep:
            phase = min(self.rep_frame_index / max(self.estimated_rep_length - 1, 1), 1.0)
        else:
            phase = 0.0

        phase_sin = np.sin(2 * np.pi * phase)
        phase_cos = np.cos(2 * np.pi * phase)

        # Velocity (approximated as frame-to-frame difference)
        if self.prev_shoulder_angle is not None:
            shoulder_velocity = shoulder - self.prev_shoulder_angle
        else:
            shoulder_velocity = 0.0

        # Acceleration
        if self.prev_shoulder_velocity is not None:
            shoulder_acceleration = shoulder_velocity - self.prev_shoulder_velocity
        else:
            shoulder_acceleration = 0.0

        # Elbow velocity
        if self.prev_elbow_angle is not None:
            elbow_velocity = elbow - self.prev_elbow_angle
        else:
            elbow_velocity = 0.0

        # Shoulder normalized (within current rep)
        range_s = self.rep_shoulder_max - self.rep_shoulder_min + 1e-8
        if self.in_rep and range_s > 5.0:  # Need meaningful range
            shoulder_normalized = (shoulder - self.rep_shoulder_min) / range_s
        else:
            shoulder_normalized = 0.0

        shoulder_normalized = np.clip(shoulder_normalized, 0.0, 1.0)
        distance_from_peak = 1.0 - shoulder_normalized
        is_peak_zone = 1.0 if shoulder_normalized > 0.8 else 0.0

        # Update history
        self.prev_shoulder_velocity = shoulder_velocity
        self.prev_shoulder_angle = shoulder
        self.prev_elbow_angle = elbow

        phase_features = {
            'phase_sin': phase_sin,
            'phase_cos': phase_cos,
            'shoulder_velocity': shoulder_velocity,
            'shoulder_acceleration': shoulder_acceleration,
            'elbow_velocity': elbow_velocity,
            'shoulder_normalized': shoulder_normalized,
            'distance_from_peak': distance_from_peak,
            'is_peak_zone': is_peak_zone,
        }

        return phase_features


# ============================================================
# BALL SMOOTHER
# ============================================================

class BallSmoother:
    """Exponential Moving Average smoothing for predictions."""

    def __init__(self, alpha=0.3, class_names=None):
        self.alpha = alpha
        self.class_names = class_names or []
        self.prev_probs = None
        self.prev_distances = None

    def smooth_probs(self, probs):
        """Smooth class probabilities."""
        if self.prev_probs is None:
            self.prev_probs = probs.copy()
            return probs

        smoothed = self.alpha * probs + (1 - self.alpha) * self.prev_probs
        self.prev_probs = smoothed.copy()
        return smoothed

    def smooth_distances(self, distances):
        """Smooth prototype distances."""
        if self.prev_distances is None:
            self.prev_distances = distances.copy()
            return distances

        smoothed = {}
        for cls in distances:
            smoothed[cls] = (self.alpha * distances[cls] +
                             (1 - self.alpha) * self.prev_distances.get(cls, distances[cls]))
        self.prev_distances = smoothed.copy()
        return smoothed


# ============================================================
# ONE EURO FILTER (for landmark smoothing)
# ============================================================

class OneEuroFilter:
    """Smooth a single value over time. Reduces jitter."""

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        # Derivative
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filter
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


# ============================================================
# INFERENCE
# ============================================================

def run_inference(interpreter, input_details, output_details, config, features_16):
    """
    Run TFLite model and return predictions + embeddings.

    Args:
        features_16: numpy array of 16 float32 features (raw, unscaled)

    Returns:
        dict with class_probs, predicted_class, embedding, distances, is_ood
    """
    # Prepare input
    input_data = np.array([features_16], dtype=np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get outputs
    logits_idx = config['tflite_output_order']['logits_index']
    embed_idx = config['tflite_output_order']['embedding_index']

    logits = interpreter.get_tensor(output_details[logits_idx]['index'])[0]
    embedding = interpreter.get_tensor(output_details[embed_idx]['index'])[0]

    # Softmax for probabilities
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    probs = exp_logits / exp_logits.sum()

    # Predicted class
    label_map_inv = config['label_map_inv']
    pred_idx = np.argmax(probs)
    pred_class = label_map_inv[str(pred_idx)]

    # Prototype distances
    prototypes = config['prototypes']
    distances = {}
    for cls_name, proto in prototypes.items():
        distances[cls_name] = float(np.linalg.norm(embedding - np.array(proto)))

    ball_class = min(distances, key=distances.get)
    min_distance = distances[ball_class]

    # OOD detection
    is_ood = min_distance > config['global_ood_threshold']

    return {
        'probs': probs,
        'predicted_class': pred_class,
        'ball_class': ball_class,
        'embedding': embedding,
        'distances': distances,
        'min_distance': min_distance,
        'is_ood': is_ood,
    }


# ============================================================
# DRAWING FUNCTIONS
# ============================================================

# Color scheme (BGR for OpenCV)
CLASS_COLORS = {
    'perfect':    (75, 204, 46),    # Green
    'over_range': (60, 76, 231),    # Red
    'low':        (18, 156, 243),   # Orange
    'bent_elbow': (180, 89, 155),   # Purple
    'unknown':    (128, 128, 128),  # Gray
}

FEEDBACK_MESSAGES = {
    'perfect':    "Perfect Form!",
    'over_range': "Too High! Lower your arms",
    'low':        "Too Low! Raise higher",
    'bent_elbow': "Straighten your elbows!",
    'unknown':    "Unknown movement",
}


def draw_header(frame, rep_count, fps):
    """Draw top header bar."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (40, 40, 40), -1)
    cv2.putText(frame, f"FormBall - Front Shoulder Raise",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {rep_count}",
                (w - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (w - 350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


def draw_probability_bars(frame, probs, class_names, smoothed_distances, x_start, y_start):
    """Draw horizontal probability bars for each class."""
    bar_width = 200
    bar_height = 25
    gap = 8

    for i, cls_name in enumerate(class_names):
        y = y_start + i * (bar_height + gap)
        prob = probs[i] if i < len(probs) else 0
        color = CLASS_COLORS.get(cls_name, (128, 128, 128))

        # Background
        cv2.rectangle(frame, (x_start, y), (x_start + bar_width, y + bar_height),
                      (60, 60, 60), -1)

        # Filled bar
        fill_width = int(bar_width * prob)
        cv2.rectangle(frame, (x_start, y), (x_start + fill_width, y + bar_height),
                      color, -1)

        # Border
        cv2.rectangle(frame, (x_start, y), (x_start + bar_width, y + bar_height),
                      (200, 200, 200), 1)

        # Label
        label = f"{cls_name}: {prob:.0%}"
        cv2.putText(frame, label, (x_start + bar_width + 10, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_feedback_box(frame, result, rep_tracker):
    """Draw the main feedback area."""
    h, w = frame.shape[:2]

    if result['is_ood']:
        cls = 'unknown'
    else:
        cls = result['ball_class']

    color = CLASS_COLORS.get(cls, (128, 128, 128))
    message = FEEDBACK_MESSAGES.get(cls, "")

    # Feedback background
    box_y = h - 120
    cv2.rectangle(frame, (10, box_y), (w - 10, h - 10), (40, 40, 40), -1)
    cv2.rectangle(frame, (10, box_y), (w - 10, h - 10), color, 3)

    # Status indicator circle
    cv2.circle(frame, (50, box_y + 55), 25, color, -1)

    # Feedback text
    cv2.putText(frame, message, (90, box_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Confidence
    confidence = 1.0 - min(result['min_distance'] / 15.0, 1.0)  # Normalize
    conf_text = f"Confidence: {confidence:.0%}"
    cv2.putText(frame, conf_text, (90, box_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Phase info
    if rep_tracker.in_rep:
        phase_text = f"In Rep (frame {rep_tracker.rep_frame_index})"
        cv2.putText(frame, phase_text, (w - 300, box_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)


def draw_ball_visualization(frame, distances, x_center, y_center, radius=100):
    """
    Draw the "ball" visualization — a 2D map showing where the current
    form is relative to each class prototype.
    """
    # Background circle
    cv2.circle(frame, (x_center, y_center), radius + 10, (40, 40, 40), -1)
    cv2.circle(frame, (x_center, y_center), radius + 10, (100, 100, 100), 2)

    # Place prototypes at fixed positions around the circle
    proto_positions = {
        'perfect':    (x_center, y_center - radius + 20),        # Top
        'over_range': (x_center + radius - 20, y_center),        # Right
        'low':        (x_center - radius + 20, y_center),        # Left
        'bent_elbow': (x_center, y_center + radius - 20),        # Bottom
    }

    # Draw prototype labels
    for cls_name, (px, py) in proto_positions.items():
        color = CLASS_COLORS.get(cls_name, (128, 128, 128))
        cv2.circle(frame, (px, py), 8, color, -1)
        cv2.circle(frame, (px, py), 8, (255, 255, 255), 1)

        # Label
        label_offset_x = -30 if 'low' in cls_name else 15
        label_offset_y = -15 if cls_name in ['perfect', 'over_range'] else 20
        if cls_name == 'perfect':
            label_offset_x = -25
        cv2.putText(frame, cls_name[:7], (px + label_offset_x, py + label_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Compute ball position based on inverse distances (closer = stronger pull)
    total_inv_dist = 0
    ball_x, ball_y = 0.0, 0.0

    for cls_name, dist in distances.items():
        if cls_name not in proto_positions:
            continue
        inv_dist = 1.0 / (dist + 0.1)
        total_inv_dist += inv_dist
        px, py = proto_positions[cls_name]
        ball_x += px * inv_dist
        ball_y += py * inv_dist

    if total_inv_dist > 0:
        ball_x = int(ball_x / total_inv_dist)
        ball_y = int(ball_y / total_inv_dist)
    else:
        ball_x, ball_y = x_center, y_center

    # Determine ball color based on closest class
    closest = min(distances, key=distances.get)
    ball_color = CLASS_COLORS.get(closest, (128, 128, 128))

    # Draw ball with glow effect
    cv2.circle(frame, (ball_x, ball_y), 18, ball_color, -1)
    cv2.circle(frame, (ball_x, ball_y), 18, (255, 255, 255), 2)
    cv2.circle(frame, (ball_x, ball_y), 5, (255, 255, 255), -1)


def draw_angle_gauges(frame, angles, x_start, y_start):
    """Draw small angle gauges for key joints."""
    gauges = [
        ('L Shoulder', angles.get('l_shoulder_angle', 0), 0, 180),
        ('L Elbow', angles.get('l_elbow_angle', 0), 0, 180),
        ('R Shoulder', angles.get('r_shoulder_angle', 0), 0, 180),
        ('R Elbow', angles.get('r_elbow_angle', 0), 0, 180),
    ]

    for i, (name, value, min_v, max_v) in enumerate(gauges):
        y = y_start + i * 25
        normalized = np.clip((value - min_v) / (max_v - min_v + 1e-8), 0, 1)
        bar_w = 100
        fill_w = int(bar_w * normalized)

        cv2.rectangle(frame, (x_start, y), (x_start + bar_w, y + 15), (60, 60, 60), -1)
        cv2.rectangle(frame, (x_start, y), (x_start + fill_w, y + 15), (180, 180, 0), -1)
        cv2.putText(frame, f"{name}: {value:.0f}", (x_start + bar_w + 5, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  FormBall Real-Time Exercise Assessment")
    print("=" * 60)

    # --- Load model ---
    interpreter, input_details, output_details, config = load_model_and_config()

    class_names = ['perfect', 'over_range', 'low', 'bent_elbow']
    feature_names = config['feature_names']
    print(f"\n  Features ({len(feature_names)}): {feature_names}")

    # --- Initialize MediaPipe ---
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,          # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("✅ MediaPipe Pose initialized")

    # --- Initialize trackers ---
    rep_tracker = RepTracker()
    ball_smoother = BallSmoother(alpha=SMOOTHING_ALPHA, class_names=class_names)

    # One Euro Filters for key angles (reduce jitter)
    angle_filters = {
        'l_elbow_angle': OneEuroFilter(min_cutoff=1.7, beta=0.01),
        'r_elbow_angle': OneEuroFilter(min_cutoff=1.7, beta=0.01),
        'l_shoulder_angle': OneEuroFilter(min_cutoff=1.7, beta=0.01),
        'r_shoulder_angle': OneEuroFilter(min_cutoff=1.7, beta=0.01),
        'l_torso_angle': OneEuroFilter(min_cutoff=1.7, beta=0.01),
        'r_torso_angle': OneEuroFilter(min_cutoff=1.7, beta=0.01),
        'elbow_symmetry': OneEuroFilter(min_cutoff=1.7, beta=0.01),
        'shoulder_symmetry': OneEuroFilter(min_cutoff=1.7, beta=0.01),
    }

    # --- Open webcam ---
    print(f"\n  Opening camera (index={CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"❌ Cannot open camera {CAMERA_INDEX}")
        print("   Try changing CAMERA_INDEX to 0 at the top of this script")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Camera opened: {actual_w}×{actual_h}")
    print(f"\n  Press 'q' to quit, 'r' to reset rep counter")
    print("=" * 60 + "\n")

    # --- FPS tracker ---
    fps_times = deque(maxlen=30)
    frame_count = 0

    # --- Last valid result (for display when landmarks not visible) ---
    last_result = None
    last_angles = None
    frames_without_pose = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame capture failed")
                break

            frame_time = time.time()
            fps_times.append(frame_time)
            frame_count += 1

            # Calculate FPS
            if len(fps_times) > 1:
                fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
            else:
                fps = 0

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # --- MediaPipe Pose Detection ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # --- Process pose ---
            current_result = None
            current_angles = None

            if results.pose_landmarks:
                frames_without_pose = 0
                landmarks = results.pose_landmarks.landmark

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # Extract angles
                raw_angles = extract_angles_from_landmarks(landmarks)

                if raw_angles is not None:
                    # Apply One Euro Filter to smooth angles
                    filtered_angles = {}
                    for key, value in raw_angles.items():
                        if key in angle_filters:
                            filtered_angles[key] = angle_filters[key](value, frame_time)
                        else:
                            filtered_angles[key] = value

                    current_angles = filtered_angles

                    # Compute phase features
                    phase_features = rep_tracker.update(filtered_angles)

                    if phase_features is not None:
                        # Build 16-feature input vector
                        # Order MUST match training: base_angles + phase_cols
                        features = np.array([
                            filtered_angles['l_elbow_angle'],
                            filtered_angles['r_elbow_angle'],
                            filtered_angles['l_shoulder_angle'],
                            filtered_angles['r_shoulder_angle'],
                            filtered_angles['l_torso_angle'],
                            filtered_angles['r_torso_angle'],
                            filtered_angles['elbow_symmetry'],
                            filtered_angles['shoulder_symmetry'],
                            phase_features['phase_sin'],
                            phase_features['phase_cos'],
                            phase_features['shoulder_velocity'],
                            phase_features['shoulder_acceleration'],
                            phase_features['elbow_velocity'],
                            phase_features['shoulder_normalized'],
                            phase_features['distance_from_peak'],
                            phase_features['is_peak_zone'],
                        ], dtype=np.float32)

                        # Run inference
                        current_result = run_inference(
                            interpreter, input_details, output_details, config, features
                        )

                        # Smooth predictions
                        current_result['probs'] = ball_smoother.smooth_probs(current_result['probs'])
                        current_result['distances'] = ball_smoother.smooth_distances(
                            current_result['distances']
                        )

                        # Update ball_class after smoothing
                        current_result['ball_class'] = min(
                            current_result['distances'],
                            key=current_result['distances'].get
                        )

                        last_result = current_result
                        last_angles = current_angles

            else:
                frames_without_pose += 1
                if frames_without_pose > 30:  # 1 second without pose
                    last_result = None
                    last_angles = None

            # --- DRAW UI ---

            # Use current or last valid result
            display_result = current_result or last_result
            display_angles = current_angles or last_angles

            # Header
            draw_header(frame, rep_tracker.rep_count, fps)

            if display_result is not None:
                # Probability bars (right side)
                draw_probability_bars(
                    frame, display_result['probs'], class_names,
                    display_result['distances'],
                    x_start=w - 420, y_start=70
                )

                # Ball visualization (right side, below bars)
                draw_ball_visualization(
                    frame, display_result['distances'],
                    x_center=w - 320, y_center=330, radius=90
                )

                # Feedback box (bottom)
                draw_feedback_box(frame, display_result, rep_tracker)

            else:
                # No pose detected
                cv2.putText(frame, "Stand in frame to begin",
                            (w // 2 - 200, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # Angle gauges (left side)
            if display_angles is not None:
                draw_angle_gauges(frame, display_angles, x_start=10, y_start=70)

            # Rep tracker status (small text)
            status = "IN REP" if rep_tracker.in_rep else "RESTING"
            status_color = (0, 255, 0) if rep_tracker.in_rep else (100, 100, 100)
            cv2.putText(frame, status, (10, h - 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # --- Show frame ---
            cv2.imshow(WINDOW_NAME, frame)

            # --- Handle keys ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                print("\n  Quitting...")
                break
            elif key == ord('r'):
                rep_tracker.reset()
                ball_smoother = BallSmoother(alpha=SMOOTHING_ALPHA, class_names=class_names)
                print("  🔄 Rep counter reset")

    except KeyboardInterrupt:
        print("\n  Interrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

        print(f"\n  Session summary:")
        print(f"    Total frames: {frame_count}")
        print(f"    Total reps:   {rep_tracker.rep_count}")
        print(f"  ✅ Cleanup complete")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()