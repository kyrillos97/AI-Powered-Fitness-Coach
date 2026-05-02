"""
╔══════════════════════════════════════════════════════════════════╗
║  LIVING DISTRIBUTION ARCHITECTURE — REAL-TIME INFERENCE v3      ║
║  Front Shoulder Shrug: Rep Detection + Form Classification      ║
║                                                                  ║
║  Environment: Windows, Python 3.10, TFLite, MediaPipe, OpenCV   ║
║  Camera: Index 1                                                 ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python realtime_inference.py

Controls:
    Q     — Quit
    R     — Reset session (clear rep count, reset state)
    S     — Screenshot
"""

import os
import sys
import json
import time
import numpy as np
import cv2

# ============================================================================
# STEP 1: CONFIGURE PATHS
# ============================================================================

MODEL_DIR = r"D:\AI-Powered-Fitness-Coach\AI Workout\Front_Shoulder\New folder"
CAMERA_INDEX = 1

REQUIRED_FILES = [
    'living_distribution_model_v2.tflite',
    'orbit_classifier_v2.tflite',
    'warm_start_state_v2.npy',
    'scaler_mean.npy',
    'scaler_std.npy',
    'orbit_scaler_mean_v2.npy',
    'orbit_scaler_std_v2.npy',
    'traj_scaler_mean.npy',
    'traj_scaler_std.npy',
    'perfect_orbit_reference_v2.npy',
    'physics_gates.json',
    'config.json',
    'class_mapping.json',
]

print("=" * 60)
print("LIVING DISTRIBUTION — REAL-TIME INFERENCE v3")
print("=" * 60)
print(f"\nModel directory: {MODEL_DIR}")
print("\nChecking files:")
all_found = True
for fname in REQUIRED_FILES:
    fpath = os.path.join(MODEL_DIR, fname)
    exists = os.path.exists(fpath)
    size = os.path.getsize(fpath) if exists else 0
    status = f"✓ {size / 1024:.1f} KB" if exists else "✗ MISSING"
    print(f"  {fname:>45} — {status}")
    if not exists:
        all_found = False

if not all_found:
    print("\n❌ Missing files! Cannot proceed.")
    sys.exit(1)

print("\n✓ All files found.")


# ============================================================================
# STEP 2: IMPORT MEDIAPIPE (Handle both API versions)
# ============================================================================

print("\nLoading MediaPipe...")
mp = None
mp_pose_module = None
mp_drawing_module = None
mp_drawing_styles = None

try:
    import mediapipe as mp
    try:
        mp_pose_module = mp.solutions.pose
        mp_drawing_module = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        print(f"  ✓ MediaPipe loaded (mp.solutions API) — v{mp.__version__}")
    except AttributeError:
        try:
            from mediapipe.python.solutions import pose as mp_pose_module
            from mediapipe.python.solutions import drawing_utils as mp_drawing_module
            mp_drawing_styles = None
            print(f"  ✓ MediaPipe loaded (mediapipe.python.solutions API) — v{mp.__version__}")
        except ImportError:
            print("  ✗ MediaPipe solutions not found!")
            print("    Try: pip uninstall mediapipe -y && pip cache purge && "
                  "pip install --no-cache-dir mediapipe==0.10.14")
            sys.exit(1)
except ImportError:
    print("  ✗ MediaPipe not installed!")
    print("    Install: pip install mediapipe==0.10.14")
    sys.exit(1)


# ============================================================================
# STEP 3: IMPORT TFLITE RUNTIME
# ============================================================================

print("\nLoading TFLite runtime...")
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    print("  ✓ Using tflite_runtime")
except ImportError:
    try:
        import tensorflow as tf
        TFLiteInterpreter = tf.lite.Interpreter
        print(f"  ✓ Using tf.lite.Interpreter (TF {tf.__version__})")
    except ImportError:
        print("  ✗ No inference runtime found!")
        print("    Install: pip install tensorflow or pip install tflite-runtime")
        sys.exit(1)


# ============================================================================
# STEP 4: LOAD ALL ARTIFACTS
# ============================================================================

print("\nLoading model artifacts...")


def load_npy(name):
    path = os.path.join(MODEL_DIR, name)
    data = np.load(path).astype(np.float32)
    print(f"  ✓ {name}: shape={data.shape}")
    return data


def load_json(name):
    path = os.path.join(MODEL_DIR, name)
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"  ✓ {name}: {len(data)} entries")
    return data


# Scalers
scaler_mean = load_npy('scaler_mean.npy')
scaler_std = load_npy('scaler_std.npy')
orbit_scaler_mean = load_npy('orbit_scaler_mean_v2.npy')
orbit_scaler_std = load_npy('orbit_scaler_std_v2.npy')
traj_scaler_mean = load_npy('traj_scaler_mean.npy')
traj_scaler_std = load_npy('traj_scaler_std.npy')

# References
warm_start_state = load_npy('warm_start_state_v2.npy')
perfect_orbit_reference = load_npy('perfect_orbit_reference_v2.npy')

# Configs
physics_gates = load_json('physics_gates.json')
config = load_json('config.json')
class_mapping_raw = load_json('class_mapping.json')

class_mapping = {int(k): v for k, v in class_mapping_raw.items()}
class_names = [class_mapping[i] for i in sorted(class_mapping.keys())]
n_classes = len(class_names)

print(f"\n  Classes: {class_names}")
print(f"  Physics gates: {physics_gates}")


# ============================================================================
# STEP 5: LOAD TFLITE MODELS + MAP I/O
# ============================================================================

print("\nLoading TFLite models...")

# --- Main model ---
main_interpreter = TFLiteInterpreter(
    model_path=os.path.join(MODEL_DIR, 'living_distribution_model_v2.tflite')
)
main_interpreter.allocate_tensors()

main_in = main_interpreter.get_input_details()
main_out = main_interpreter.get_output_details()

print(f"\n  Main model inputs ({len(main_in)}):")
main_input_map = {}
for d in main_in:
    print(f"    {d['name']:>30}: shape={d['shape']}, dtype={d['dtype']}")
    name_lower = d['name'].lower()
    if 'frame' in name_lower:
        main_input_map['frame'] = d['index']
    elif 'prev_state' in name_lower or ('state' in name_lower and 'prev' in name_lower):
        main_input_map['prev_state'] = d['index']
    elif 'prev_z' in name_lower:
        main_input_map['prev_z'] = d['index']

# Fallback: map by shape — frame is (1,132), others are (1,8)
if len(main_input_map) < 3:
    print("  ⚠ Name-based mapping incomplete, using shape-based mapping")
    eights = []
    for d in main_in:
        if d['shape'][1] == 132:
            main_input_map['frame'] = d['index']
        else:
            eights.append(d['index'])
    if len(eights) >= 2:
        main_input_map['prev_state'] = eights[0]
        main_input_map['prev_z'] = eights[1]

print(f"  Main input map: {main_input_map}")

print(f"\n  Main model outputs ({len(main_out)}):")
for d in main_out:
    print(f"    {d['name']:>30}: shape={tuple(d['shape'])}")

scalar_output_indices = [d['index'] for d in main_out if tuple(d['shape']) == (1, 1)]
vector_output_indices = [d['index'] for d in main_out if tuple(d['shape']) == (1, 8)]

print(f"  Scalar outputs (1,1): {len(scalar_output_indices)}")
print(f"  Vector outputs (1,8): {len(vector_output_indices)}")

# --- Classifier model ---
clf_interpreter = TFLiteInterpreter(
    model_path=os.path.join(MODEL_DIR, 'orbit_classifier_v2.tflite')
)
clf_interpreter.allocate_tensors()

clf_in = clf_interpreter.get_input_details()
clf_out = clf_interpreter.get_output_details()

print(f"\n  Classifier inputs ({len(clf_in)}):")
clf_input_map = {}
for d in clf_in:
    print(f"    {d['name']:>30}: shape={d['shape']}")
    if len(d['shape']) == 3:
        clf_input_map['trajectory'] = d['index']
    elif len(d['shape']) == 2:
        clf_input_map['scalar'] = d['index']

print(f"  Classifier input map: {clf_input_map}")
print(f"\n  Classifier outputs ({len(clf_out)}):")
for d in clf_out:
    print(f"    {d['name']:>30}: shape={d['shape']}")

print("\n✓ All models loaded and mapped.")


# ============================================================================
# STEP 6: FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def landmarks_to_features(pose_landmarks):
    """
    Convert MediaPipe pose landmarks to 132-dim feature vector.
    Order: x0, y0, z0, v0, x1, y1, z1, v1, ..., x32, y32, z32, v32
    """
    features = np.zeros(132, dtype=np.float32)
    for i, lm in enumerate(pose_landmarks.landmark):
        features[i * 4 + 0] = lm.x
        features[i * 4 + 1] = lm.y
        features[i * 4 + 2] = lm.z
        features[i * 4 + 3] = lm.visibility
    return features


def compute_angle(a, b, c):
    """Angle at vertex b in degrees."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def extract_physics_features(raw_frames_list):
    """
    Extract 6 physics features from a list of raw 132-dim frames.
    Must match training Cell 17-V3 EXACTLY.
    
    Features:
        0: F1 — Min Elbow Angle
        1: F2 — Shoulder ROM
        2: F3 — Max Shoulder Velocity
        3: F4 — Wrist-Hip Horizontal Spread
        4: F5 — Shoulder Elevation Asymmetry
        5: F6 — Elbow Angle Variance
    """
    raw_frames = np.stack(raw_frames_list)
    n_frames = len(raw_frames)

    def get_xyz(lm_idx):
        col = lm_idx * 4
        return raw_frames[:, col:col + 3]

    l_shoulder = get_xyz(11)
    r_shoulder = get_xyz(12)
    l_elbow = get_xyz(13)
    r_elbow = get_xyz(14)
    l_wrist = get_xyz(15)
    r_wrist = get_xyz(16)
    l_hip = get_xyz(23)
    r_hip = get_xyz(24)

    # F1: Min Elbow Angle
    elbow_angles_l = []
    elbow_angles_r = []
    for i in range(n_frames):
        elbow_angles_l.append(compute_angle(l_shoulder[i], l_elbow[i], l_wrist[i]))
        elbow_angles_r.append(compute_angle(r_shoulder[i], r_elbow[i], r_wrist[i]))
    elbow_angles_l = np.array(elbow_angles_l)
    elbow_angles_r = np.array(elbow_angles_r)
    f1 = np.min(np.minimum(elbow_angles_l, elbow_angles_r))

    # F2: Shoulder ROM
    mid_shoulder_y = (l_shoulder[:, 1] + r_shoulder[:, 1]) / 2.0
    mid_hip_y = (l_hip[:, 1] + r_hip[:, 1]) / 2.0
    relative_height = mid_hip_y - mid_shoulder_y
    f2 = np.max(relative_height) - np.min(relative_height)

    # F3: Max Shoulder Velocity
    mid_shoulder = (l_shoulder + r_shoulder) / 2.0
    if n_frames > 1:
        velocity = np.linalg.norm(np.diff(mid_shoulder, axis=0), axis=1)
        f3 = np.max(velocity)
    else:
        f3 = 0.0

    # F4: Wrist-Hip Horizontal Spread
    mid_hip_x = (l_hip[:, 0] + r_hip[:, 0]) / 2.0
    l_wrist_spread = np.abs(l_wrist[:, 0] - mid_hip_x)
    r_wrist_spread = np.abs(r_wrist[:, 0] - mid_hip_x)
    f4 = np.max((l_wrist_spread + r_wrist_spread) / 2.0)

    # F5: Shoulder Elevation Asymmetry
    l_shoulder_rom = np.max(l_shoulder[:, 1]) - np.min(l_shoulder[:, 1])
    r_shoulder_rom = np.max(r_shoulder[:, 1]) - np.min(r_shoulder[:, 1])
    f5 = np.abs(l_shoulder_rom - r_shoulder_rom)

    # F6: Elbow Angle Variance
    f6 = (np.var(elbow_angles_l) + np.var(elbow_angles_r)) / 2.0

    return np.array([f1, f2, f3, f4, f5, f6], dtype=np.float32)


def time_normalize(trajectory, target_length):
    """Resample trajectory to target_length frames via linear interpolation."""
    n = len(trajectory)
    if n < 2:
        return np.zeros((target_length, trajectory.shape[1]), dtype=np.float32)
    indices = np.linspace(0, n - 1, target_length)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, n - 1)
    frac = (indices - idx_floor)[:, np.newaxis]
    return ((1 - frac) * trajectory[idx_floor] + frac * trajectory[idx_ceil]).astype(np.float32)


def compute_orbit_area_2d(traj_2d):
    """Shoelace formula for orbit area."""
    x, y = traj_2d[:, 0], traj_2d[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_frechet_deviation(trajectory, reference):
    """Simplified Fréchet-like distance."""
    n_ref, n_traj = len(reference), len(trajectory)
    if n_traj < 2 or n_ref < 2:
        return 0.0
    indices = np.linspace(0, n_traj - 1, n_ref).astype(int)
    return np.mean(np.linalg.norm(trajectory[indices] - reference, axis=1))


def extract_orbit_features(mu_trajectory):
    """
    Extract 15 orbit geometry features from μ trajectory.
    Must match training Cell 17-V3 EXACTLY.
    
    Features:
         0: orbit_area (2D PCA)
         1: orbit_max_radius
         2: orbit_smoothness
         3: orbit_symmetry
         4: orbit_frechet (filled later)
         5: orbit_mean_radius
         6: orbit_velocity_mean
         7: orbit_velocity_std
         8: orbit_velocity_max
         9: orbit_curvature
        10: orbit_pca_ratio
        11: orbit_total_distance
        12: orbit_closure
        13: orbit_dim_variance_max
        14: orbit_dim_variance_min
    """
    from sklearn.decomposition import PCA

    if len(mu_trajectory) < 5:
        return np.zeros(15, dtype=np.float32)

    # 0: area via PCA to 2D
    pca = PCA(n_components=2)
    traj_2d = pca.fit_transform(mu_trajectory)
    area = compute_orbit_area_2d(traj_2d)

    # 1: max radius from center
    center = mu_trajectory.mean(axis=0)
    dists = np.linalg.norm(mu_trajectory - center, axis=1)
    max_radius = dists.max()

    # 2: smoothness (mean 2nd derivative norm)
    if len(mu_trajectory) >= 3:
        d2 = np.diff(mu_trajectory, n=2, axis=0)
        smoothness = np.mean(np.linalg.norm(d2, axis=1))
    else:
        smoothness = 0.0

    # 3: symmetry (first half vs reversed second half)
    n = len(mu_trajectory)
    mid = n // 2
    fh = mu_trajectory[:mid]
    sh = mu_trajectory[mid:2 * mid][::-1]
    ml = min(len(fh), len(sh))
    symmetry = np.mean(np.linalg.norm(fh[:ml] - sh[:ml], axis=1)) if ml > 0 else 0.0

    # 4: frechet deviation (placeholder — filled after this function)
    frechet = 0.0

    # 5: mean radius
    mean_radius = dists.mean()

    # 6, 7, 8: velocity statistics
    if len(mu_trajectory) >= 2:
        vels = np.linalg.norm(np.diff(mu_trajectory, axis=0), axis=1)
        vel_mean, vel_std, vel_max = vels.mean(), vels.std(), vels.max()
    else:
        vel_mean, vel_std, vel_max = 0.0, 0.0, 0.0

    # 9: curvature (mean angle change between consecutive velocity vectors)
    if len(mu_trajectory) >= 3:
        d1 = np.diff(mu_trajectory, axis=0)
        norms = np.linalg.norm(d1, axis=1, keepdims=True) + 1e-8
        d1n = d1 / norms
        dots = np.sum(d1n[:-1] * d1n[1:], axis=1)
        dots = np.clip(dots, -1, 1)
        curvature = np.mean(np.arccos(dots))
    else:
        curvature = 0.0

    # 10: PCA ratio (elongation)
    pca_full = PCA(n_components=min(mu_trajectory.shape[1], 3))
    pca_full.fit(mu_trajectory)
    evr = pca_full.explained_variance_ratio_
    pca_ratio = evr[0] / (evr[1] + 1e-8) if len(evr) >= 2 else 0.0

    # 11: total distance traveled
    total_dist = np.sum(np.linalg.norm(np.diff(mu_trajectory, axis=0), axis=1)) \
        if len(mu_trajectory) >= 2 else 0.0

    # 12: closure (distance between start and end)
    closure = np.linalg.norm(mu_trajectory[-1] - mu_trajectory[0]) \
        if len(mu_trajectory) >= 2 else 0.0

    # 13, 14: per-dimension variance extremes
    dim_var = np.var(mu_trajectory, axis=0)
    dim_var_max = dim_var.max()
    dim_var_min = dim_var.min()

    return np.array([
        area, max_radius, smoothness, symmetry, frechet,
        mean_radius, vel_mean, vel_std, vel_max,
        curvature, pca_ratio, total_dist, closure,
        dim_var_max, dim_var_min
    ], dtype=np.float32)


# ============================================================================
# STEP 7: CYCLE (REP) DETECTION — ROBUST STATE MACHINE
# ============================================================================

class RepDetector:
    """
    Detects rep boundaries from cycle_prob using robust hysteresis.

    State machine:
        IDLE     → RISING   (when smoothed prob > rise_threshold)
        RISING   → PEAKED   (after sustained decrease AND peak high enough)
        PEAKED   → TRIGGER  (when prob drops to fraction of peak)
        TRIGGER  → COOLDOWN (immediate)
        COOLDOWN → IDLE     (when prob returns near zero)

    Protections:
        - Smoothing window filters noise
        - Minimum rising duration prevents micro-triggers
        - Minimum peak height prevents false positives
        - Fall ratio (not absolute threshold) adapts to signal scale
        - Cooldown prevents double-triggers
        - First 30 frames ignored (cold start)
    """

    def __init__(self, rise_threshold=0.25, min_peak_value=0.35,
                 min_rising_frames=5, min_falling_frames=3,
                 min_frames_between=25, fall_ratio=0.4,
                 smooth_window=5):
        self.rise_threshold = rise_threshold
        self.min_peak_value = min_peak_value
        self.min_rising_frames = min_rising_frames
        self.min_falling_frames = min_falling_frames
        self.min_frames_between = min_frames_between
        self.fall_ratio = fall_ratio
        self.smooth_window = smooth_window

        self.state = 'IDLE'
        self.peak_value = 0.0
        self.rising_count = 0
        self.falling_count = 0
        self.frames_since_last = 0
        self.total_frames = 0
        self.prev_prob = 0.0
        self.prob_history = []

    def _smooth(self, raw_prob):
        """Moving average to filter noise."""
        self.prob_history.append(raw_prob)
        if len(self.prob_history) > self.smooth_window:
            self.prob_history.pop(0)
        return float(np.mean(self.prob_history))

    def update(self, raw_cycle_prob):
        """Feed one frame's cycle_prob. Returns True if rep completed."""
        self.total_frames += 1
        self.frames_since_last += 1

        # Smooth the signal
        cycle_prob = self._smooth(raw_cycle_prob)

        # Ignore first 30 frames (model cold start settling)
        if self.total_frames < 30:
            self.prev_prob = cycle_prob
            return False

        triggered = False

        if self.state == 'IDLE':
            if cycle_prob > self.rise_threshold:
                self.state = 'RISING'
                self.peak_value = cycle_prob
                self.rising_count = 1
                self.falling_count = 0

        elif self.state == 'RISING':
            if cycle_prob >= self.prev_prob:
                # Still rising
                self.rising_count += 1
                if cycle_prob > self.peak_value:
                    self.peak_value = cycle_prob
                self.falling_count = 0
            else:
                # Started falling
                self.falling_count += 1

                # Transition to PEAKED only if:
                #   1. Peak was high enough
                #   2. We were rising long enough
                #   3. We've been falling consistently
                if (self.falling_count >= self.min_falling_frames and
                        self.peak_value >= self.min_peak_value and
                        self.rising_count >= self.min_rising_frames):
                    self.state = 'PEAKED'
                elif self.falling_count > 15:
                    # False alarm — never reached a real peak
                    self.state = 'IDLE'
                    self.peak_value = 0.0
                    self.rising_count = 0
                    self.falling_count = 0

        elif self.state == 'PEAKED':
            # Wait for signal to drop to fraction of peak value
            drop_threshold = self.peak_value * self.fall_ratio
            if cycle_prob < drop_threshold:
                # Rep completed
                if self.frames_since_last >= self.min_frames_between:
                    triggered = True
                    self.frames_since_last = 0
                # Enter cooldown
                self.state = 'COOLDOWN'
                self.peak_value = 0.0
                self.rising_count = 0
                self.falling_count = 0

        elif self.state == 'COOLDOWN':
            # Must return to near-zero before allowing next rep detection
            if cycle_prob < 0.1:
                self.state = 'IDLE'

        self.prev_prob = cycle_prob
        return triggered

    def reset(self):
        self.state = 'IDLE'
        self.peak_value = 0.0
        self.rising_count = 0
        self.falling_count = 0
        self.frames_since_last = 0
        self.total_frames = 0
        self.prev_prob = 0.0
        self.prob_history = []


# ============================================================================
# STEP 8: SESSION STATE
# ============================================================================

class SessionState:
    """Manages all real-time state for one workout session."""

    def __init__(self, warm_start, latent_dim=8):
        self.latent_dim = latent_dim

        # SSM state (carried across frames)
        self.current_state = warm_start.copy().reshape(1, latent_dim)
        self.prev_z = np.zeros((1, latent_dim), dtype=np.float32)
        self.warm_start = warm_start.copy()

        # Current rep buffers
        self.rep_raw_frames = []
        self.rep_mu_trajectory = []
        self.global_frame_count = 0

        # Session stats
        self.rep_count = 0
        self.rep_results = []
        self.last_classification = None
        self.last_confidence = 0.0

        # Real-time signals
        self.current_phase = 0.0
        self.current_cycle_prob = 0.0
        self.current_cycle_prob_raw = 0.0
        self.current_halluc_error = 0.0
        self.form_quality = 1.0

        # Rep detector
        self.rep_detector = RepDetector(
            rise_threshold=0.25,
            min_peak_value=0.35,
            min_rising_frames=5,
            min_falling_frames=3,
            min_frames_between=25,
            fall_ratio=0.4,
            smooth_window=5
        )

        # Form quality smoothing
        self.halluc_error_history = []
        self.HALLUC_SMOOTH_WINDOW = 15

        # Flash effect counter
        self._flash_frames = 0

    def reset(self):
        """Full session reset."""
        self.__init__(self.warm_start, self.latent_dim)

    def update_form_quality(self, halluc_error):
        """Exponential moving average of form quality."""
        self.halluc_error_history.append(halluc_error)
        if len(self.halluc_error_history) > self.HALLUC_SMOOTH_WINDOW:
            self.halluc_error_history.pop(0)
        avg_error = np.mean(self.halluc_error_history)
        self.form_quality = max(0.0, min(1.0, 1.0 - avg_error * 2.0))


# ============================================================================
# STEP 9: MAIN MODEL INFERENCE (Single Frame)
# ============================================================================

def run_main_model(frame_features_scaled, session):
    """
    Run one frame through the main TFLite model.
    Updates session state in-place.
    Returns raw cycle_prob.
    """
    frame_input = frame_features_scaled.reshape(1, 132).astype(np.float32)
    state_input = session.current_state.astype(np.float32)
    prev_z_input = session.prev_z.astype(np.float32)

    # Set inputs
    main_interpreter.set_tensor(main_input_map['frame'], frame_input)
    main_interpreter.set_tensor(main_input_map['prev_state'], state_input)
    main_interpreter.set_tensor(main_input_map['prev_z'], prev_z_input)

    # Invoke
    main_interpreter.invoke()

    # Gather all outputs
    all_outputs = []
    for d in main_out:
        all_outputs.append(main_interpreter.get_tensor(d['index']).copy())

    # Separate by shape
    scalars = [o for o in all_outputs if o.shape == (1, 1)]
    vectors = [o for o in all_outputs if o.shape == (1, 8)]

    # Parse scalars: order is phase_sin, phase_cos, cycle_prob, halluc_error
    if len(scalars) >= 4:
        phase_sin = float(scalars[0][0, 0])
        phase_cos = float(scalars[1][0, 0])
        cycle_prob = float(scalars[2][0, 0])
        halluc_error = float(scalars[3][0, 0])
    elif len(scalars) >= 3:
        phase_sin = float(scalars[0][0, 0])
        phase_cos = float(scalars[1][0, 0])
        cycle_prob = float(scalars[2][0, 0])
        halluc_error = 0.0
    else:
        phase_sin, phase_cos, cycle_prob, halluc_error = 0.0, 1.0, 0.0, 0.0

    # Parse vectors: order is new_state, z_current, z_hat_next
    if len(vectors) >= 3:
        new_state = vectors[0]
        z_current = vectors[1]
        z_hat_next = vectors[2]
    elif len(vectors) >= 1:
        new_state = vectors[0]
        z_current = vectors[0]
        z_hat_next = vectors[0]
    else:
        new_state = session.current_state
        z_current = session.prev_z
        z_hat_next = session.prev_z

    # Update session state for next frame
    session.current_state = new_state.copy()
    session.prev_z = z_hat_next.copy()

    # Update display signals
    session.current_phase = float(np.arctan2(phase_sin, phase_cos))
    session.current_cycle_prob_raw = cycle_prob
    session.current_cycle_prob = cycle_prob
    session.current_halluc_error = halluc_error
    session.update_form_quality(halluc_error)

    # Store μ for orbit trajectory
    session.rep_mu_trajectory.append(new_state.flatten().copy())
    session.global_frame_count += 1

    return cycle_prob


# ============================================================================
# STEP 10: ORBIT CLASSIFIER INFERENCE (Post-Rep)
# ============================================================================

def classify_rep(session):
    """
    Called when a rep is detected.
    Extracts orbit + physics features, runs TCN classifier.

    Returns:
        class_name: str or None
        confidence: float
        accepted: bool
    """
    raw_frames = session.rep_raw_frames
    mu_traj = session.rep_mu_trajectory

    n_frames = len(raw_frames)
    n_mu = len(mu_traj)

    # === Gate 1: Minimum frame count ===
    min_frames = max(int(physics_gates.get('min_rep_frames', 20)), 30)
    if n_frames < min_frames:
        print(f"    [REJECT] Too few frames: {n_frames} < {min_frames}")
        return None, 0.0, False

    if n_mu < 10:
        print(f"    [REJECT] Too few μ states: {n_mu}")
        return None, 0.0, False

    # === Gate 2: ROM threshold ===
    physics_feats = extract_physics_features(raw_frames)
    rom = physics_feats[1]
    rom_threshold = float(physics_gates.get('rom_threshold', 0.001))
    if rom < rom_threshold:
        print(f"    [REJECT] ROM too small: {rom:.4f} < {rom_threshold:.4f}")
        return None, 0.0, False

    # === Gate 3: μ trajectory has actual variation ===
    mu_traj_arr = np.stack(mu_traj)
    if np.std(mu_traj_arr) < 1e-6:
        print(f"    [REJECT] μ trajectory is flat")
        return None, 0.0, False

    # === Extract 15 orbit features ===
    orbit_feats = extract_orbit_features(mu_traj_arr)
    orbit_feats[4] = compute_frechet_deviation(mu_traj_arr, perfect_orbit_reference)

    # === Combine scalars: 15 orbit + 6 physics = 21 ===
    combined = np.concatenate([orbit_feats, physics_feats]).reshape(1, -1)
    combined_scaled = ((combined - orbit_scaler_mean) / (orbit_scaler_std + 1e-8)).astype(np.float32)

    # === Time-normalize trajectory for TCN ===
    mu_traj_norm = time_normalize(mu_traj_arr, 50)
    mu_traj_norm_scaled = ((mu_traj_norm - traj_scaler_mean) / traj_scaler_std).astype(np.float32)
    traj_input = mu_traj_norm_scaled[np.newaxis, :, :]  # (1, 50, 8)

    # === Run TCN classifier ===
    clf_interpreter.set_tensor(clf_input_map['trajectory'], traj_input)
    clf_interpreter.set_tensor(clf_input_map['scalar'], combined_scaled)
    clf_interpreter.invoke()

    probs = clf_interpreter.get_tensor(clf_out[0]['index'])[0]
    predicted_idx = int(np.argmax(probs))
    confidence = float(probs[predicted_idx])
    predicted_class = class_names[predicted_idx]

    # === Debug output ===
    print(f"    [DEBUG] Frames: {n_frames}, μ states: {n_mu}")
    print(f"    [DEBUG] Physics: F1={physics_feats[0]:.1f}° F2(ROM)={physics_feats[1]:.4f} "
          f"F3(vel)={physics_feats[2]:.4f} F4={physics_feats[3]:.4f} "
          f"F5={physics_feats[4]:.4f} F6={physics_feats[5]:.1f}")
    print(f"    [DEBUG] Orbit: area={orbit_feats[0]:.3f} maxR={orbit_feats[1]:.3f} "
          f"smooth={orbit_feats[2]:.4f} dev={orbit_feats[4]:.3f}")
    print(f"    [DEBUG] Probs: {dict(zip(class_names, [f'{p:.2f}' for p in probs]))}")

    return predicted_class, confidence, True


# ============================================================================
# STEP 11: OVERLAY DRAWING
# ============================================================================

COLORS = {
    'perfect': (0, 200, 0),
    'low': (0, 165, 255),
    'over_range': (0, 0, 255),
    'bent_elbow': (255, 0, 100),
    'bg_panel': (30, 30, 30),
    'text': (255, 255, 255),
    'phase_ring': (200, 200, 0),
}

STATE_COLORS = {
    'IDLE': (150, 150, 150),
    'RISING': (0, 255, 255),
    'PEAKED': (0, 165, 255),
    'COOLDOWN': (255, 100, 0),
}


def draw_overlay(frame, session, fps):
    """Draw all HUD elements on the frame."""
    h, w = frame.shape[:2]

    # ==========================================
    # LEFT PANEL: Rep count + classification
    # ==========================================
    panel_w = 300
    panel_h = 230
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), COLORS['bg_panel'], -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Rep count (large)
    cv2.putText(frame, f"REPS: {session.rep_count}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS['text'], 3)

    # Last classification
    if session.last_classification:
        cls = session.last_classification
        color = COLORS.get(cls, COLORS['text'])
        cv2.putText(frame, f"Form: {cls.upper()}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Conf: {session.last_confidence:.0%}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Detector state
    det = session.rep_detector
    state_color = STATE_COLORS.get(det.state, (150, 150, 150))
    cv2.putText(frame, f"State: {det.state}", (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)

    # Buffer + peak info
    cv2.putText(frame,
                f"Buf: {len(session.rep_raw_frames)}f  Peak: {det.peak_value:.2f}",
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

    # ==========================================
    # RIGHT PANEL: Real-time signals
    # ==========================================
    right_x = w - 270
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (right_x - 10, 10), (w - 10, 280), COLORS['bg_panel'], -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    # --- Phase indicator (circular) ---
    phase_center = (right_x + 40, 60)
    phase_radius = 30
    cv2.circle(frame, phase_center, phase_radius, (80, 80, 80), 2)
    px = int(phase_center[0] + phase_radius * np.cos(session.current_phase))
    py = int(phase_center[1] + phase_radius * np.sin(session.current_phase))
    cv2.circle(frame, (px, py), 6, COLORS['phase_ring'], -1)
    cv2.putText(frame, "Phase", (right_x + 80, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)

    # --- Bar drawing helper ---
    bar_w = 180
    bar_h = 18

    def draw_bar(y_pos, label, value, max_val=1.0, color=(0, 200, 200)):
        cv2.putText(frame, label, (right_x, y_pos - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)
        cv2.rectangle(frame, (right_x, y_pos), (right_x + bar_w, y_pos + bar_h),
                      (60, 60, 60), -1)
        fill = int(bar_w * min(value / max_val, 1.0))
        cv2.rectangle(frame, (right_x, y_pos), (right_x + fill, y_pos + bar_h),
                      color, -1)
        cv2.putText(frame, f"{value:.3f}", (right_x + bar_w + 5, y_pos + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)

    # Cycle probability bar
    prob_raw = session.current_cycle_prob_raw
    if prob_raw < 0.25:
        prob_color = (100, 100, 100)
    elif prob_raw < 0.4:
        prob_color = (0, 255, 255)
    else:
        prob_color = (0, 0, 255)
    draw_bar(110, "P_cycle", prob_raw, 1.0, prob_color)

    # Form quality bar
    fq = session.form_quality
    fq_r = int(255 * (1 - fq))
    fq_g = int(255 * fq)
    draw_bar(155, "Form Q", fq, 1.0, (0, fq_g, fq_r))

    # Hallucinator error bar
    draw_bar(200, "Deviation", session.current_halluc_error, 0.5, (0, 100, 255))

    # Buffer size
    cv2.putText(frame, f"Buffer: {len(session.rep_raw_frames)} frames",
                (right_x, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # ==========================================
    # BOTTOM: Rep history
    # ==========================================
    if session.rep_results:
        hist_y = h - 40
        cv2.putText(frame, "History:", (10, hist_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
        for i, result in enumerate(session.rep_results[-8:]):
            cls = result['class']
            color = COLORS.get(cls, COLORS['text'])
            x_pos = 100 + i * 95
            label = cls[:5].upper()
            cv2.putText(frame, label, (x_pos, hist_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    # ==========================================
    # FLASH EFFECT on new rep
    # ==========================================
    if session._flash_frames > 0:
        flash_alpha = session._flash_frames / 15.0
        flash_color = COLORS.get(session.last_classification, (0, 255, 0))
        flash_overlay = frame.copy()
        cv2.rectangle(flash_overlay, (0, 0), (w, h), flash_color, -1)
        cv2.addWeighted(flash_overlay, flash_alpha * 0.15, frame, 1.0, 0, frame)
        session._flash_frames -= 1

    return frame


# ============================================================================
# STEP 12: MAIN LOOP
# ============================================================================

MAX_REP_BUFFER = 300  # 10 seconds at 30fps — flush if no rep detected


def main():
    print("\n" + "=" * 60)
    print("STARTING REAL-TIME INFERENCE")
    print("=" * 60)
    print(f"Camera index: {CAMERA_INDEX}")
    print("Controls: Q=Quit, R=Reset, S=Screenshot")
    print("=" * 60)

    # --- Open camera ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"\n❌ Cannot open camera {CAMERA_INDEX}")
        print("   Try changing CAMERA_INDEX to 0 or 2")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")

    # --- Initialize MediaPipe Pose ---
    pose = mp_pose_module.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # --- Initialize session ---
    session = SessionState(warm_start_state, latent_dim=8)

    # --- FPS tracking ---
    fps = 0.0
    frame_times = []
    frame_count = 0

    print("\n🟢 Running... Press Q to quit.\n")

    try:
        while True:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            # Mirror for natural feedback
            frame = cv2.flip(frame, 1)

            # --- MediaPipe Pose Detection ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            if results.pose_landmarks:
                # Draw skeleton
                try:
                    if mp_drawing_styles is not None:
                        mp_drawing_module.draw_landmarks(
                            frame, results.pose_landmarks,
                            mp_pose_module.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                    else:
                        mp_drawing_module.draw_landmarks(
                            frame, results.pose_landmarks,
                            mp_pose_module.POSE_CONNECTIONS
                        )
                except Exception:
                    # If drawing fails, just skip it
                    pass

                # --- Extract features ---
                raw_features = landmarks_to_features(results.pose_landmarks)
                scaled_features = (raw_features - scaler_mean) / (scaler_std + 1e-8)

                # Store raw frame in rep buffer
                session.rep_raw_frames.append(raw_features.copy())

                # --- Run main model ---
                cycle_prob = run_main_model(scaled_features, session)

                # --- Buffer overflow protection ---
                if len(session.rep_raw_frames) > MAX_REP_BUFFER:
                    session.rep_raw_frames = session.rep_raw_frames[-60:]
                    session.rep_mu_trajectory = session.rep_mu_trajectory[-60:]
                    print("  [INFO] Buffer overflow — flushed old frames")

                # --- Check for rep completion ---
                rep_triggered = session.rep_detector.update(cycle_prob)

                if rep_triggered:
                    print(f"\n  === REP TRIGGERED at frame {frame_count} ===")

                    pred_class, confidence, accepted = classify_rep(session)

                    if accepted and pred_class is not None:
                        session.rep_count += 1
                        session.last_classification = pred_class
                        session.last_confidence = confidence
                        session.rep_results.append({
                            'class': pred_class,
                            'confidence': confidence,
                            'frames': len(session.rep_raw_frames)
                        })
                        session._flash_frames = 15

                        print(f"  ✅ Rep #{session.rep_count}: {pred_class.upper()} "
                              f"({confidence:.0%}) — {len(session.rep_raw_frames)} frames")
                    else:
                        print(f"  ⛔ Rep rejected by physics gates")

                    # Keep small tail for continuity (next rep's start)
                    tail_size = 10
                    session.rep_raw_frames = session.rep_raw_frames[-tail_size:]
                    session.rep_mu_trajectory = session.rep_mu_trajectory[-tail_size:]

            else:
                # No pose detected
                cv2.putText(frame, "No pose detected - step into frame",
                            (frame.shape[1] // 2 - 250, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # --- FPS ---
            t_end = time.time()
            frame_times.append(t_end - t_start)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / (np.mean(frame_times) + 1e-8)

            # --- Draw overlay ---
            frame = draw_overlay(frame, session, fps)

            # --- Display ---
            cv2.imshow('Living Distribution — Front Shoulder Shrug', frame)

            # --- Keyboard controls ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n🔴 Quit requested.")
                break
            elif key == ord('r') or key == ord('R'):
                session.reset()
                print("\n🔄 Session reset.")
            elif key == ord('s') or key == ord('S'):
                screenshot_path = os.path.join(
                    MODEL_DIR, f"screenshot_{int(time.time())}.png"
                )
                cv2.imwrite(screenshot_path, frame)
                print(f"📸 Screenshot saved: {screenshot_path}")

    except KeyboardInterrupt:
        print("\n🔴 Interrupted.")

    finally:
        # --- Cleanup ---
        pose.close()
        cap.release()
        cv2.destroyAllWindows()

        # --- Session summary ---
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        print(f"Total reps detected:    {session.rep_count}")
        if session.rep_results:
            class_counts = {}
            for r in session.rep_results:
                cls = r['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            print("\nClass breakdown:")
            for cls, count in sorted(class_counts.items()):
                print(f"  {cls:>12}: {count} reps")
            avg_conf = np.mean([r['confidence'] for r in session.rep_results])
            avg_frames = np.mean([r['frames'] for r in session.rep_results])
            print(f"\n  Avg confidence:  {avg_conf:.0%}")
            print(f"  Avg frames/rep:  {avg_frames:.0f}")
        print("=" * 60)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    main()