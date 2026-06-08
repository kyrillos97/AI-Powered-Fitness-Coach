"""
realtime_shrug_test.py  (revised)
==================================
Changes from original:
  1. Pre-trigger sliding window (PRE_TRIGGER_FRAMES frames captured before the
     shoulder crosses the start line and prepended to every rep buffer).
  2. Two reference lines (SHOULDER + ELBOW) anchored to the NOSE landmark —
     the closest stable face point to the shoulders.  The nose barely moves
     during a shrug, so lines are rock-steady.  If the user steps closer /
     farther the nose shifts → lines shift by exactly the same amount.
  3. Elbow-spread indicator line drawn between both elbows; turns red when
     spread exceeds calibrated baseline (signals bent-elbow form).
  4. Clean VAE OOD gate: every completed motion goes through the VAE;
     in-distribution → classifier label, out-of-distribution → "unknown".
  5. All tuneable constants consolidated in one CONFIG block at the top.

Anchor logic  (MediaPipe Y: 0 = top of frame, 1 = bottom):
  calibration  →  nose_sho_gap  = mean(sho_y)  − mean(nose_y)   > 0
  runtime line →  shoulder_line_y = smoothed_nose_y + nose_sho_gap
  START record →  sho_y_now  <  shoulder_line_y   (shoulder rose above the line)
  STOP  record →  sho_y_now  >= shoulder_line_y   for STOP_CONFIRM_FRAMES frames

Keys:  Q / ESC = quit    R = reset
"""

import argparse
import collections
import os
import numpy as np
import cv2
import mediapipe as mp

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║               C O N F I G  —  edit only this section                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Paths & hardware ──────────────────────────────────────────────────────────
MODEL_PATH   = r"D:\AI-Powered-Fitness-Coach\AI Workout\shrugs\shrug_classifier_fp16.tflite"
LABELS_PATH  = r"D:\AI-Powered-Fitness-Coach\AI Workout\shrugs\shrug_label_classes.txt"
CAMERA_ID    = 0
CAM_WIDTH    = 640
CAM_HEIGHT   = 480

# ── MediaPipe pose detector ────────────────────────────────────────────────────
MEDIAPIPE_DET_CONF   = 0.5
MEDIAPIPE_TRACK_CONF = 0.5

# ── Calibration ────────────────────────────────────────────────────────────────
BASELINE_FRAMES = 40         # stand-still frames at startup (~2 s at 30 fps)

# ── Trigger ───────────────────────────────────────────────────────────────────
# START: shoulder Y < shoulder_line_y  (no multiplier needed — line IS the boundary)
# STOP : shoulder stays below line for this many consecutive frames
STOP_CONFIRM_FRAMES = 5      # lower → stops faster; raise if reps cut off early

# ── Nose-anchor smoothing ──────────────────────────────────────────────────────
# The line follows a low-pass filtered nose Y so small head bobs don't jitter it.
# 0.05 = very smooth / slow tracking    0.3 = faster but slightly noisier
NOSE_SMOOTH_ALPHA = 0.05

# ── Pre-trigger sliding window ─────────────────────────────────────────────────
PRE_TRIGGER_FRAMES = 10

# ── Rep quality guards ─────────────────────────────────────────────────────────
MIN_FRAMES    = 8
ARM_MAX_ANGLE = 80.0

# ── Feature pipeline ──────────────────────────────────────────────────────────
SMOOTH_ALPHA  = 0.5

# ── Elbow-spread indicator ─────────────────────────────────────────────────────
ELBOW_BENT_SPREAD_MULT = 1.1

# ── VAE OOD gate ───────────────────────────────────────────────────────────────
VAE_ENABLED              = True
VAE_KL_THRESHOLD_OVERRIDE = None  # set a float (e.g. 10.0) to override file value

# ── Display ────────────────────────────────────────────────────────────────────
HISTORY_MAXLEN = 5
COLORS = {
    "perfect":    ( 50, 200,  80),
    "bent_elbow": (  0, 100, 255),
    "unknown":    (120, 120, 120),
}

# ── Reference line colours ─────────────────────────────────────────────────────
COLOR_SHOULDER_LINE = (  0, 255, 180)  # teal-green  — calibrated shoulder height
COLOR_ELBOW_LINE    = (200, 200,   0)  # yellow      — calibrated elbow height

# ── Elbow spread line colours ──────────────────────────────────────────────────
COLOR_ELBOW_CALIB = (150, 150, 150)
COLOR_ELBOW_OK    = ( 50, 200,  80)
COLOR_ELBOW_BENT  = (  0,  60, 255)

# ── Internal joint indices ─────────────────────────────────────────────────────
JOINT_IDX = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]

# ╚═══════════════════════════════════════════════════════════════════════════╝


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE PIPELINE  (identical to training — do not modify)
# ─────────────────────────────────────────────────────────────────────────────

def angle_batch(a, b, c):
    ba  = a - b;  bc = c - b
    dot = np.sum(ba * bc, axis=-1)
    n   = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-6
    return np.degrees(np.arccos(np.clip(dot / n, -1.0, 1.0)))[:, None]


def ema_smooth(seq, alpha):
    out = [seq[0]]
    for i in range(1, len(seq)):
        out.append(alpha * seq[i] + (1.0 - alpha) * out[-1])
    return np.array(out, dtype=np.float32)


def resample(seq, target):
    T = len(seq)
    if T == target:
        return seq.astype(np.float32)
    idx = np.linspace(0, T - 1, target)
    out = np.zeros((target, seq.shape[1]), dtype=np.float32)
    for f in range(seq.shape[1]):
        out[:, f] = np.interp(idx, np.arange(T), seq[:, f])
    return out


def process_rep(frames, target_frames):
    T    = len(frames)
    norm = ema_smooth(frames, SMOOTH_ALPHA)
    elbow_l  = angle_batch(norm[:, 11], norm[:, 13], norm[:, 15])
    elbow_r  = angle_batch(norm[:, 12], norm[:, 14], norm[:, 16])
    sho_y_l  = -norm[:, 11, 1];  sho_y_r = -norm[:, 12, 1]
    sho_elev = ((sho_y_l + sho_y_r) / 2.0)[:, None]
    sho_vel  = np.concatenate([np.zeros((1,1), np.float32), np.diff(sho_elev, axis=0)], axis=0)
    sho_acc  = np.concatenate([np.zeros((1,1), np.float32), np.diff(sho_vel,  axis=0)], axis=0)
    min_el_l = float(np.min(elbow_l));  max_el_l = float(np.max(elbow_l))
    min_el_r = float(np.min(elbow_r));  max_el_r = float(np.max(elbow_r))
    ev_l     = np.abs(np.diff(elbow_l[:, 0]));  ev_r = np.abs(np.diff(elbow_r[:, 0]))
    sho_max  = float(np.max(sho_elev))
    ear_y_l  = -norm[:, 7, 1];   ear_y_r = -norm[:, 8, 1]
    wrist_l  = -norm[:, 15, 1];  wrist_r = -norm[:, 16, 1]
    nose_y   = -norm[:, 0, 1]
    peak_f   = int(np.argmax(sho_elev[:, 0]))
    t_l      = float(np.argmin(elbow_l[:, 0])) / max(T - 1, 1)
    t_r      = float(np.argmin(elbow_r[:, 0])) / max(T - 1, 1)
    el_mean  = float(np.mean(elbow_l));  el_med = float(np.median(elbow_l))
    el_std   = float(np.std(elbow_l)) + 1e-6
    depth_feat = np.array([
        min_el_l, min_el_r, max_el_l - min_el_l, max_el_r - min_el_r,
        float(np.max(np.abs(elbow_l - elbow_r))),
        sho_max, sho_max - float(np.min(sho_elev)),
        float(np.min((np.abs(sho_y_l - ear_y_l) + np.abs(sho_y_r - ear_y_r)) / 2.0)),
        float(np.max(np.maximum(wrist_l - wrist_l[0], wrist_r - wrist_r[0]))),
        float(np.max(np.abs(sho_vel))), float(np.mean(np.abs(sho_acc))),
        float(np.max(ev_l)) if len(ev_l) > 0 else 0.0,
        float(np.max(ev_r)) if len(ev_r) > 0 else 0.0,
        float(np.std(sho_y_l - sho_y_r)),
        float(np.max(nose_y) - np.min(nose_y)),
        float(np.mean((np.abs(norm[:, 15, 1]) + np.abs(norm[:, 16, 1])) / 2.0)),
        float(abs(sho_y_l[peak_f] - sho_y_r[peak_f])),
        float(np.max(np.abs(norm[:, 15, 0] - norm[:, 16, 0]))),
        float(abs(float(elbow_l[peak_f, 0]) - float(elbow_r[peak_f, 0]))),
        t_l, t_r, (el_mean - el_med) / el_std, float(abs(t_l - t_r)),
    ], dtype=np.float32)
    pos      = norm[:, JOINT_IDX, :].reshape(T, -1)
    vel      = np.concatenate([np.zeros_like(pos[:1]), np.diff(pos, axis=0)], axis=0)
    acc      = np.concatenate([np.zeros_like(vel[:1]), np.diff(vel, axis=0)], axis=0)
    seq_feat = np.concatenate(
        [pos, vel, acc, elbow_l, elbow_r, sho_elev, sho_vel, sho_acc], axis=1
    ).astype(np.float32)
    return resample(seq_feat, target_frames), depth_feat


# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_sho_y(lm):
    """Mean shoulder Y (MediaPipe normalised: 0=top, 1=bottom)."""
    return float((lm[11, 1] + lm[12, 1]) / 2.0)

def get_elbow_y(lm):
    """Mean elbow Y."""
    return float((lm[13, 1] + lm[14, 1]) / 2.0)

def get_nose_y(lm):
    """Nose Y (landmark 0) — stable anchor during shrugs."""
    return float(lm[0, 1])

def get_elbow_spread_ratio(lm):
    """Elbow-to-elbow / shoulder-to-shoulder distance ratio (scale-invariant)."""
    e_dist = float(np.linalg.norm(lm[13, :2] - lm[14, :2]))
    s_dist = float(np.linalg.norm(lm[11, :2] - lm[12, :2])) + 1e-6
    return e_dist / s_dist

def upper_arm_angle(lm):
    """Max deviation of either upper arm from vertical (degrees)."""
    down = np.array([0.0, 1.0, 0.0], np.float32)
    def va(v):
        n = np.linalg.norm(v)
        return float(np.degrees(np.arccos(np.clip(np.dot(v / n, down), -1, 1)))) \
            if n > 1e-6 else 0.0
    return max(va(lm[13] - lm[11]), va(lm[14] - lm[12]))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(model_path, labels_path, camera_id):

    # ── Labels ──────────────────────────────────────────────────────────────
    label_names = ["bent_elbow", "perfect"]
    try:
        with open(labels_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
            if lines:
                label_names = [
                    ln.split(",", 1)[1].strip() if "," in ln else ln
                    for ln in lines
                ]
    except FileNotFoundError:
        pass
    print(f"Labels: {label_names}")

    # ── Classifier TFLite ────────────────────────────────────────────────────
    try:
        interp = Interpreter(model_path=model_path, num_threads=4)
    except TypeError:
        interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    TARGET_FRAMES = 48
    for d in inp:
        if len(d["shape"]) == 3:
            TARGET_FRAMES = int(d["shape"][1])
            break
    print(f"TARGET_FRAMES={TARGET_FRAMES}  classes={int(out[0]['shape'][-1])}")

    # ── VAE encoder (optional OOD gate) ─────────────────────────────────────
    vae_interp   = None
    vae_inp      = None
    vae_out_det  = None
    kl_threshold = None
    vae_sc_mean  = None
    vae_sc_scale = None

    if VAE_ENABLED:
        _vae_paths = [
            model_path.replace("shrug_classifier_fp16.tflite", "shrug_vae_encoder.tflite"),
            os.path.join(os.path.dirname(model_path), "shrug_vae_encoder.tflite"),
        ]
        for _vc in _vae_paths:
            if os.path.exists(_vc):
                try:
                    vae_interp  = Interpreter(model_path=_vc, num_threads=2)
                    vae_interp.allocate_tensors()
                    vae_inp     = vae_interp.get_input_details()
                    vae_out_det = vae_interp.get_output_details()
                    print(f"VAE loaded: {_vc}")
                except Exception as e:
                    print(f"VAE load failed: {e}")
                    vae_interp = None
                break

        _stats_paths = [
            model_path.replace("shrug_classifier_fp16.tflite",
                               "best_shrug_model_feature_stats.npz"),
            os.path.join(os.path.dirname(model_path),
                         "best_shrug_model_feature_stats.npz"),
        ]
        for _sc in _stats_paths:
            if os.path.exists(_sc):
                try:
                    _st = np.load(_sc, allow_pickle=True)
                    if "vae_kl_threshold" in _st:
                        kl_threshold = float(_st["vae_kl_threshold"][0])
                        vae_sc_mean  = _st["vae_scaler_mean"].astype(np.float32)
                        vae_sc_scale = _st["vae_scaler_scale"].astype(np.float32)
                        print(f"KL threshold (from file) = {kl_threshold:.3f}")
                except Exception as e:
                    print(f"Stats load failed: {e}")
                break

        # Apply manual override
        if VAE_KL_THRESHOLD_OVERRIDE is not None:
            kl_threshold = VAE_KL_THRESHOLD_OVERRIDE
            print(f"[VAE] KL threshold OVERRIDDEN → {kl_threshold:.3f}")

    def compute_kl_vae(depth_feat):
        if vae_interp is None or vae_sc_mean is None:
            return None
        n = min(len(depth_feat), len(vae_sc_mean))
        x = ((depth_feat[:n] - vae_sc_mean[:n]) / (vae_sc_scale[:n] + 1e-6)) \
            .reshape(1, -1).astype(np.float32)
        vae_interp.set_tensor(vae_inp[0]["index"], x)
        vae_interp.invoke()
        mu_t = lv_t = None
        for d in vae_out_det:
            nm = d["name"].lower()
            if "mu" in nm or ("mean" in nm and "log" not in nm):
                mu_t = vae_interp.get_tensor(d["index"])[0]
            elif "log_var" in nm or "log_v" in nm:
                lv_t = vae_interp.get_tensor(d["index"])[0]
        if mu_t is None or lv_t is None:
            mu_t = vae_interp.get_tensor(vae_out_det[0]["index"])[0]
            lv_t = vae_interp.get_tensor(vae_out_det[1]["index"])[0]
        return float(0.5 * np.sum(np.square(mu_t) + np.exp(lv_t) - lv_t - 1.0))

    def run_model(seq_feat, depth_feat):
        if len(inp) == 1:
            interp.set_tensor(inp[0]["index"], seq_feat[np.newaxis].astype(np.float32))
        else:
            for d in inp:
                if len(d["shape"]) == 3:
                    interp.set_tensor(d["index"], seq_feat[np.newaxis].astype(np.float32))
                else:
                    n = int(d["shape"][-1])
                    interp.set_tensor(d["index"],
                                      depth_feat[:n][np.newaxis].astype(np.float32))
        interp.invoke()
        return interp.get_tensor(out[0]["index"])[0].astype(np.float32)

    # ── MediaPipe ────────────────────────────────────────────────────────────
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=MEDIAPIPE_DET_CONF,
        min_tracking_confidence=MEDIAPIPE_TRACK_CONF,
    )

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # ── State ────────────────────────────────────────────────────────────────
    # Calibration accumulators
    baseline_sho_y   = []   # mean shoulder Y per frame
    baseline_nose_y  = []   # nose Y per frame  ← stable face anchor
    baseline_elbow_y = []   # mean elbow Y per frame
    baseline_spreads = []   # elbow spread ratio per frame
    baseline_done    = False

    # Calibrated nose→landmark gaps  (all positive: both shoulders and elbows
    # are BELOW the nose, i.e. have larger Y in MediaPipe coordinates)
    #   line_y  =  smoothed_nose_y  +  nose_XXX_gap
    nose_sho_gap      = None   # sho_y_calibrated  − nose_y_calibrated
    nose_elbow_gap    = None   # elbow_y_calibrated − nose_y_calibrated
    rest_elbow_spread = None

    # Smoothed nose Y — updated every frame post-calibration
    smoothed_nose_y   = None

    # Stop-confirmation counter (shoulder must stay below line for N frames)
    stop_confirm_count = 0

    pre_trigger_buf = collections.deque(maxlen=PRE_TRIGGER_FRAMES)
    recording       = False
    frame_buffer    = []
    rep_count       = 0
    last_label      = "—"
    last_conf       = 0.0
    discard_msg     = ""
    discard_timer   = 0
    history         = collections.deque(maxlen=HISTORY_MAXLEN)

    print("Stand still ~2 seconds for calibration…")
    print("Q/ESC = quit   R = reset\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        res   = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        H, W  = frame.shape[:2]
        arm_angle = 0.0

        if res.pose_landmarks:
            lms       = res.pose_landmarks.landmark
            lm        = np.array([[l.x, l.y, l.z] for l in lms], dtype=np.float32)
            arm_angle = upper_arm_angle(lm)

            # ── LAYER 1: Skeleton ──────────────────────────────────────────
            mp_drawing.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(80,  80,  80),  thickness=1),
            )
            arm_col = (0, 0, 220) if arm_angle > ARM_MAX_ANGLE else (180, 180, 180)
            for si, ei in [(11, 13), (12, 14)]:
                cv2.line(frame,
                         (int(lm[si, 0] * W), int(lm[si, 1] * H)),
                         (int(lm[ei, 0] * W), int(lm[ei, 1] * H)),
                         arm_col, 3)

            # ── LAYER 2: Elbow-spread line (between actual elbow points) ───
            el_l_px        = (int(lm[13, 0] * W), int(lm[13, 1] * H))
            el_r_px        = (int(lm[14, 0] * W), int(lm[14, 1] * H))
            current_spread = get_elbow_spread_ratio(lm)

            if baseline_done:
                elbow_bent = current_spread > rest_elbow_spread * ELBOW_BENT_SPREAD_MULT
                el_color   = COLOR_ELBOW_BENT if elbow_bent else COLOR_ELBOW_OK
            else:
                el_color   = COLOR_ELBOW_CALIB

            cv2.line(frame, el_l_px, el_r_px, el_color, 3)
            mid_x = (el_l_px[0] + el_r_px[0]) // 2
            mid_y = (el_l_px[1] + el_r_px[1]) // 2 - 10
            cv2.putText(frame, f"{current_spread:.2f}",
                        (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, el_color, 1)

            # ── CALIBRATION PHASE ──────────────────────────────────────────
            if not baseline_done:
                baseline_sho_y.append(get_sho_y(lm))
                baseline_nose_y.append(get_nose_y(lm))
                baseline_elbow_y.append(get_elbow_y(lm))
                baseline_spreads.append(current_spread)

                pct = int(len(baseline_sho_y) / BASELINE_FRAMES * 100)
                cv2.putText(frame, f"Stand still... {pct}%",
                            (W // 2 - 130, H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)

                if len(baseline_sho_y) >= BASELINE_FRAMES:
                    cal_nose_y  = float(np.mean(baseline_nose_y))
                    cal_sho_y   = float(np.mean(baseline_sho_y))
                    cal_elbow_y = float(np.mean(baseline_elbow_y))

                    # Gaps are always positive: shoulder & elbow are below nose
                    # (larger Y in MediaPipe convention)
                    nose_sho_gap      = cal_sho_y   - cal_nose_y   # > 0
                    nose_elbow_gap    = cal_elbow_y - cal_nose_y   # > 0
                    rest_elbow_spread = float(np.mean(baseline_spreads))

                    # Initialise smoothed nose at the calibrated value
                    smoothed_nose_y   = cal_nose_y

                    baseline_done = True
                    print("Calibration complete:")
                    print(f"  nose_y        = {cal_nose_y:.4f}")
                    print(f"  shoulder_y    = {cal_sho_y:.4f}  "
                          f"(gap from nose = {nose_sho_gap:.4f})")
                    print(f"  elbow_y       = {cal_elbow_y:.4f}  "
                          f"(gap from nose = {nose_elbow_gap:.4f})")
                    print(f"  elbow spread  = {rest_elbow_spread:.3f}  "
                          f"(bent threshold = "
                          f"{rest_elbow_spread * ELBOW_BENT_SPREAD_MULT:.3f})")

            # ── POST-CALIBRATION ───────────────────────────────────────────
            else:
                # ── Update smoothed nose Y (EMA low-pass filter) ───────────
                # Small alpha = very stable but slow to follow intentional moves.
                # The line tracks the user when they step closer/farther but
                # ignores fast noise (breathing, micro head-bobs).
                smoothed_nose_y = (NOSE_SMOOTH_ALPHA * get_nose_y(lm)
                                   + (1.0 - NOSE_SMOOTH_ALPHA) * smoothed_nose_y)

                # ── Compute nose-anchored reference line positions ──────────
                # shoulder_line_y is WHERE the shoulder should be at rest.
                # When the shoulder shrugs UP its Y drops below this value.
                shoulder_line_y = smoothed_nose_y + nose_sho_gap
                elbow_line_y    = smoothed_nose_y + nose_elbow_gap

                # Current live shoulder Y
                sho_y_now = get_sho_y(lm)

                # ── LAYER 3: Two reference lines (nose-anchored) ───────────
                #
                # Formula guarantees:
                #   • If nose is still  → line is still
                #   • If shoulder shrugs → sho_y drops, but line stays fixed
                #     (it only tracks nose_y, not shoulder_y)
                #   • If user steps closer/farther → nose_y shifts →
                #     line shifts by exactly the same amount
                #
                # Line 1 — Shoulder rest level (teal-green)
                sho_line_px = max(0, min(H - 1, int(shoulder_line_y * H)))
                cv2.line(frame, (0, sho_line_px), (W, sho_line_px),
                         COLOR_SHOULDER_LINE, 2)
                cv2.putText(frame, "SHOULDER",
                            (8, sho_line_px - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_SHOULDER_LINE, 1)

                # Line 2 — Elbow rest level (yellow)
                elbow_line_px = max(0, min(H - 1, int(elbow_line_y * H)))
                cv2.line(frame, (0, elbow_line_px), (W, elbow_line_px),
                         COLOR_ELBOW_LINE, 2)
                cv2.putText(frame, "ELBOW",
                            (8, elbow_line_px - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ELBOW_LINE, 1)

                # ── State machine ──────────────────────────────────────────
                if not recording:
                    pre_trigger_buf.append(lm.copy())

                    # START: shoulder rose above the rest line
                    # (sho_y DECREASES as shoulder goes UP in MediaPipe coords)
                    if sho_y_now < shoulder_line_y:
                        recording          = True
                        stop_confirm_count = 0
                        frame_buffer       = list(pre_trigger_buf)
                        print(f"● REC start  sho={sho_y_now:.3f}  "
                              f"line={shoulder_line_y:.3f}  "
                              f"pre={len(frame_buffer)}f")

                else:
                    # Arm-guard: abort if arms drift laterally
                    if arm_angle > ARM_MAX_ANGLE:
                        recording          = False
                        stop_confirm_count = 0
                        discard_msg        = f"ARMS DRIFTED ({arm_angle:.1f}°)"
                        discard_timer      = 60
                        frame_buffer       = []
                        print(f"✗ {discard_msg}")

                    else:
                        frame_buffer.append(lm.copy())

                        # STOP: shoulder returned to or below the rest line
                        # Require STOP_CONFIRM_FRAMES consecutive frames to
                        # avoid stopping prematurely during brief oscillations.
                        if sho_y_now >= shoulder_line_y:
                            stop_confirm_count += 1

                            if stop_confirm_count >= STOP_CONFIRM_FRAMES:
                                recording          = False
                                stop_confirm_count = 0
                                n                  = len(frame_buffer)
                                print(f"■ REC stop  frames={n}")

                                if n >= MIN_FRAMES:
                                    seq, dep = process_rep(
                                        np.stack(frame_buffer), TARGET_FRAMES)

                                    # ── Step 1: VAE OOD gate ───────────────
                                    is_ood = False
                                    kl     = None

                                    if VAE_ENABLED:
                                        kl = compute_kl_vae(dep)
                                        if kl is not None and kl_threshold is not None:
                                            if kl > kl_threshold:
                                                is_ood = True
                                                print(f"[VAE-OOD] KL={kl:.2f} > "
                                                      f"{kl_threshold:.2f} → unknown")
                                            else:
                                                print(f"[VAE-OK]  KL={kl:.2f}")

                                    if is_ood:
                                        last_label = "unknown"
                                        last_conf  = 0.0
                                        rep_count += 1
                                        history.appendleft(("unknown", 0.0))
                                        print(f"→ Rep #{rep_count}: unknown (OOD)")

                                    else:
                                        # ── Step 2: Classifier ─────────────
                                        probs      = run_model(seq, dep)
                                        i_max      = int(np.argmax(probs))
                                        conf       = float(probs[i_max])
                                        last_label = (label_names[i_max]
                                                      if i_max < len(label_names)
                                                      else str(i_max))
                                        last_conf  = conf
                                        rep_count += 1
                                        history.appendleft((last_label, conf))
                                        kl_str = f"  KL={kl:.2f}" if kl is not None else ""
                                        print(f"→ Rep #{rep_count}: {last_label} "
                                              f"({conf * 100:.1f}%){kl_str}")

                                frame_buffer = []

                        else:
                            # Shoulder still above line — keep recording, reset counter
                            stop_confirm_count = 0

                # ── Debug text (bottom of frame) ───────────────────────────
                kl_hint = f"  KL<{kl_threshold:.2f}" if kl_threshold else ""
                cv2.putText(
                    frame,
                    f"sho={sho_y_now:.3f}  line={shoulder_line_y:.3f}  "
                    f"nose={smoothed_nose_y:.3f}  gap={nose_sho_gap:.3f}"
                    f"{kl_hint}",
                    (20, H - 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 110), 1)

        # ── HUD ────────────────────────────────────────────────────────────
        if recording:
            cv2.circle(frame, (W - 25, 25), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {len(frame_buffer)}f",
                        (W - 100, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        ang_col = (0, 0, 220) if arm_angle > ARM_MAX_ANGLE else (50, 200, 80)
        cv2.putText(frame, f"ARM {arm_angle:.1f}°",
                    (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ang_col, 2)

        if discard_timer > 0:
            cv2.putText(frame, discard_msg,
                        (W // 2 - 180, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 80, 255), 2)
            discard_timer -= 1

        cv2.putText(frame, f"REPS: {rep_count}",
                    (20, H - 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        col = COLORS.get(last_label.lower(), (180, 180, 180))
        cv2.putText(frame, f"{last_label.upper()}  {last_conf * 100:.0f}%",
                    (20, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)

        for i, (hl, hc) in enumerate(history):
            cv2.putText(frame, f"{hl}  {hc * 100:.0f}%",
                        (W - 210, 55 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        COLORS.get(hl.lower(), (180, 180, 180)), 1)

        cv2.imshow("Shrug Classifier", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if key in (ord("r"), ord("R")):
            baseline_sho_y    = []
            baseline_nose_y   = []
            baseline_elbow_y  = []
            baseline_spreads  = []
            baseline_done     = False
            nose_sho_gap      = None
            nose_elbow_gap    = None
            rest_elbow_spread = None
            smoothed_nose_y   = None
            stop_confirm_count = 0
            pre_trigger_buf.clear()
            recording    = False;  frame_buffer = []
            rep_count    = 0;      history.clear()
            last_label   = "—";    last_conf = 0.0;  discard_timer = 0
            print("Reset.")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=MODEL_PATH)
    parser.add_argument("--labels", default=LABELS_PATH)
    parser.add_argument("--camera", type=int, default=CAMERA_ID)
    args = parser.parse_args()
    main(args.model, args.labels, args.camera)