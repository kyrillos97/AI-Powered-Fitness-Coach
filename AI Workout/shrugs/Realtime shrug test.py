# realtime_shrug_test.py
#
# Real-time shrug classifier using MediaPipe + TFLite
# Classes: perfect, bent_elbow
# Plus rule-based gates: too_fast, unknown
#
# HOW IT WORKS:
#   1. Reads webcam, runs MediaPipe Pose every frame
#   2. Computes shoulder elevation each frame (normalised, up = positive)
#   3. When elevation RISES above RECORD_START_ELEV → start buffering
#   4. When elevation FALLS back below RECORD_STOP_ELEV → stop buffer
#      → Gate A (fast rep): duration < FAST_REP_MAX_SECONDS
#                        OR peak_sho_vel > FAST_REP_VELOCITY_THRESHOLD → "too_fast"
#      → Gate B (sanity):   sho_elev_rom < MIN_SHO_ROM
#                        OR duration < MIN_REP_SECONDS → rejected silently
#      → Model:             perfect vs bent_elbow
#      → Gate C (OOD):      max_prob < CONF_THRESHOLD → "unknown"
#
# VISUALIZATIONS:
#   • Live pose skeleton with shoulder/elbow/wrist highlights
#   • Shoulder elevation (large, colour-coded)
#   • Left + right elbow angles
#   • Rolling graph: shoulder elevation (green) + avg elbow angle (orange)
#   • Recording state badge
#   • Rep counter, last prediction + confidence bar
#   • Per-class vertical confidence bars
#   • Rep history (last 5)
#   • Debug overlay: raw probs, live features, gate status
#
# USAGE:
#   python realtime_shrug_test.py
#   python realtime_shrug_test.py --model shrug_classifier_fp16.tflite --camera 1

import argparse
import collections
import time
import numpy as np
import cv2

import mediapipe as mp
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_MODEL  = r"D:\AI-Powered-Fitness-Coach\AI Workout\shrugs\shrug_classifier_fp16.tflite"
DEFAULT_LABELS = r"D:\AI-Powered-Fitness-Coach\AI Workout\shrugs\shrug_label_classes.txt"
DEFAULT_CAMERA = 1

# TARGET_FRAMES must match what was used during training.
# Set to None to read from the model input shape automatically.
TARGET_FRAMES  = None        # auto-read from model
SMOOTH_ALPHA   = 0.5

# ── Dynamic recording trigger ────────────────────────────────────────────
# No fixed thresholds. A slow EMA tracks the resting shoulder elevation.
# Recording starts when shoulders rise above baseline by ELEV_RISE_DELTA.
# Recording stops when elevation drops back from its peak by ELEV_DROP_DELTA.
# Works for any body size and camera distance automatically.

ELEV_RISE_DELTA  = 0.03   # how much above resting baseline → start recording
                           # raise to 0.05 if breathing falsely triggers
ELEV_DROP_DELTA  = 0.02   # how much below peak → stop recording
                           # lower to 0.015 if reps are cut short
BASELINE_ALPHA   = 0.02   # EMA speed for baseline update (only when idle)
                           # smaller = slower baseline, more stable

MIN_FRAMES_FOR_REP = 8    # fewer frames than this → silently ignore

# ── OOD gate — confidence only ────────────────────────────────────────────
# The model only knows 2 classes. If it's not confident, say "unknown".
CONF_THRESHOLD   = 0.65   # below this → "unknown"

# ── Joints (MediaPipe indices) ────────────────────────────────────────────
SHRUG_JOINTS = {
    "nose":       0,
    "l_ear":      7,  "r_ear":  8,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow":    13, "r_elbow":    14,
    "l_wrist":    15, "r_wrist":    16,
    "l_hip":      23, "r_hip":      24,
}
JOINT_IDX = list(SHRUG_JOINTS.values())

# ── Colours (BGR) ─────────────────────────────────────────────────────────
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,   0,   0)
C_GREEN  = (80,  200, 80)
C_YELLOW = (50,  210, 230)
C_RED    = (80,  80,  220)
C_BLUE   = (220, 150, 50)
C_GREY   = (120, 120, 120)
C_DARK   = (30,  30,  30)
C_ORANGE = (50,  160, 240)
C_PURPLE = (200, 80,  180)

LABEL_COLORS = {
    "perfect":    C_GREEN,
    "bent_elbow": C_RED,
    "too_fast":   C_PURPLE,
    "unknown":    C_GREY,
}

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE PIPELINE  — must exactly match train_shrug_model.py
# ═══════════════════════════════════════════════════════════════════════════

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

def torso_normalize(lm):
    mid_hip = (lm[23] + lm[24]) / 2.0
    mid_sho = (lm[11] + lm[12]) / 2.0
    scale   = np.linalg.norm(mid_sho - mid_hip)
    return (lm - mid_hip) / max(scale, 1e-6)

def resample(seq, target):
    T = len(seq)
    if T == target:
        return seq.astype(np.float32)
    idx = np.linspace(0, T - 1, target)
    out = np.zeros((target, seq.shape[1]), dtype=np.float32)
    for f in range(seq.shape[1]):
        out[:, f] = np.interp(idx, np.arange(T), seq[:, f])
    return out

def process_shrug_rep(frames, target_frames):
    """
    frames: (T, 33, 3) raw MediaPipe landmarks.
    Returns (seq_feat, depth_feat) matching training pipeline exactly.
    depth_feat indices:
      [0]  min_el_l      [1]  min_el_r
      [2]  rom_el_l      [3]  rom_el_r
      [4]  max_asym      [5]  sho_max
      [6]  sho_rom       [7]  sho_ear_min
      [8]  wrist_drift   [9]  peak_vel   ← fast rep detector
      [10] mean_acc      [11] ev_max_l
      [12] ev_max_r      [13] sho_sym
      [14] head_drift    [15] wrist_hip
    """
    T    = len(frames)
    fsm  = ema_smooth(frames, SMOOTH_ALPHA)
    norm = np.array([torso_normalize(f) for f in fsm], dtype=np.float32)

    # Elbow angles: shoulder → elbow → wrist
    elbow_l = angle_batch(norm[:, 11], norm[:, 13], norm[:, 15])   # (T,1)
    elbow_r = angle_batch(norm[:, 12], norm[:, 14], norm[:, 16])   # (T,1)

    # Shoulder elevation (up = positive)
    sho_y_l  = -norm[:, 11, 1]
    sho_y_r  = -norm[:, 12, 1]
    sho_elev = ((sho_y_l + sho_y_r) / 2.0)[:, None]

    # Velocity and acceleration
    sho_vel = np.concatenate([
        np.zeros((1, 1), dtype=np.float32),
        np.diff(sho_elev, axis=0)
    ], axis=0)
    sho_acc = np.concatenate([
        np.zeros((1, 1), dtype=np.float32),
        np.diff(sho_vel, axis=0)
    ], axis=0)

    # Scalar features
    min_el_l = float(np.min(elbow_l));   max_el_l = float(np.max(elbow_l))
    min_el_r = float(np.min(elbow_r));   max_el_r = float(np.max(elbow_r))
    rom_el_l = max_el_l - min_el_l
    rom_el_r = max_el_r - min_el_r
    max_asym = float(np.max(np.abs(elbow_l - elbow_r)))

    ev_l     = np.abs(np.diff(elbow_l[:, 0]))
    ev_r     = np.abs(np.diff(elbow_r[:, 0]))
    ev_max_l = float(np.max(ev_l)) if len(ev_l) > 0 else 0.0
    ev_max_r = float(np.max(ev_r)) if len(ev_r) > 0 else 0.0

    sho_max  = float(np.max(sho_elev))
    sho_rom  = float(np.max(sho_elev) - np.min(sho_elev))

    ear_y_l  = -norm[:, 7,  1];   ear_y_r = -norm[:, 8,  1]
    sho_ear_min = float(np.min(
        (np.abs(sho_y_l - ear_y_l) + np.abs(sho_y_r - ear_y_r)) / 2.0
    ))

    wrist_l_y = -norm[:, 15, 1];  wrist_r_y = -norm[:, 16, 1]
    wrist_drift = float(np.max(
        np.maximum(wrist_l_y - wrist_l_y[0], wrist_r_y - wrist_r_y[0])
    ))
    wrist_hip = float(np.mean(
        (np.abs(norm[:, 15, 1]) + np.abs(norm[:, 16, 1])) / 2.0
    ))

    peak_vel  = float(np.max(np.abs(sho_vel)))
    mean_acc  = float(np.mean(np.abs(sho_acc)))
    sho_sym   = float(np.std(sho_y_l - sho_y_r))

    nose_y     = -norm[:, 0, 1]
    head_drift = float(np.max(nose_y) - np.min(nose_y))

    depth_feat = np.array([
        min_el_l, min_el_r,
        rom_el_l, rom_el_r,
        max_asym,
        sho_max, sho_rom,
        sho_ear_min,
        wrist_drift,
        peak_vel,          # index 9 — fast rep detector
        mean_acc,
        ev_max_l, ev_max_r,
        sho_sym,
        head_drift,
        wrist_hip,
    ], dtype=np.float32)

    pos      = norm[:, JOINT_IDX, :].reshape(T, -1)
    vel      = np.concatenate([np.zeros_like(pos[:1]), np.diff(pos, axis=0)], axis=0)
    acc      = np.concatenate([np.zeros_like(vel[:1]), np.diff(vel, axis=0)], axis=0)
    seq_feat = np.concatenate(
        [pos, vel, acc, elbow_l, elbow_r, sho_elev, sho_vel, sho_acc],
        axis=1
    ).astype(np.float32)

    return resample(seq_feat, target_frames), depth_feat


def single_frame_shoulder_elevation(lm33):
    """
    Current shoulder elevation in torso-normalised units.
    Higher value = shoulders raised more. Resting ~0.85–1.0.
    """
    mid_hip = (lm33[23] + lm33[24]) / 2.0
    mid_sho = (lm33[11] + lm33[12]) / 2.0
    scale   = np.linalg.norm(mid_sho - mid_hip)
    if scale < 1e-6:
        return 0.0
    norm_sho_l = -(lm33[11, 1] - mid_hip[1]) / scale
    norm_sho_r = -(lm33[12, 1] - mid_hip[1]) / scale
    return float((norm_sho_l + norm_sho_r) / 2.0)

def single_frame_elbow_angles(lm33):
    """Returns (left_elbow_deg, right_elbow_deg)."""
    el_l = angle_batch(
        lm33[11:12], lm33[13:14], lm33[15:16]
    )[0, 0]
    el_r = angle_batch(
        lm33[12:13], lm33[14:15], lm33[16:17]
    )[0, 0]
    return float(el_l), float(el_r)


# ═══════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def draw_rounded_rect(img, x, y, w, h, r, color, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, -1)
    for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def put_text(img, text, pos, scale=0.6, color=C_WHITE, thickness=1,
             font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(img, text, pos, font, scale, C_BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color,   thickness,     cv2.LINE_AA)

def draw_bar(img, x, y, w, h, fraction, fg_color, bg_color=C_DARK, value_str=""):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    fill = int(np.clip(fraction, 0.0, 1.0) * w)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), fg_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), C_GREY, 1)
    if value_str:
        put_text(img, value_str, (x + w - 60, y + h - 5), scale=0.42, color=C_WHITE)

def elev_color(elev, lo=0.85, hi=1.25):
    """Blue at rest, green when elevated."""
    t = float(np.clip((elev - lo) / (hi - lo), 0.0, 1.0))
    return (int(220 * (1 - t)), int(200 * t), int(50 * (1 - t) + 150 * t))

def elbow_color(angle):
    """Green = straight (180°), red = bent (<140°)."""
    t = np.clip((angle - 100.0) / (180.0 - 100.0), 0.0, 1.0)
    return (50, int(200 * t), int(220 * (1 - t)))

def draw_shrug_graph(canvas, history_elev, history_elbow, x, y, w, h):
    """
    Rolling graph: shoulder elevation (green) + avg elbow angle (orange).
    Two separate y-axes overlaid — elevation on left scale, elbow on right.
    """
    draw_rounded_rect(canvas, x, y, w, h, 6, (25, 25, 25), alpha=0.75)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), C_GREY, 1)

    # Elevation grid (0.7 to 1.4)
    for ev in [0.8, 1.0, 1.1, 1.2]:
        gy = y + h - int((ev - 0.7) / (1.45 - 0.7) * h)
        cv2.line(canvas, (x, gy), (x + w, gy), (50, 50, 50), 1)
        put_text(canvas, f"{ev:.1f}", (x + 3, gy - 2), scale=0.30, color=(80, 80, 80))

    # Record threshold line
    # Dynamic baseline marker drawn in main loop, not here

    put_text(canvas, "Elev (green)  Elbow (orange)",
             (x + 8, y + 14), scale=0.38, color=C_GREY)

    def plot(data, color, lo, hi):
        if len(data) < 2:
            return
        pts = []
        for i, v in enumerate(data):
            px = x + int(i / (len(data) - 1) * (w - 1))
            py = y + h - int(np.clip((v - lo) / (hi - lo), 0, 1) * h)
            pts.append((px, py))
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)
        cv2.circle(canvas, pts[-1], 4, color, -1, cv2.LINE_AA)

    plot(list(history_elev),  C_GREEN,  0.70, 1.45)
    plot(list(history_elbow), C_ORANGE, 100,  185)

def draw_confidence_bars(canvas, probs, labels, x, y, w):
    n     = len(labels)
    bar_w = max(1, (w - (n + 1) * 6) // n)
    bar_h = 90
    for i, (label, prob) in enumerate(zip(labels, probs)):
        bx    = x + 6 + i * (bar_w + 6)
        color = LABEL_COLORS.get(label.lower(), C_BLUE)
        cv2.rectangle(canvas, (bx, y), (bx + bar_w, y + bar_h), C_DARK, -1)
        fill  = int(prob * bar_h)
        if fill > 0:
            cv2.rectangle(canvas, (bx, y + bar_h - fill),
                          (bx + bar_w, y + bar_h), color, -1)
        cv2.rectangle(canvas, (bx, y), (bx + bar_w, y + bar_h), C_GREY, 1)
        put_text(canvas, f"{prob*100:.0f}%",
                 (bx + 3, y + bar_h - 5), scale=0.42, color=C_WHITE)
        put_text(canvas, label[:6],
                 (bx + 3, y + bar_h + 14), scale=0.38,
                 color=color)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(model_path, labels_path, camera_id):

    # ── Labels ────────────────────────────────────────────────────────────
    label_names = []
    try:
        with open(labels_path) as f:
            for line in f:
                line = line.strip()
                if "," in line:
                    label_names.append(line.split(",", 1)[1].strip())
                elif line:
                    label_names.append(line)
    except FileNotFoundError:
        print(f"WARNING: '{labels_path}' not found.")
        print("  Fallback: bent_elbow=0, perfect=1  (alphabetical LabelEncoder order)")
        label_names = ["bent_elbow", "perfect"]

    # ── TFLite model ──────────────────────────────────────────────────────
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()

    print("Model inputs:")
    for d in inp_details:
        print(f"  [{d['index']}] {d['name']}  shape={d['shape']}")

    NUM_CLASSES = int(out_details[0]["shape"][-1])
    print(f"Model outputs: {NUM_CLASSES} classes")

    # Auto-detect TARGET_FRAMES from model input shape
    global TARGET_FRAMES
    for d in inp_details:
        if len(d["shape"]) == 3:              # (1, frames, features)
            model_frames = int(d["shape"][1])
            if TARGET_FRAMES is None:
                TARGET_FRAMES = model_frames
                print(f"Auto-detected TARGET_FRAMES = {TARGET_FRAMES} from model")
            elif TARGET_FRAMES != model_frames:
                print(f"WARNING: TARGET_FRAMES={TARGET_FRAMES} but model expects "
                      f"{model_frames} — using model value")
                TARGET_FRAMES = model_frames
            break
    if TARGET_FRAMES is None:
        TARGET_FRAMES = 48
        print(f"Could not detect TARGET_FRAMES from model — defaulting to {TARGET_FRAMES}")

    # Reconcile labels with model output count
    import os as _os
    if not _os.path.exists(labels_path) and len(label_names) == NUM_CLASSES:
        with open(labels_path, "w") as _f:
            for _i, _n in enumerate(label_names):
                _f.write(f"{_i},{_n}\n")
        print(f"Auto-saved '{labels_path}'")
    if len(label_names) != NUM_CLASSES:
        print(f"WARNING: label count {len(label_names)} != model outputs {NUM_CLASSES}")
        label_names = (label_names + [f"class_{i}" for i in range(NUM_CLASSES)])[:NUM_CLASSES]
    print(f"Labels: {label_names}")

    num_inputs = len(inp_details)
    print(f"Architecture: {'single' if num_inputs == 1 else 'dual'}-branch")

    def run_model(seq_feat, depth_feat):
        if num_inputs == 1:
            interp.set_tensor(inp_details[0]["index"],
                              seq_feat[np.newaxis].astype(np.float32))
        else:
            for d in inp_details:
                if len(d["shape"]) == 3:
                    interp.set_tensor(d["index"],
                                      seq_feat[np.newaxis].astype(np.float32))
                else:
                    expected = int(d["shape"][-1])
                    interp.set_tensor(d["index"],
                                      depth_feat[:expected][np.newaxis].astype(np.float32))
        interp.invoke()
        return interp.get_tensor(out_details[0]["index"])[0].astype(np.float32)

    # ── MediaPipe ─────────────────────────────────────────────────────────
    pose = mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1
    )

    # ── Webcam ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {W}x{H}")

    # ── State ─────────────────────────────────────────────────────────────
    recording       = False
    frame_buffer    = []
    rec_start_time  = 0.0
    elev_baseline   = None   # slow EMA of resting elevation, set on first frame
    elev_peak       = 0.0    # highest elevation seen during current rep
    rep_count       = 0
    last_probs      = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
    last_label      = "—"
    last_conf       = 0.0
    last_rep_info   = ""      # printed in history: elbow angle + sho_rom

    rep_history     = collections.deque(maxlen=5)   # (label, conf, info_str)
    graph_elev      = collections.deque(maxlen=150)
    graph_elbow     = collections.deque(maxlen=150)

    fps_timer       = time.time()
    fps_val         = 0.0
    frame_count_fps = 0
    pulse           = 0

    PANEL_W = 340
    GRAPH_H = 140

    print("\nPress  Q / ESC = quit    R = reset counter\n")
    print(f"Trigger: dynamic — shoulders rise > {ELEV_RISE_DELTA} above resting baseline")
    print(f"Stop:    shoulders drop > {ELEV_DROP_DELTA} below peak")
    print(f"Confidence gate: {CONF_THRESHOLD}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = pose.process(rgb)

        # FPS
        frame_count_fps += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_val = frame_count_fps / (now - fps_timer)
            fps_timer = now; frame_count_fps = 0

        current_elev   = 0.0
        current_elbow_l = 180.0
        current_elbow_r = 180.0
        lm_array       = None

        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            lm_array = np.array(
                [[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32
            )

            current_elev    = single_frame_shoulder_elevation(lm_array)
            current_elbow_l, current_elbow_r = single_frame_elbow_angles(lm_array)
            avg_elbow       = (current_elbow_l + current_elbow_r) / 2.0

            graph_elev.append(current_elev)
            graph_elbow.append(avg_elbow)

            # ── Dynamic baseline (only updates when idle) ─────────────────
            if elev_baseline is None:
                elev_baseline = current_elev          # initialise on first frame
            if not recording:
                # Slow EMA tracks the true resting elevation
                elev_baseline = (BASELINE_ALPHA * current_elev
                                 + (1.0 - BASELINE_ALPHA) * elev_baseline)

            # ── Rep detection state machine ───────────────────────────────
            if not recording:
                if current_elev > elev_baseline + ELEV_RISE_DELTA:
                    recording      = True
                    elev_peak      = current_elev
                    frame_buffer   = [lm_array.copy()]
                    rec_start_time = time.time()
            else:
                frame_buffer.append(lm_array.copy())
                elev_peak = max(elev_peak, current_elev)  # track highest point

                # Stop when shoulders fall back from peak
                if current_elev < elev_peak - ELEV_DROP_DELTA:
                    recording    = False
                    rep_duration = time.time() - rec_start_time

                    if len(frame_buffer) >= MIN_FRAMES_FOR_REP:
                        frames = np.stack(frame_buffer, axis=0)   # (T,33,3)
                        seq_feat, depth_feat = process_shrug_rep(frames, TARGET_FRAMES)

                        min_el_l     = float(depth_feat[0])
                        min_el_r     = float(depth_feat[1])
                        sho_rom_val  = float(depth_feat[6])

                        # ── Run model ─────────────────────────────────────
                        probs    = run_model(seq_feat, depth_feat)
                        pred_idx = int(np.argmax(probs))
                        top_conf = float(probs[pred_idx])
                        last_probs    = probs
                        last_rep_info = (f"el_l={min_el_l:.0f}°  "
                                         f"el_r={min_el_r:.0f}°  "
                                         f"rom={sho_rom_val:.3f}")

                        # ── Confidence gate ────────────────────────────────
                        if top_conf < CONF_THRESHOLD:
                            last_label = "unknown"
                            last_conf  = top_conf
                            print(f"[LOW CONF] {top_conf*100:.1f}%  "
                                  f"raw={[f'{p*100:.0f}%' for p in probs]}")
                        else:
                            last_label = label_names[pred_idx]
                            last_conf  = top_conf
                            rep_count += 1
                            rep_history.appendleft(
                                (last_label, last_conf, last_rep_info)
                            )
                            print(f"Rep #{rep_count}: {last_label} "
                                  f"({last_conf*100:.1f}%)  "
                                  f"dur={rep_duration:.2f}s  "
                                  f"el_l={min_el_l:.0f}° el_r={min_el_r:.0f}°  "
                                  f"raw={[f'{p*100:.0f}%' for p in probs]}")

                    frame_buffer = []

            # ── Draw skeleton ─────────────────────────────────────────────
            mp_drawing.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(180, 180, 180), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(140, 140, 140), thickness=2)
            )

            # Highlight shrug-critical joints
            h_img, w_img = frame.shape[:2]
            def lm_px(idx):
                return (int(lms[idx].x * w_img), int(lms[idx].y * h_img))

            # Shoulders — large circles, colour = elevation
            ec = elev_color(current_elev)
            for idx in [11, 12]:
                cv2.circle(frame, lm_px(idx), 9, ec, -1, cv2.LINE_AA)

            # Elbows
            el_c_l = elbow_color(current_elbow_l)
            el_c_r = elbow_color(current_elbow_r)
            cv2.circle(frame, lm_px(13), 7, el_c_l, -1, cv2.LINE_AA)
            cv2.circle(frame, lm_px(14), 7, el_c_r, -1, cv2.LINE_AA)

            # Wrists
            cv2.circle(frame, lm_px(15), 5, C_BLUE, -1, cv2.LINE_AA)
            cv2.circle(frame, lm_px(16), 5, C_BLUE, -1, cv2.LINE_AA)

            # Elbow angle labels
            put_text(frame, f"L:{current_elbow_l:.0f}",
                     (lm_px(13)[0] - 20, lm_px(13)[1] - 12),
                     scale=0.5, color=el_c_l, thickness=1)
            put_text(frame, f"R:{current_elbow_r:.0f}",
                     (lm_px(14)[0] - 20, lm_px(14)[1] - 12),
                     scale=0.5, color=el_c_r, thickness=1)

            # Shoulder-to-ear connecting lines (visual shrug depth indicator)
            cv2.line(frame, lm_px(7),  lm_px(11), (100, 200, 100), 2)
            cv2.line(frame, lm_px(8),  lm_px(12), (100, 200, 100), 2)

        # ═══════════════════════════════════════════════════════════════════
        # RIGHT PANEL
        # ═══════════════════════════════════════════════════════════════════
        panel_x = W - PANEL_W
        draw_rounded_rect(frame, panel_x, 0, PANEL_W, H, 0, (20, 20, 20), alpha=0.60)

        cy = 18

        # FPS
        put_text(frame, f"FPS {fps_val:.1f}", (panel_x + 8, cy),
                 scale=0.42, color=C_GREY)
        cy += 22

        # Recording state
        pulse = (pulse + 1) % 40
        if recording:
            rc = C_RED if pulse < 20 else C_ORANGE
            draw_rounded_rect(frame, panel_x + 8, cy, PANEL_W - 16, 28, 5, rc, 0.7)
            put_text(frame, f"● RECORDING  ({len(frame_buffer)} frames)",
                     (panel_x + 18, cy + 19), scale=0.52, color=C_WHITE)
        else:
            draw_rounded_rect(frame, panel_x + 8, cy, PANEL_W - 16, 28, 5,
                              (40, 40, 40), 0.7)
            put_text(frame, "◉ IDLE — waiting for shrug",
                     (panel_x + 18, cy + 19), scale=0.52, color=C_GREY)
        cy += 36

        # Shoulder elevation (primary metric)
        ec = elev_color(current_elev)
        put_text(frame, "SHOULDER ELEVATION", (panel_x + 10, cy),
                 scale=0.42, color=C_GREY)
        cy += 4
        put_text(frame, f"{current_elev:.3f}",
                 (panel_x + 10, cy + 34), scale=1.2, color=ec, thickness=2)
        # Mini threshold indicator
        bar_y = cy + 38
        cv2.rectangle(frame, (panel_x + 10, bar_y),
                      (panel_x + PANEL_W - 20, bar_y + 6), C_DARK, -1)
        frac = np.clip((current_elev - 0.7) / (1.45 - 0.7), 0, 1)
        fill_x = panel_x + 10 + int(frac * (PANEL_W - 30))
        cv2.rectangle(frame, (panel_x + 10, bar_y), (fill_x, bar_y + 6), ec, -1)
        # Threshold marker
        # White tick = current baseline + rise delta (dynamic trigger point)
        if elev_baseline is not None:
            trigger_x = panel_x + 10 + int(
                np.clip((elev_baseline + ELEV_RISE_DELTA - 0.7) / (1.45 - 0.7), 0, 1)
                * (PANEL_W - 30)
            )
            cv2.line(frame, (trigger_x, bar_y - 2), (trigger_x, bar_y + 8), C_WHITE, 2)
        cy += 52

        # Elbow angles (side by side)
        put_text(frame, "ELBOW ANGLES", (panel_x + 10, cy),
                 scale=0.42, color=C_GREY)
        cy += 4
        el_c_l = elbow_color(current_elbow_l)
        el_c_r = elbow_color(current_elbow_r)
        put_text(frame, f"L: {current_elbow_l:.0f}°",
                 (panel_x + 10, cy + 22), scale=0.75, color=el_c_l, thickness=2)
        put_text(frame, f"R: {current_elbow_r:.0f}°",
                 (panel_x + 150, cy + 22), scale=0.75, color=el_c_r, thickness=2)
        cy += 32

        # Graph
        draw_shrug_graph(frame, graph_elev, graph_elbow,
                         panel_x + 4, cy, PANEL_W - 8, GRAPH_H)
        cy += GRAPH_H + 8

        # Rep counter
        draw_rounded_rect(frame, panel_x + 8, cy, PANEL_W - 16, 38, 6,
                          (40, 40, 80), 0.7)
        put_text(frame, "REPS", (panel_x + 16, cy + 14),
                 scale=0.45, color=C_GREY)
        put_text(frame, str(rep_count), (panel_x + 70, cy + 30),
                 scale=1.1, color=C_BLUE, thickness=2)
        cy += 46

        # Last prediction
        pred_color = LABEL_COLORS.get(last_label.lower(), C_BLUE)
        draw_rounded_rect(frame, panel_x + 8, cy, PANEL_W - 16, 44, 6,
                          (30, 30, 30), 0.7)
        put_text(frame, "LAST REP", (panel_x + 16, cy + 14),
                 scale=0.42, color=C_GREY)
        put_text(frame, last_label.upper(), (panel_x + 16, cy + 34),
                 scale=0.75, color=pred_color, thickness=2)
        draw_bar(frame, panel_x + 150, cy + 20, PANEL_W - 164, 18,
                 last_conf, pred_color, value_str=f"{last_conf*100:.0f}%")
        cy += 52

        # Per-class confidence bars (only model classes, not rule-based labels)
        put_text(frame, "CONFIDENCE", (panel_x + 10, cy),
                 scale=0.42, color=C_GREY)
        cy += 4
        draw_confidence_bars(frame, last_probs, label_names,
                             panel_x + 4, cy, PANEL_W - 8)
        cy += 112

        # Rep history
        put_text(frame, "HISTORY", (panel_x + 10, cy),
                 scale=0.42, color=C_GREY)
        cy += 4
        for i, (rl, rc, ri) in enumerate(rep_history):
            rc_col  = LABEL_COLORS.get(rl.lower(), C_BLUE)
            rep_num = rep_count - i if rl not in ("too_fast", "unknown") else "—"
            put_text(frame,
                     f"#{rep_num}  {rl:<11}  {rc*100:.0f}%",
                     (panel_x + 10, cy + 14 + i * 18),
                     scale=0.38, color=rc_col)
        cy += len(rep_history) * 18 + 6

        # ── No pose warning ───────────────────────────────────────────────
        if not res.pose_landmarks:
            put_text(frame, "NO POSE DETECTED", (W // 2 - 130, H // 2),
                     scale=1.0, color=C_RED, thickness=2)

        # ── Debug overlay (bottom-left) ───────────────────────────────────
        dbg_x, dbg_y = 10, H - 130
        draw_rounded_rect(frame, dbg_x, dbg_y, 350, 120, 5, (20, 20, 20), 0.7)
        put_text(frame, "DEBUG", (dbg_x+8, dbg_y+14), scale=0.38, color=C_GREY)
        put_text(frame,
                 f"elev={current_elev:.3f}  el_l={current_elbow_l:.0f}  el_r={current_elbow_r:.0f}",
                 (dbg_x+8, dbg_y+30), scale=0.40, color=C_WHITE)
        put_text(frame,
                 f"buf={len(frame_buffer)}  recording={recording}",
                 (dbg_x+8, dbg_y+46), scale=0.40, color=C_WHITE)
        prob_str = "  ".join(
            [f"{label_names[i][:5]}={last_probs[i]*100:.0f}%"
             for i in range(NUM_CLASSES)]
        )
        put_text(frame, prob_str, (dbg_x+8, dbg_y+62), scale=0.38, color=C_YELLOW)
        _bl = f"{elev_baseline:.3f}" if elev_baseline else "init"
        put_text(frame,
                 f"baseline={_bl}  rise>{ELEV_RISE_DELTA}  drop>{ELEV_DROP_DELTA}  conf>{CONF_THRESHOLD}",
                 (dbg_x+8, dbg_y+78), scale=0.35, color=C_GREY)
        put_text(frame,
                 f"TARGET_FRAMES={TARGET_FRAMES}  inputs={num_inputs}  classes={NUM_CLASSES}",
                 (dbg_x+8, dbg_y+94), scale=0.35, color=C_GREY)
        put_text(frame, f"last_info: {last_rep_info}",
                 (dbg_x+8, dbg_y+110), scale=0.35, color=C_GREY)

        cv2.imshow("Shrug Classifier — Real-time", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("r"), ord("R")):
            rep_count = 0
            rep_history.clear()
            last_probs    = np.ones(NUM_CLASSES) / NUM_CLASSES
            last_label    = "—"; last_conf = 0.0; last_rep_info = ""
            elev_baseline = None   # re-learn baseline after reset
            print("Rep counter reset. Baseline will re-learn.")

    cap.release()
    pose.close()
    cv2.destroyAllWindows()
    print(f"\nSession ended.  Total reps: {rep_count}")
    if rep_history:
        print("Rep history:")
        for i, (rl, rc, ri) in enumerate(reversed(rep_history)):
            print(f"  {i+1}: {rl}  conf={rc*100:.0f}%  {ri}")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time shrug classifier")
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--camera", default=DEFAULT_CAMERA, type=int)
    args = parser.parse_args()

    print(f"Model:  {args.model}")
    print(f"Labels: {args.labels}")
    print(f"Camera: {args.camera}")
    print("Controls:  Q / ESC = quit    R = reset counter\n")

    main(args.model, args.labels, args.camera)