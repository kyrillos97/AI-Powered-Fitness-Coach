# realtime_squat_test.py
#
# Real-time squat classifier using MediaPipe + TFLite
#
# HOW IT WORKS:
#   1. Reads webcam, runs MediaPipe Pose every frame
#   2. Computes average knee angle each frame
#   3. When knee angle drops BELOW RECORD_START_ANGLE (170°) → start buffering frames
#   4. When knee angle returns ABOVE RECORD_START_ANGLE → stop buffer, run full
#      feature pipeline, feed to TFLite model, show result
#
# VISUALIZATIONS (all on one window):
#   • Live pose skeleton overlay
#   • Current knee angle (large, colour-coded)
#   • Current spine angle
#   • Recording state indicator (IDLE / RECORDING pulse)
#   • Knee angle + spine angle rolling graph (last 150 frames)
#   • Rep counter
#   • Last prediction label + colour-coded confidence bar
#   • Per-class probability bars (all 3 classes)
#   • Rep history log (last 5 reps)
#
# REQUIREMENTS:
#   pip install mediapipe opencv-python numpy
#   TFLite runtime included in: pip install tensorflow  OR
#                               pip install tflite-runtime
#
# USAGE:
#   python realtime_squat_test.py
#   python realtime_squat_test.py --model squat_classifier_fp16.tflite --camera 0

import argparse
import collections
import time
import numpy as np
import cv2

# ── MediaPipe ──────────────────────────────────────────────────────────────
import mediapipe as mp
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# ── TFLite ─────────────────────────────────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG  — edit to match your setup
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_MODEL       = r"squat_classifier_fp16.tflite"
DEFAULT_LABELS      = r"label_classes.txt"
DEFAULT_CAMERA      = 1

TARGET_FRAMES       = 64
SMOOTH_ALPHA        = 0.6
RECORD_START_ANGLE  = 138.0   # knee drops below this  → start recording
RECORD_STOP_ANGLE   = 135.0   # knee rises above this  → stop  recording (small hysteresis)
MIN_FRAMES_FOR_REP  = 8     # ignore blips shorter than this

# Joints used in training (MediaPipe indices)
JOINTS = {
    "nose":       0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_hip":      23, "r_hip":      24,
    "l_knee":     25, "r_knee":     26,
    "l_ankle":    27, "r_ankle":    28,
}
JOINT_IDX = list(JOINTS.values())

# Colours (BGR)
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,   0,   0)
C_GREEN   = (80,  200, 80)
C_YELLOW  = (50,  210, 230)
C_RED     = (80,  80,  220)
C_BLUE    = (220, 150, 50)
C_GREY    = (120, 120, 120)
C_DARK    = (30,  30,  30)
C_ORANGE  = (50,  160, 240)

# Label colours
LABEL_COLORS = {
    "perfect":      C_GREEN,
    "shallow":      C_YELLOW,
    "backrounding": C_RED,
}

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE PIPELINE  (must exactly match training)
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

def resample(seq, target=TARGET_FRAMES):
    T = len(seq)
    if T == target:
        return seq.astype(np.float32)
    idx = np.linspace(0, T - 1, target)
    out = np.zeros((target, seq.shape[1]), dtype=np.float32)
    for f in range(seq.shape[1]):
        out[:, f] = np.interp(idx, np.arange(T), seq[:, f])
    return out

def process_rep(frames, tiled_depth=False):
    """
    frames: (T,33,3) raw landmarks.

    tiled_depth=False  → dual-branch model (seq shape: J*9+1, depth shape: 8)
    tiled_depth=True   → single-branch model (seq shape: J*9+1+3, depth tiled in)
                         This is the OLD pipeline where depth scalars were
                         concatenated into every frame of the sequence.
    Returns: (seq_feat, depth_feat)
      dual  : seq (TARGET_FRAMES, J*9+1),   depth (8,)
      tiled : seq (TARGET_FRAMES, J*9+1+3), depth (8,)  [depth also in seq]
    """
    T         = len(frames)
    frames_sm = ema_smooth(frames, SMOOTH_ALPHA)
    norm      = np.array([torso_normalize(f) for f in frames_sm], dtype=np.float32)

    knee_r    = angle_batch(norm[:, 24], norm[:, 26], norm[:, 28])
    knee_l    = angle_batch(norm[:, 23], norm[:, 25], norm[:, 27])
    knee_ang  = (knee_r + knee_l) / 2.0

    # hip_drop from RAW frames (pre-normalisation) — fixes always-zero bug
    hip_y_raw = (frames_sm[:, 23, 1] + frames_sm[:, 24, 1]) / 2.0
    hip_drop  = float(np.max(hip_y_raw) - np.min(hip_y_raw))
    min_hip_y = float(np.max(hip_y_raw))

    mid_sho   = (norm[:, 11] + norm[:, 12]) / 2.0
    mid_hip2  = (norm[:, 23] + norm[:, 24]) / 2.0
    mid_kne   = (norm[:, 25] + norm[:, 26]) / 2.0

    # Forward lean angle — key BackRounding signal
    torso_vec = mid_sho - mid_hip2
    vertical  = np.array([[0.0, -1.0, 0.0]])
    dot_v     = np.sum(torso_vec * vertical, axis=-1)
    lean_ang  = np.degrees(np.arccos(
        np.clip(dot_v / (np.linalg.norm(torso_vec, axis=-1) + 1e-6), -1.0, 1.0)
    ))
    max_lean  = float(np.max(lean_ang))
    mean_lean = float(np.mean(lean_ang))
    sho_fwd   = float(np.mean(mid_sho[:, 0] - mid_hip2[:, 0]))

    spine_ang  = angle_batch(mid_sho, mid_hip2, mid_kne)
    min_spine  = float(np.min(spine_ang))
    mean_spine = float(np.mean(spine_ang))

    min_knee  = float(np.min(knee_ang))
    max_knee  = float(np.max(knee_ang))
    knee_rom  = max_knee - min_knee
    norm_min  = float(np.clip((min_knee - 70.0) / (170.0 - 70.0), 0.0, 1.0))

    depth_feat = np.array(
        [min_knee, knee_rom, hip_drop, norm_min, min_hip_y, max_knee,
         min_spine, mean_spine, max_lean, mean_lean, sho_fwd],  # (11,)
        dtype=np.float32
    )

    pos      = norm[:, JOINT_IDX, :].reshape(T, -1)
    vel      = np.concatenate([np.zeros_like(pos[:1]), np.diff(pos, axis=0)], axis=0)
    acc      = np.concatenate([np.zeros_like(vel[:1]), np.diff(vel, axis=0)], axis=0)

    if tiled_depth:
        # OLD single-branch pipeline: tile the first 3 depth scalars into every frame
        # gives feature dim = J*9 + 1 + 3 = 82 + 3 = 85
        d3 = np.tile(
            np.array([min_knee, knee_rom, hip_drop], dtype=np.float32),
            (T, 1)
        )
        seq_feat = np.concatenate([pos, vel, acc, knee_ang, d3], axis=1).astype(np.float32)
    else:
        # NEW dual-branch pipeline: depth kept separate
        seq_feat = np.concatenate([pos, vel, acc, knee_ang], axis=1).astype(np.float32)

    seq_feat = resample(seq_feat, TARGET_FRAMES)
    return seq_feat, depth_feat

def single_frame_knee_angle(lm33):
    """Quick knee angle for a single frame (no smoothing). Returns float."""
    a = np.array([[lm33[24, 0], lm33[24, 1], lm33[24, 2]],
                  [lm33[23, 0], lm33[23, 1], lm33[23, 2]]])
    b = np.array([[lm33[26, 0], lm33[26, 1], lm33[26, 2]],
                  [lm33[25, 0], lm33[25, 1], lm33[25, 2]]])
    c = np.array([[lm33[28, 0], lm33[28, 1], lm33[28, 2]],
                  [lm33[27, 0], lm33[27, 1], lm33[27, 2]]])
    angles = angle_batch(a, b, c)
    return float(np.mean(angles))

def single_frame_spine_angle(lm33):
    """Quick spine angle for a single frame."""
    mid_sho = (lm33[11] + lm33[12]) / 2.0
    mid_hip = (lm33[23] + lm33[24]) / 2.0
    mid_kne = (lm33[25] + lm33[26]) / 2.0
    a = mid_sho[None, :]
    b = mid_hip[None, :]
    c = mid_kne[None, :]
    return float(angle_batch(a, b, c)[0, 0])

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

def put_text(img, text, pos, scale=0.6, color=C_WHITE, thickness=1, font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(img, text, pos, font, scale, C_BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color,   thickness,     cv2.LINE_AA)

def draw_bar(img, x, y, w, h, fraction, fg_color, bg_color=C_DARK, label="", value_str=""):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    fill = int(np.clip(fraction, 0.0, 1.0) * w)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), fg_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), C_GREY, 1)
    if label:
        put_text(img, label, (x + 5, y + h - 5), scale=0.42, color=C_WHITE)
    if value_str:
        put_text(img, value_str, (x + w - 60, y + h - 5), scale=0.42, color=C_WHITE)

def angle_color(angle, lo=90, hi=170):
    """Green when near hi (standing), red when near lo (deep squat)."""
    t = np.clip((angle - lo) / (hi - lo), 0.0, 1.0)
    r = int(220 * (1 - t))
    g = int(200 * t)
    return (50, g, r)   # BGR

def draw_angle_graph(canvas, history_knee, history_spine, x, y, w, h, record_start=170):
    """Rolling line graph of knee + spine angle over last N frames."""
    # Background
    draw_rounded_rect(canvas, x, y, w, h, 6, (25, 25, 25), alpha=0.75)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), C_GREY, 1)

    # Grid lines
    for ang in [90, 120, 150, 170]:
        gy = y + h - int((ang - 60) / (190 - 60) * h)
        cv2.line(canvas, (x, gy), (x + w, gy), (55, 55, 55), 1)
        put_text(canvas, f"{ang}", (x + 3, gy - 3), scale=0.32, color=C_GREY)

    # Record threshold line
    ty = y + h - int((record_start - 60) / (190 - 60) * h)
    cv2.line(canvas, (x, ty), (x + w, ty), (80, 80, 180), 1)

    # Title
    put_text(canvas, "Knee (green)  Spine (orange)", (x + 8, y + 14), scale=0.38, color=C_GREY)

    def plot_series(data, color):
        if len(data) < 2:
            return
        pts = []
        for i, v in enumerate(data):
            px = x + int(i / (len(data) - 1) * (w - 1))
            py = y + h - int(np.clip((v - 60) / (190 - 60), 0, 1) * h)
            pts.append((px, py))
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)
        # Dot at latest
        cv2.circle(canvas, pts[-1], 4, color, -1, cv2.LINE_AA)

    plot_series(list(history_knee),  C_GREEN)
    plot_series(list(history_spine), C_ORANGE)


def draw_confidence_bars(canvas, probs, labels, x, y, w):
    """Vertical confidence bars for each class."""
    n      = len(labels)
    bar_w  = (w - (n + 1) * 6) // n
    bar_h  = 90
    top_y  = y

    for i, (label, prob) in enumerate(zip(labels, probs)):
        bx    = x + 6 + i * (bar_w + 6)
        color = LABEL_COLORS.get(label.lower(), C_BLUE)
        # Background
        cv2.rectangle(canvas, (bx, top_y), (bx + bar_w, top_y + bar_h), C_DARK, -1)
        # Fill from bottom
        fill  = int(prob * bar_h)
        if fill > 0:
            fy = top_y + bar_h - fill
            cv2.rectangle(canvas, (bx, fy), (bx + bar_w, top_y + bar_h), color, -1)
        cv2.rectangle(canvas, (bx, top_y), (bx + bar_w, top_y + bar_h), C_GREY, 1)
        # Percentage
        put_text(canvas, f"{prob*100:.0f}%", (bx + 3, top_y + bar_h - 5),
                 scale=0.42, color=C_WHITE)
        # Label below
        short = label[:6]
        put_text(canvas, short, (bx + 3, top_y + bar_h + 14),
                 scale=0.38, color=color)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(model_path, labels_path, camera_id):

    # ── Load labels ────────────────────────────────────────────────────────
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
        print(f"WARNING: Label file '{labels_path}' not found.")
        print("  Using alphabetical fallback: backrounding=0, perfect=1, shallow=2")
        print("  To fix permanently: copy label_classes.txt next to this script.")
        # LabelEncoder sorts alphabetically: backrounding < perfect < shallow
        label_names = ["backrounding", "perfect", "shallow"]
    # ── Load TFLite model ──────────────────────────────────────────────────
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_details  = interp.get_input_details()
    out_details  = interp.get_output_details()
    print("Model inputs:")
    for d in inp_details:
        print(f"  [{d['index']}] {d['name']}  shape={d['shape']}  dtype={d['dtype']}")

    # ── NUM_CLASSES from model output — always correct ─────────────────────
    NUM_CLASSES = int(out_details[0]["shape"][-1])
    print(f"Model output: {NUM_CLASSES} classes")

    # Auto-save label_classes.txt if missing, so future runs load correctly
    import os as _os
    if not _os.path.exists(labels_path) and len(label_names) == NUM_CLASSES:
        with open(labels_path, "w") as _f:
            for _i, _n in enumerate(label_names):
                _f.write(f"{_i},{_n}\n")
        print(f"Auto-saved '{labels_path}' for future runs.")

    # Reconcile label_names with actual model output size
    if len(label_names) != NUM_CLASSES:
        print(f"WARNING: label file has {len(label_names)} names but model has "
              f"{NUM_CLASSES} outputs — regenerating labels.")
        # Try to keep matching names, pad or trim as needed
        label_names = (label_names + [f"class_{i}" for i in range(NUM_CLASSES)])[:NUM_CLASSES]
    print(f"Labels: {label_names}")

    # ── Auto-detect model architecture ───────────────────────────────────
    num_inputs  = len(inp_details)
    TILED_DEPTH = (num_inputs == 1)   # single input → old tiled pipeline
    expected_seq_dim = int(inp_details[0]["shape"][-1])

    if TILED_DEPTH:
        print(f"Detected SINGLE-BRANCH model  (1 input, seq_dim={expected_seq_dim})")
        print("  → using tiled-depth feature pipeline to match training")
    else:
        print(f"Detected DUAL-BRANCH model  ({num_inputs} inputs)")

    def run_model(seq_feat, depth_feat):
        if num_inputs == 1:
            interp.set_tensor(inp_details[0]["index"],
                seq_feat[np.newaxis].astype(np.float32))
        else:
            for d in inp_details:
                if len(d["shape"]) == 3:          # (1, TARGET_FRAMES, seq_dim)
                    interp.set_tensor(d["index"],
                        seq_feat[np.newaxis].astype(np.float32))
                else:                              # (1, depth_dim)
                    # Auto-trim depth_feat to whatever size this model expects.
                    # Old model = 8, new retrained model = 11.
                    expected_depth = int(d["shape"][-1])
                    df = depth_feat[:expected_depth].copy()
                    interp.set_tensor(d["index"],
                        df[np.newaxis].astype(np.float32))
        interp.invoke()
        probs = interp.get_tensor(out_details[0]["index"])[0]
        return probs.astype(np.float32)

    # ── MediaPipe ─────────────────────────────────────────────────────────
    pose = mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1
    )

    # ── Webcam ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {W}x{H}")

    # ── State ─────────────────────────────────────────────────────────────
    recording       = False
    frame_buffer    = []           # list of (33,3) landmark arrays
    rep_count       = 0
    last_probs      = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
    last_label      = "—"
    last_conf       = 0.0
    last_min_knee   = 180.0

    rep_history     = collections.deque(maxlen=5)   # (label, conf, min_knee)

    graph_knee      = collections.deque(maxlen=150)
    graph_spine     = collections.deque(maxlen=150)

    fps_timer       = time.time()
    fps_val         = 0.0
    frame_count_fps = 0
    pulse           = 0            # blink counter for RECORDING indicator

    # ═════════════════════════════════════════════════════════════════════
    # PANEL LAYOUT  (right side panel 340 px wide)
    # ═════════════════════════════════════════════════════════════════════
    PANEL_W = 340
    GRAPH_H = 140

    print("\nPress  Q  or  ESC  to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = pose.process(rgb)

        # ── FPS ──────────────────────────────────────────────────────────
        frame_count_fps += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_val         = frame_count_fps / (now - fps_timer)
            fps_timer       = now
            frame_count_fps = 0

        current_knee  = 180.0
        current_spine = 180.0
        lm_array      = None

        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            lm_array = np.array(
                [[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32
            )   # (33,3)

            current_knee  = single_frame_knee_angle(lm_array)
            current_spine = single_frame_spine_angle(lm_array)

            graph_knee.append(current_knee)
            graph_spine.append(current_spine)

            # ── Rep detection state machine ───────────────────────────
            if not recording:
                if current_knee < RECORD_START_ANGLE:
                    recording    = True
                    frame_buffer = [lm_array.copy()]
            else:
                frame_buffer.append(lm_array.copy())
                if current_knee >= RECORD_STOP_ANGLE:
                    # Rep ended — process
                    recording = False
                    if len(frame_buffer) >= MIN_FRAMES_FOR_REP:
                        frames    = np.stack(frame_buffer, axis=0)  # (T,33,3)
                        seq_feat, depth_feat = process_rep(frames, tiled_depth=TILED_DEPTH)
                        probs     = run_model(seq_feat, depth_feat)
                        pred_idx  = int(np.argmax(probs))
                        last_probs = probs
                        last_label = label_names[pred_idx]
                        last_conf  = float(probs[pred_idx])
                        last_min_knee = float(np.min(
                            [(single_frame_knee_angle(f)) for f in frame_buffer]
                        ))
                        rep_count += 1
                        rep_history.appendleft(
                            (last_label, last_conf, last_min_knee)
                        )
                        # Console debug — see exact probabilities each rep
                        print(f"Rep #{rep_count}: {last_label} ({last_conf*100:.1f}%)"
                              f"  min_knee={last_min_knee:.1f}°"
                              f"  raw_probs={[f'{p*100:.1f}%' for p in probs]}")
                    frame_buffer = []

            # ── Draw skeleton ─────────────────────────────────────────
            mp_drawing.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=C_GREEN, thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(200, 200, 200), thickness=2)
            )

            # ── Annotate key joints ───────────────────────────────────
            h_img, w_img = frame.shape[:2]
            def lm_px(idx):
                return (int(lms[idx].x * w_img), int(lms[idx].y * h_img))

            # Knee angle arc label
            r_knee_px = lm_px(26)
            kc = angle_color(current_knee)
            put_text(frame, f"{current_knee:.0f}", 
                     (r_knee_px[0] - 30, r_knee_px[1] - 15),
                     scale=0.7, color=kc, thickness=2)

            # Hip depth line
            l_hip_px = lm_px(23);  r_hip_px = lm_px(24)
            cv2.line(frame, l_hip_px, r_hip_px, C_BLUE, 2)

        # ═══════════════════════════════════════════════════════════════
        # RIGHT PANEL
        # ═══════════════════════════════════════════════════════════════
        panel_x = W - PANEL_W
        # Semi-transparent panel background
        draw_rounded_rect(frame, panel_x, 0, PANEL_W, H, 0, (20, 20, 20), alpha=0.60)

        cy = 18   # cursor y

        # ── FPS ──────────────────────────────────────────────────────
        put_text(frame, f"FPS {fps_val:.1f}", (panel_x + 8, cy), scale=0.42, color=C_GREY)
        cy += 22

        # ── Recording state ───────────────────────────────────────────
        pulse = (pulse + 1) % 40
        if recording:
            rec_color = C_RED if pulse < 20 else C_ORANGE
            draw_rounded_rect(frame, panel_x + 8, cy, PANEL_W - 16, 28, 5,
                              rec_color, alpha=0.7)
            put_text(frame, f"● RECORDING  ({len(frame_buffer)} frames)",
                     (panel_x + 18, cy + 19), scale=0.52, color=C_WHITE, thickness=1)
        else:
            draw_rounded_rect(frame, panel_x + 8, cy, PANEL_W - 16, 28, 5,
                              (40, 40, 40), alpha=0.7)
            put_text(frame, "◉ IDLE — waiting for squat",
                     (panel_x + 18, cy + 19), scale=0.52, color=C_GREY, thickness=1)
        cy += 36

        # ── Live angles ───────────────────────────────────────────────
        kc = angle_color(current_knee)
        put_text(frame, "KNEE ANGLE", (panel_x + 10, cy), scale=0.42, color=C_GREY)
        cy += 4
        put_text(frame, f"{current_knee:.1f}°", (panel_x + 10, cy + 34),
                 scale=1.2, color=kc, thickness=2)
        cy += 42

        sc = angle_color(current_spine, lo=120, hi=180)
        put_text(frame, "SPINE ANGLE", (panel_x + 10, cy), scale=0.42, color=C_GREY)
        cy += 4
        put_text(frame, f"{current_spine:.1f}°", (panel_x + 10, cy + 28),
                 scale=0.9, color=sc, thickness=2)
        cy += 36

        # ── Graph ─────────────────────────────────────────────────────
        draw_angle_graph(frame, graph_knee, graph_spine,
                         panel_x + 4, cy, PANEL_W - 8, GRAPH_H)
        cy += GRAPH_H + 8

        # ── Rep counter ───────────────────────────────────────────────
        draw_rounded_rect(frame, panel_x + 8, cy, PANEL_W - 16, 38, 6,
                          (40, 40, 80), alpha=0.7)
        put_text(frame, "REPS", (panel_x + 16, cy + 14), scale=0.45, color=C_GREY)
        put_text(frame, str(rep_count), (panel_x + 70, cy + 30),
                 scale=1.1, color=C_BLUE, thickness=2)
        cy += 46

        # ── Last prediction ───────────────────────────────────────────
        pred_color = LABEL_COLORS.get(last_label.lower(), C_BLUE)
        draw_rounded_rect(frame, panel_x + 8, cy, PANEL_W - 16, 44, 6,
                          (30, 30, 30), alpha=0.7)
        put_text(frame, "LAST REP", (panel_x + 16, cy + 14), scale=0.42, color=C_GREY)
        put_text(frame, last_label.upper(), (panel_x + 16, cy + 34),
                 scale=0.75, color=pred_color, thickness=2)
        # Confidence bar
        draw_bar(frame, panel_x + 130, cy + 20, PANEL_W - 144, 18,
                 last_conf, pred_color,
                 value_str=f"{last_conf*100:.0f}%")
        cy += 52

        # ── Per-class probability bars ────────────────────────────────
        put_text(frame, "CONFIDENCE", (panel_x + 10, cy), scale=0.42, color=C_GREY)
        cy += 4
        draw_confidence_bars(frame, last_probs, label_names,
                             panel_x + 4, cy, PANEL_W - 8)
        cy += 112

        # ── Rep history ───────────────────────────────────────────────
        put_text(frame, "HISTORY", (panel_x + 10, cy), scale=0.42, color=C_GREY)
        cy += 4
        for i, (rl, rc, rk) in enumerate(rep_history):
            rc_col = LABEL_COLORS.get(rl.lower(), C_BLUE)
            rep_num = rep_count - i
            put_text(frame,
                     f"#{rep_num}  {rl:<12}  {rc*100:.0f}%  knee:{rk:.0f}°",
                     (panel_x + 10, cy + 14 + i * 18),
                     scale=0.38, color=rc_col)
        cy += len(rep_history) * 18 + 6

        # ── Threshold indicator on main frame ─────────────────────────
        thresh_text = f"Record threshold: {RECORD_START_ANGLE:.0f}°"
        put_text(frame, thresh_text, (10, H - 12), scale=0.45, color=C_GREY)

        # ── No pose warning ───────────────────────────────────────────
        if not res.pose_landmarks:
            put_text(frame, "NO POSE DETECTED", (W // 2 - 130, H // 2),
                     scale=1.0, color=C_RED, thickness=2)

        # ── Debug overlay (bottom-left) ───────────────────────────────
        dbg_x, dbg_y = 10, H - 120
        draw_rounded_rect(frame, dbg_x, dbg_y, 320, 110, 5, (20,20,20), alpha=0.7)
        put_text(frame, "DEBUG", (dbg_x+8, dbg_y+14), scale=0.38, color=C_GREY)
        put_text(frame,
                 f"knee={current_knee:.1f}  spine={current_spine:.1f}",
                 (dbg_x+8, dbg_y+30), scale=0.40, color=C_WHITE)
        put_text(frame,
                 f"buf={len(frame_buffer)}  recs={recording}",
                 (dbg_x+8, dbg_y+46), scale=0.40, color=C_WHITE)
        # Raw probabilities for every class
        prob_str = "  ".join(
            [f"{label_names[i][:5]}={last_probs[i]*100:.0f}%"
             for i in range(NUM_CLASSES)]
        )
        put_text(frame, prob_str, (dbg_x+8, dbg_y+62), scale=0.38, color=C_YELLOW)
        # Model info
        put_text(frame,
                 f"model={'single' if TILED_DEPTH else 'dual'}-branch  "
                 f"seq_dim={expected_seq_dim}",
                 (dbg_x+8, dbg_y+78), scale=0.38, color=C_GREY)
        put_text(frame, f"inputs={num_inputs}  classes={NUM_CLASSES}",
                 (dbg_x+8, dbg_y+94), scale=0.38, color=C_GREY)

        cv2.imshow("Squat Classifier — Real-time", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        # Press R to reset rep counter
        if key in (ord("r"), ord("R")):
            rep_count   = 0
            rep_history.clear()
            last_probs  = np.ones(NUM_CLASSES) / NUM_CLASSES
            last_label  = "—"
            last_conf   = 0.0
            print("Rep counter reset.")

    cap.release()
    pose.close()
    cv2.destroyAllWindows()
    print(f"\nSession ended.  Total reps: {rep_count}")
    if rep_history:
        print("Rep history:")
        for i, (rl, rc, rk) in enumerate(reversed(rep_history)):
            print(f"  Rep {i+1}: {rl}  conf={rc*100:.0f}%  min_knee={rk:.1f}°")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time squat classifier")
    parser.add_argument("--model",  default=DEFAULT_MODEL,  help="TFLite model path")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="label_classes.txt path")
    parser.add_argument("--camera", default=DEFAULT_CAMERA, type=int, help="Camera index")
    args = parser.parse_args()

    print(f"Model:  {args.model}")
    print(f"Labels: {args.labels}")
    print(f"Camera: {args.camera}")
    print(f"Record when knee < {RECORD_START_ANGLE}°, stop when > {RECORD_STOP_ANGLE}°")
    print("Controls:  Q / ESC = quit    R = reset counter\n")

    main(args.model, args.labels, args.camera)