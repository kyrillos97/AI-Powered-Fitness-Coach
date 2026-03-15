"""
realtime_shrug_test.py
======================
Baseline 60 frames measures:
  target_y   — fixed line halfway between shoulder and mouth (drawn forever)
  rest_sho_y — resting shoulder position

Logic:
  shoulder at rest         → show "raise to line"
  shoulder reaches line    → start recording
  shoulder returns to rest → stop → run model

Keys: Q / ESC = quit    R = reset
"""

import argparse
import collections
import numpy as np
import cv2
import mediapipe as mp

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.60
SMOOTH_ALPHA    = 0.5
MIN_FRAMES      = 8
BASELINE_FRAMES = 60    # stand still for ~2s at startup

# How close shoulder must get to target_y to count as "reached"
# in fraction of the total rest-to-target distance
REACH_FRACTION  = 0.15   # top 15% of the travel counts as reached
# How close shoulder must return to rest_sho_y to count as "returned"
RETURN_FRACTION = 0.15   # bottom 15% of the travel counts as returned

ARM_MAX_ANGLE   = 70.0   # upper-arm deviation from vertical — discard if exceeded

# ──────────────────────────────────────────────────────────────────
# JOINTS
# ──────────────────────────────────────────────────────────────────
JOINT_IDX = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]

# ──────────────────────────────────────────────────────────────────
# FEATURE PIPELINE
# ──────────────────────────────────────────────────────────────────
def angle_batch(a, b, c):
    ba = a - b; bc = c - b
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
    elbow_l  = angle_batch(norm[:,11], norm[:,13], norm[:,15])
    elbow_r  = angle_batch(norm[:,12], norm[:,14], norm[:,16])
    sho_y_l  = -norm[:,11,1];  sho_y_r = -norm[:,12,1]
    sho_elev = ((sho_y_l + sho_y_r) / 2.0)[:,None]
    sho_vel  = np.concatenate([np.zeros((1,1),np.float32), np.diff(sho_elev,axis=0)],axis=0)
    sho_acc  = np.concatenate([np.zeros((1,1),np.float32), np.diff(sho_vel, axis=0)],axis=0)
    min_el_l = float(np.min(elbow_l)); max_el_l = float(np.max(elbow_l))
    min_el_r = float(np.min(elbow_r)); max_el_r = float(np.max(elbow_r))
    ev_l     = np.abs(np.diff(elbow_l[:,0])); ev_r = np.abs(np.diff(elbow_r[:,0]))
    sho_max  = float(np.max(sho_elev))
    ear_y_l  = -norm[:,7,1];  ear_y_r = -norm[:,8,1]
    wrist_l  = -norm[:,15,1]; wrist_r = -norm[:,16,1]
    nose_y   = -norm[:,0,1]
    peak_f   = int(np.argmax(sho_elev[:,0]))
    t_l = float(np.argmin(elbow_l[:,0]))/max(T-1,1)
    t_r = float(np.argmin(elbow_r[:,0]))/max(T-1,1)
    el_mean = float(np.mean(elbow_l)); el_med = float(np.median(elbow_l))
    el_std  = float(np.std(elbow_l)) + 1e-6
    depth_feat = np.array([
        min_el_l, min_el_r, max_el_l-min_el_l, max_el_r-min_el_r,
        float(np.max(np.abs(elbow_l-elbow_r))),
        sho_max, sho_max-float(np.min(sho_elev)),
        float(np.min((np.abs(sho_y_l-ear_y_l)+np.abs(sho_y_r-ear_y_r))/2.0)),
        float(np.max(np.maximum(wrist_l-wrist_l[0], wrist_r-wrist_r[0]))),
        float(np.max(np.abs(sho_vel))), float(np.mean(np.abs(sho_acc))),
        float(np.max(ev_l)) if len(ev_l)>0 else 0.0,
        float(np.max(ev_r)) if len(ev_r)>0 else 0.0,
        float(np.std(sho_y_l-sho_y_r)),
        float(np.max(nose_y)-np.min(nose_y)),
        float(np.mean((np.abs(norm[:,15,1])+np.abs(norm[:,16,1]))/2.0)),
        float(abs(sho_y_l[peak_f]-sho_y_r[peak_f])),
        float(np.max(np.abs(norm[:,15,0]-norm[:,16,0]))),
        float(abs(float(elbow_l[peak_f,0])-float(elbow_r[peak_f,0]))),
        t_l, t_r, (el_mean-el_med)/el_std, float(abs(t_l-t_r)),
    ], dtype=np.float32)
    pos      = norm[:,JOINT_IDX,:].reshape(T,-1)
    vel      = np.concatenate([np.zeros_like(pos[:1]), np.diff(pos,axis=0)],axis=0)
    acc      = np.concatenate([np.zeros_like(vel[:1]), np.diff(vel,axis=0)],axis=0)
    seq_feat = np.concatenate([pos,vel,acc,elbow_l,elbow_r,sho_elev,sho_vel,sho_acc],axis=1).astype(np.float32)
    return resample(seq_feat, target_frames), depth_feat

def get_sho_y(lm):
    """Average shoulder y in image space (0=top, 1=bottom)."""
    return float((lm[11,1] + lm[12,1]) / 2.0)

def get_sho_x(lm):
    return float((lm[11,0] + lm[12,0]) / 2.0)

def get_mouth_y(lm):
    return float((lm[9,1] + lm[10,1]) / 2.0)

def upper_arm_angle(lm):
    down = np.array([0.0,1.0,0.0], np.float32)
    def va(v):
        n = np.linalg.norm(v)
        return float(np.degrees(np.arccos(np.clip(np.dot(v/n,down),-1,1)))) if n>1e-6 else 0.0
    return max(va(lm[13]-lm[11]), va(lm[14]-lm[12]))

# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────
def main(model_path, labels_path, camera_id):

    label_names = ["bent_elbow", "perfect"]
    try:
        with open(labels_path) as f:
            lines = [l.strip() for l in f if l.strip()]
            if lines:
                label_names = [l.split(",",1)[1].strip() if "," in l else l for l in lines]
    except FileNotFoundError:
        pass
    print(f"Labels: {label_names}")

    try:
        interp = Interpreter(model_path=model_path, num_threads=4)
    except TypeError:
        interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    NUM_CLASSES = int(out[0]["shape"][-1])

    TARGET_FRAMES = 48
    for d in inp:
        if len(d["shape"]) == 3:
            TARGET_FRAMES = int(d["shape"][1]); break
    print(f"TARGET_FRAMES={TARGET_FRAMES}  classes={NUM_CLASSES}")

    def run_model(seq_feat, depth_feat):
        if len(inp) == 1:
            interp.set_tensor(inp[0]["index"], seq_feat[np.newaxis].astype(np.float32))
        else:
            for d in inp:
                if len(d["shape"]) == 3:
                    interp.set_tensor(d["index"], seq_feat[np.newaxis].astype(np.float32))
                else:
                    n = int(d["shape"][-1])
                    interp.set_tensor(d["index"], depth_feat[:n][np.newaxis].astype(np.float32))
        interp.invoke()
        return interp.get_tensor(out[0]["index"])[0].astype(np.float32)

    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── Baseline state ─────────────────────────────────────────────
    baseline_sho_ys = []   # collect shoulder y during baseline
    baseline_done   = False

    # After baseline these are set and never change:
    rest_sho_y  = None    # average resting shoulder y (image space)
    target_y    = None    # fixed line = halfway between rest shoulder and mouth
    travel      = None    # total distance from rest to target (image y units)

    # ── Rep state ──────────────────────────────────────────────────
    recording     = False
    frame_buffer  = []
    reached_line  = False   # shoulder touched target_y this rep
    rep_count     = 0
    last_label    = "—"
    last_conf     = 0.0
    discard_msg   = ""
    discard_timer = 0
    history       = collections.deque(maxlen=5)

    COLORS = {"perfect":(50,200,80), "bent_elbow":(0,100,255), "unknown":(120,120,120)}
    print("Stand still for 2 seconds to calibrate...\nPress Q/ESC=quit  R=reset\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        res   = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        H, W  = frame.shape[:2]

        arm_angle = 0.0

        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            lm  = np.array([[l.x,l.y,l.z] for l in lms], dtype=np.float32)
            arm_angle = upper_arm_angle(lm)

            # ── BASELINE ───────────────────────────────────────────
            if not baseline_done:
                baseline_sho_ys.append(get_sho_y(lm))
                pct = int(len(baseline_sho_ys) / BASELINE_FRAMES * 100)
                cv2.putText(frame, f"Stand still... {pct}%",
                            (W//2-130, H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,200), 2)

                if len(baseline_sho_ys) >= BASELINE_FRAMES:
                    rest_sho_y = float(np.mean(baseline_sho_ys))
                    mouth_y    = get_mouth_y(lm)
                    # target = midpoint between resting shoulder and mouth
                    # image y: shoulder > mouth (shoulder is lower on screen)
                    target_y   = (rest_sho_y + mouth_y) / 2.0
                    travel     = rest_sho_y - target_y   # always positive
                    baseline_done = True
                    print(f"Baseline done:")
                    print(f"  rest_sho_y = {rest_sho_y:.4f}")
                    print(f"  mouth_y    = {mouth_y:.4f}")
                    print(f"  target_y   = {target_y:.4f}  (fixed line)")
                    print(f"  travel     = {travel:.4f}  (shoulder must rise this far)")

            # ── REP LOGIC ──────────────────────────────────────────
            else:
                sho_y = get_sho_y(lm)

                # Thresholds in image y space (higher y = lower on screen)
                reach_y  = target_y  + travel * REACH_FRACTION   # shoulder y <= this = at line
                return_y = target_y  + travel * (1.0 - RETURN_FRACTION)  # shoulder y >= this = back at rest

                if not recording:
                    if sho_y <= reach_y:   # shoulder reached the line
                        recording    = True
                        reached_line = True
                        frame_buffer = [lm.copy()]
                        print(f"● REC start  sho_y={sho_y:.3f}  line={target_y:.3f}")
                else:
                    # ARM GUARD
                    if arm_angle > ARM_MAX_ANGLE:
                        recording    = False
                        reached_line = False
                        discard_msg  = f"ARMS DRIFTED ({arm_angle:.1f}° > {ARM_MAX_ANGLE:.0f}°)"
                        discard_timer = 60
                        frame_buffer = []
                        print(f"✗ {discard_msg}")
                    else:
                        frame_buffer.append(lm.copy())
                        # Stop when shoulder returns to rest position
                        if sho_y >= return_y:
                            recording    = False
                            reached_line = False
                            n = len(frame_buffer)
                            print(f"■ REC stop  frames={n}")
                            if n >= MIN_FRAMES:
                                seq, dep = process_rep(np.stack(frame_buffer), TARGET_FRAMES)
                                probs    = run_model(seq, dep)
                                i_max    = int(np.argmax(probs))
                                conf     = float(probs[i_max])
                                if conf >= CONF_THRESHOLD:
                                    last_label = label_names[i_max] if i_max < len(label_names) else str(i_max)
                                    last_conf  = conf
                                    rep_count += 1
                                    history.appendleft((last_label, conf))
                                    print(f"→ Rep #{rep_count}: {last_label} ({conf*100:.1f}%)")
                                else:
                                    last_label = "unknown"
                                    last_conf  = conf
                                    print(f"→ Low conf {conf*100:.1f}%")
                            frame_buffer = []

            # Skeleton
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(180,180,180), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(80,80,80), thickness=1))

            # Upper arm lines coloured by guard
            arm_col = (0,0,220) if arm_angle > ARM_MAX_ANGLE else (180,180,180)
            for si, ei in [(11,13),(12,14)]:
                cv2.line(frame,
                         (int(lm[si,0]*W), int(lm[si,1]*H)),
                         (int(lm[ei,0]*W), int(lm[ei,1]*H)),
                         arm_col, 3)

        # ── DRAW FIXED TARGET LINE ──────────────────────────────────
        if baseline_done and target_y is not None:
            line_px  = int(target_y * H)
            line_col = (0,220,80) if recording else (0,180,220)
            cv2.line(frame, (0, line_px), (W, line_px), line_col, 2)
            cv2.putText(frame, "TARGET — raise shoulders here",
                        (20, line_px - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_col, 2)

            # Instruction when idle
            if not recording and res.pose_landmarks:
                sho_y_now = get_sho_y(np.array(
                    [[l.x,l.y,l.z] for l in res.pose_landmarks.landmark], np.float32))
                reach_y_now = target_y + travel * REACH_FRACTION
                if sho_y_now > reach_y_now:
                    cv2.putText(frame, "Raise shoulders to the line",
                                (20, line_px + 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,220), 2)

        # ── HUD ────────────────────────────────────────────────────
        if recording:
            cv2.circle(frame, (W-25, 25), 10, (0,0,255), -1)
            cv2.putText(frame, f"REC {len(frame_buffer)}f",
                        (W-100,32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

        ang_col = (0,0,220) if arm_angle > ARM_MAX_ANGLE else (50,200,80)
        cv2.putText(frame, f"ARM {arm_angle:.1f}°/{ARM_MAX_ANGLE:.0f}°",
                    (20,42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ang_col, 2)

        if discard_timer > 0:
            cv2.putText(frame, discard_msg,
                        (W//2-200, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,80,255), 2)
            discard_timer -= 1

        cv2.putText(frame, f"REPS: {rep_count}",
                    (20,H-55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

        col = COLORS.get(last_label.lower(), (180,180,180))
        cv2.putText(frame, f"{last_label.upper()}  {last_conf*100:.0f}%",
                    (20,H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)

        for i,(hl,hc) in enumerate(history):
            cv2.putText(frame, f"{hl}  {hc*100:.0f}%",
                        (W-210,55+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        COLORS.get(hl.lower(),(180,180,180)), 1)

        if baseline_done and res.pose_landmarks:
            _sy = get_sho_y(np.array([[l.x,l.y,l.z] for l in res.pose_landmarks.landmark],np.float32))
            cv2.putText(frame,
                        f"sho={_sy:.3f} target={target_y:.3f} rest={rest_sho_y:.3f} arm={arm_angle:.1f}° rec={recording}",
                        (20,H-78), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110,110,110), 1)

        cv2.imshow("Shrug Classifier", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27): break
        if key in (ord("r"), ord("R")):
            baseline_sho_ys=[]; baseline_done=False
            rest_sho_y=None; target_y=None; travel=None
            recording=False; frame_buffer=[]; reached_line=False
            rep_count=0; history.clear()
            last_label="—"; last_conf=0.0; discard_timer=0
            print("Reset.")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=r"D:\AI-Powered-Fitness-Coach\AI Workout\shrugs\shrug_classifier_fp16.tflite")
    parser.add_argument("--labels", default=r"D:\AI-Powered-Fitness-Coach\AI Workout\shrugs\shrug_label_classes.txt")
    parser.add_argument("--camera", type=int, default=1)
    args = parser.parse_args()
    main(args.model, args.labels, args.camera)