"""
realtime_shrug_test.py
======================
Baseline 60 frames measures:
  rest_sho_y — resting shoulder position  (line is drawn HERE)

Logic:
  shoulder rises RISE_DELTA above rest  → start recording
  shoulder returns within STOP_TOL of rest → stop → run model

The stop has a dead-zone: small drops during hold/squeeze don't stop recording.

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
SMOOTH_ALPHA    = 0.5
MIN_FRAMES      = 8
BASELINE_FRAMES = 60     # stand still ~2s at startup

# Trigger: shoulder must rise this fraction of baseline noise above rest
# Computed from baseline std — adapts to camera angle automatically
RISE_MULT       = 3.0    # start_threshold = rest_sho_y - baseline_std * RISE_MULT
                          # raise to 4.0 if breathing triggers false starts

# Stop: shoulder must return within this fraction of the rest position
# Dead-zone prevents small drops during hold from stopping recording
STOP_TOL_MULT   = 2.0    # stop_threshold = rest_sho_y - baseline_std * STOP_TOL_MULT
                          # STOP_TOL_MULT < RISE_MULT  so stop > start threshold
                          # lower to 1.5 if reps stop too late

ARM_MAX_ANGLE   = 80.0   # upper-arm deviation from vertical — discard if exceeded

# ──────────────────────────────────────────────────────────────────
# JOINTS
# ──────────────────────────────────────────────────────────────────
JOINT_IDX = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]

# ──────────────────────────────────────────────────────────────────
# FEATURE PIPELINE  (must match training exactly)
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
    return float((lm[11,1] + lm[12,1]) / 2.0)

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

    # ── VAE encoder ────────────────────────────────────────────────
    import os as _os
    vae_interp   = None; vae_inp = None; vae_out_det = None
    kl_threshold = None; vae_sc_mean = None; vae_sc_scale = None

    _vae_candidates = [
        model_path.replace("shrug_classifier_fp16.tflite", "shrug_vae_encoder.tflite"),
        _os.path.join(_os.path.dirname(model_path), "shrug_vae_encoder.tflite"),
    ]
    for _vc in _vae_candidates:
        if _os.path.exists(_vc):
            try:
                vae_interp = Interpreter(model_path=_vc, num_threads=2)
                vae_interp.allocate_tensors()
                vae_inp     = vae_interp.get_input_details()
                vae_out_det = vae_interp.get_output_details()
                print(f"VAE loaded: {_vc}")
            except Exception as e:
                print(f"VAE load failed: {e}"); vae_interp = None
            break

    _stats_candidates = [
        model_path.replace("shrug_classifier_fp16.tflite","best_shrug_model_feature_stats.npz"),
        _os.path.join(_os.path.dirname(model_path), "best_shrug_model_feature_stats.npz"),
    ]
    for _sc in _stats_candidates:
        if _os.path.exists(_sc):
            try:
                _st = np.load(_sc, allow_pickle=True)
                if "vae_kl_threshold" in _st:
                    kl_threshold = float(_st["vae_kl_threshold"][0])
                    vae_sc_mean  = _st["vae_scaler_mean"].astype(np.float32)
                    vae_sc_scale = _st["vae_scaler_scale"].astype(np.float32)
                    print(f"KL threshold={kl_threshold:.3f}")
            except Exception as e:
                print(f"Stats load failed: {e}")
            break

    def compute_kl_vae(depth_feat):
        if vae_interp is None or vae_sc_mean is None:
            return None
        n = min(len(depth_feat), len(vae_sc_mean))
        x = ((depth_feat[:n] - vae_sc_mean[:n]) / (vae_sc_scale[:n] + 1e-6))\
            .reshape(1,-1).astype(np.float32)
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
                    interp.set_tensor(d["index"], depth_feat[:n][np.newaxis].astype(np.float32))
        interp.invoke()
        return interp.get_tensor(out[0]["index"])[0].astype(np.float32)

    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── State ──────────────────────────────────────────────────────
    baseline_sho_ys = []
    baseline_done   = False
    rest_sho_y      = None   # resting shoulder y — line drawn here
    start_thresh    = None   # image y: below this = shoulder rose enough → start
    stop_thresh     = None   # image y: above this = shoulder returned → stop
                             # stop_thresh > start_thresh (dead-zone between them)

    recording     = False
    frame_buffer  = []
    rep_count     = 0
    last_label    = "—"
    last_conf     = 0.0
    discard_msg   = ""
    discard_timer = 0
    history       = collections.deque(maxlen=5)

    COLORS = {"perfect":(50,200,80), "bent_elbow":(0,100,255), "unknown":(120,120,120)}
    print("Stand still 2 seconds...\nQ/ESC=quit  R=reset\n")

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
                    rest_sho_y   = float(np.mean(baseline_sho_ys))
                    noise_std    = float(np.std(baseline_sho_ys))
                    # In image space: smaller y = higher on screen = shoulder raised
                    # start_thresh: shoulder rises above rest by RISE_MULT * noise
                    start_thresh = rest_sho_y - noise_std * RISE_MULT
                    # stop_thresh:  shoulder back within STOP_TOL_MULT * noise of rest
                    # dead-zone = [start_thresh .. stop_thresh] — no state change here
                    stop_thresh  = rest_sho_y - noise_std * STOP_TOL_MULT
                    baseline_done = True
                    print(f"Baseline:  rest={rest_sho_y:.4f}  noise_std={noise_std:.4f}")
                    print(f"  start when sho_y < {start_thresh:.4f}  (rise {noise_std*RISE_MULT:.4f})")
                    print(f"  stop  when sho_y > {stop_thresh:.4f}   (dead-zone={noise_std*(RISE_MULT-STOP_TOL_MULT):.4f})")

            # ── REP LOGIC ──────────────────────────────────────────
            else:
                sho_y = get_sho_y(lm)

                if not recording:
                    if sho_y < start_thresh:
                        recording    = True
                        frame_buffer = [lm.copy()]
                        print(f"● REC start  sho_y={sho_y:.3f}")
                else:
                    # ARM GUARD
                    if arm_angle > ARM_MAX_ANGLE:
                        recording    = False
                        discard_msg  = f"ARMS DRIFTED ({arm_angle:.1f}°)"
                        discard_timer = 60
                        frame_buffer = []
                        print(f"✗ {discard_msg}")
                    else:
                        frame_buffer.append(lm.copy())
                        # Stop only when shoulder returns past the dead-zone
                        if sho_y > stop_thresh:
                            recording = False
                            n = len(frame_buffer)
                            print(f"■ REC stop  frames={n}")
                            if n >= MIN_FRAMES:
                                seq, dep = process_rep(np.stack(frame_buffer), TARGET_FRAMES)

                                # ── VAE OOD gate ──────────────────
                                kl = compute_kl_vae(dep)
                                if kl is not None and kl_threshold is not None:
                                    if kl > kl_threshold:
                                        print(f"[VAE-OOD] KL={kl:.2f} > {kl_threshold:.2f} — rejected")
                                        last_label = "unknown"; last_conf = 0.0
                                        frame_buffer = []
                                        continue
                                    print(f"[VAE-OK]  KL={kl:.2f}")

                                # ── Classifier ────────────────────
                                probs  = run_model(seq, dep)
                                i_max  = int(np.argmax(probs))
                                conf   = float(probs[i_max])
                                last_label = label_names[i_max] if i_max < len(label_names) else str(i_max)
                                last_conf  = conf
                                rep_count += 1
                                history.appendleft((last_label, conf))
                                kl_str = f"  KL={kl:.2f}" if kl else ""
                                print(f"→ Rep #{rep_count}: {last_label} ({conf*100:.1f}%){kl_str}")
                            frame_buffer = []

            # Skeleton
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(180,180,180), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(80,80,80), thickness=1))
            arm_col = (0,0,220) if arm_angle > ARM_MAX_ANGLE else (180,180,180)
            for si, ei in [(11,13),(12,14)]:
                cv2.line(frame,
                         (int(lm[si,0]*W), int(lm[si,1]*H)),
                         (int(lm[ei,0]*W), int(lm[ei,1]*H)),
                         arm_col, 3)

        # ── TARGET LINE at rest shoulder position ───────────────────
        if baseline_done and rest_sho_y is not None:
            rest_px  = int(rest_sho_y * H)
            start_px = int(start_thresh * H)
            stop_px  = int(stop_thresh * H)

            # Rest line (white, thin) — where shoulder sits normally
            cv2.line(frame, (0, rest_px), (W, rest_px), (200,200,200), 1)

            # Start threshold (cyan) — must rise above this
            cv2.line(frame, (0, start_px), (W, start_px),
                     (0,220,80) if recording else (0,180,220), 2)
            cv2.putText(frame, "START" if not recording else "RECORDING",
                        (8, start_px - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0,220,80) if recording else (0,180,220), 1)

            # Dead-zone band (semi-transparent yellow) between start and stop
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, start_px), (W, stop_px), (0,200,200), -1)
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

            # Stop threshold (orange) — must return past this
            cv2.line(frame, (0, stop_px), (W, stop_px), (0,140,255), 1)
            cv2.putText(frame, "STOP",
                        (8, stop_px + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,140,255), 1)

        # ── HUD ────────────────────────────────────────────────────
        if recording:
            cv2.circle(frame, (W-25, 25), 10, (0,0,255), -1)
            cv2.putText(frame, f"REC {len(frame_buffer)}f",
                        (W-100,32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

        ang_col = (0,0,220) if arm_angle > ARM_MAX_ANGLE else (50,200,80)
        cv2.putText(frame, f"ARM {arm_angle:.1f}°",
                    (20,42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ang_col, 2)

        if discard_timer > 0:
            cv2.putText(frame, discard_msg,
                        (W//2-180, H//2),
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
            _kl = f"  KL<{kl_threshold:.2f}" if kl_threshold else ""
            cv2.putText(frame,
                        f"sho={_sy:.3f} rest={rest_sho_y:.3f} start={start_thresh:.3f} stop={stop_thresh:.3f}{_kl}",
                        (20,H-78), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110,110,110), 1)

        cv2.imshow("Shrug Classifier", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27): break
        if key in (ord("r"), ord("R")):
            baseline_sho_ys=[]; baseline_done=False
            rest_sho_y=None; start_thresh=None; stop_thresh=None
            recording=False; frame_buffer=[]
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