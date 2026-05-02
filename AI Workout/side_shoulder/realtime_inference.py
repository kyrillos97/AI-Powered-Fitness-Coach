"""
================================================================
  REAL-TIME SIDE SHOULDER RAISE CLASSIFIER  —  v4.1
  Angle-based pipeline (6 angles + 6 velocities = 12 channels)
  Triple OOD gate: Recon + Mahalanobis + Motion-range
================================================================
"""

import os, sys, json, time, traceback
from collections import deque

import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from scipy.interpolate import interp1d

# =============================================================
# 🔧 CONFIG
# =============================================================
CFG = {
    # files
    "MODEL_DIR"       : ".",
    "TFLITE_FILE"     : "full_pipeline_v4.tflite",
    "CONFIG_FILE"     : "realtime_config_v4.json",

    # webcam
    "CAMERA_INDEX"    : 1,
    "CAMERA_W"        : 1280,
    "CAMERA_H"        : 720,
    "FLIP_HORIZONTAL" : True,

    # mediapipe
    "MP_MODEL_COMPLEXITY" : 1,
    "MP_MIN_DET_CONF"     : 0.5,
    "MP_MIN_TRK_CONF"     : 0.5,

    # rep state machine (uses mean shoulder abduction)
    "REP_HIGH_DEG"   : 35.0,    # start a rep when angle exceeds this
    "REP_LOW_DEG"    : 25.0,    # end a rep when angle drops below this
    "REP_MIN_FRAMES" : 8,
    "REP_MAX_FRAMES" : 250,
    "REP_COOLDOWN_S" : 0.3,

    # threshold overrides (None -> use config values)
    "OOD_REC_OVERRIDE"    : None,
    "OOD_MH_OVERRIDE"     : None,
    "OOD_MOTION_OVERRIDE" : None,

    # display
    "PREDICTION_HOLD_S"    : 3.0,
    "MIN_CLASS_CONFIDENCE" : 0.50,
    "SHOW_DEBUG_OVERLAY"   : True,
    "LOG_REP_DETAILS"      : True,
    "SAVE_LAST_REP_NPZ"    : True,
}

# =============================================================
# 0) BANNER
# =============================================================
def banner(t):
    print("\n" + "="*64); print(" " + t); print("="*64)

banner("ENVIRONMENT")
print(f"  Python     : {sys.version.split()[0]}")
print(f"  TensorFlow : {tf.__version__}")
print(f"  MediaPipe  : {mp.__version__}")
print(f"  OpenCV     : {cv2.__version__}")
print(f"  NumPy      : {np.__version__}")

# =============================================================
# 1) LOAD CONFIG
# =============================================================
banner("LOAD CONFIG")
cfg_path   = os.path.join(CFG["MODEL_DIR"], CFG["CONFIG_FILE"])
model_path = os.path.join(CFG["MODEL_DIR"], CFG["TFLITE_FILE"])
assert os.path.exists(cfg_path),   f"Config not found: {cfg_path}"
assert os.path.exists(model_path), f"Model not found: {model_path}"

with open(cfg_path) as f:
    rcfg = json.load(f)

VERSION    = rcfg.get("version", "v4.1")
T          = int(rcfg["T"])
INPUT_C    = int(rcfg["input_dim"])               # 12
N_ANG      = int(rcfg["n_angles"])                # 6
N_LM       = int(rcfg["n_landmarks"])             # 33
LATENT     = int(rcfg["latent_dim"])
INV_LABEL  = {int(k): v for k, v in rcfg["inv_label_map"].items()}
LABEL_MAP  = rcfg["label_map"]
RAW_KP_COLS_ORDER = rcfg["raw_kp_columns_order"]

X_MEAN     = np.array(rcfg["x_mean"], dtype=np.float32)
X_STD      = np.array(rcfg["x_std"],  dtype=np.float32)
CLIP_RANGE = float(rcfg["clip_range"])

OOD_REC_THR    = float(CFG["OOD_REC_OVERRIDE"]    or rcfg["ood_recon_threshold"])
OOD_MH_THR     = float(CFG["OOD_MH_OVERRIDE"]     or rcfg["ood_mahalanobis_threshold"])
OOD_MOTION_THR = float(CFG["OOD_MOTION_OVERRIDE"] or rcfg["ood_motion_threshold"])

print(f"  version          : {VERSION}")
print(f"  T                : {T}")
print(f"  input_dim        : {INPUT_C}  (6 angles + 6 velocities)")
print(f"  latent_dim       : {LATENT}")
print(f"  labels           : {INV_LABEL}")
print(f"  OOD recon  thr   : {OOD_REC_THR:.2f}")
print(f"  OOD mahal  thr   : {OOD_MH_THR:.2f}")
print(f"  OOD motion thr   : {OOD_MOTION_THR:.3f}")
print(f"  trained_at       : {rcfg.get('trained_at','?')}")

# =============================================================
# 2) LOAD TFLITE
# =============================================================
banner("LOAD TFLITE MODEL")
itp = tf.lite.Interpreter(model_path=model_path); itp.allocate_tensors()
IN_DET  = itp.get_input_details()
OUT_DET = itp.get_output_details()
KP_IDX  = IN_DET[0]['index']
print(f"  Inputs ({len(IN_DET)}):")
for d in IN_DET:
    print(f"    {d['name']:55s} idx={d['index']:3d} shape={d['shape']} dtype={d['dtype'].__name__}")
print(f"  Outputs ({len(OUT_DET)}):")
for d in OUT_DET:
    print(f"    {d['name']:55s} idx={d['index']:3d} shape={d['shape']} dtype={d['dtype'].__name__}")
assert tuple(IN_DET[0]['shape']) == (1, T, INPUT_C), \
    f"Model expects shape (1,{T},{INPUT_C}) but got {tuple(IN_DET[0]['shape'])}"

# =============================================================
# 3) HELPERS — must mirror training math exactly
# =============================================================
def landmarks_to_vec132(landmarks):
    out = np.empty(132, dtype=np.float32)
    for i, lm in enumerate(landmarks):
        out[i*4 + 0] = lm.x
        out[i*4 + 1] = lm.y
        out[i*4 + 2] = lm.z
        out[i*4 + 3] = lm.visibility
    return out

def resample_T(arr, T_):
    n = arr.shape[0]
    if n == T_: return arr.astype(np.float32)
    if n < 2:   return np.repeat(arr, T_, axis=0).astype(np.float32)
    f = interp1d(np.linspace(0,1,n), arr, axis=0, kind='linear')
    return f(np.linspace(0,1,T_)).astype(np.float32)

def _angle_3pt(a, b, c):
    """Angle at vertex b. Works on (T,3), (T,2), or (3,)/(2,) inputs."""
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32); c = np.asarray(c, np.float32)
    ba = a - b; bc = c - b
    if ba.ndim == 1:
        cos = (ba @ bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    nba = np.linalg.norm(ba, axis=-1, keepdims=True) + 1e-8
    nbc = np.linalg.norm(bc, axis=-1, keepdims=True) + 1e-8
    cos = np.sum((ba/nba) * (bc/nbc), axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def _signed_torso_lean(sho_mid, hip_mid):
    v = sho_mid - hip_mid
    return np.degrees(np.arctan2(v[:, 0], -v[:, 1]))

def compute_angles(kp_raw):
    """(T,132) raw mediapipe -> (T,6) joint angles in degrees."""
    T_ = kp_raw.shape[0]
    kp = kp_raw.reshape(T_, N_LM, 4)
    p3 = kp[:, :, :3].astype(np.float32)
    p2 = kp[:, :, :2].astype(np.float32)
    R_HIP, L_HIP = 24, 23
    R_SHO, L_SHO = 12, 11
    R_ELB, L_ELB = 14, 13
    R_WRI, L_WRI = 16, 15
    R_KNE, L_KNE = 26, 25
    R_sho_abd  = _angle_3pt(p3[:, R_HIP], p3[:, R_SHO], p3[:, R_ELB])
    L_sho_abd  = _angle_3pt(p3[:, L_HIP], p3[:, L_SHO], p3[:, L_ELB])
    R_elb_flex = _angle_3pt(p3[:, R_SHO], p3[:, R_ELB], p3[:, R_WRI])
    L_elb_flex = _angle_3pt(p3[:, L_SHO], p3[:, L_ELB], p3[:, L_WRI])
    sho_mid_2d = 0.5 * (p2[:, R_SHO] + p2[:, L_SHO])
    hip_mid_2d = 0.5 * (p2[:, R_HIP] + p2[:, L_HIP])
    torso_lean = _signed_torso_lean(sho_mid_2d, hip_mid_2d)
    sho_mid_3d = 0.5 * (p3[:, R_SHO] + p3[:, L_SHO])
    hip_mid_3d = 0.5 * (p3[:, R_HIP] + p3[:, L_HIP])
    kne_mid_3d = 0.5 * (p3[:, R_KNE] + p3[:, L_KNE])
    hip_flex   = _angle_3pt(sho_mid_3d, hip_mid_3d, kne_mid_3d)
    return np.stack([R_sho_abd, L_sho_abd, R_elb_flex, L_elb_flex,
                     torso_lean, hip_flex], axis=-1).astype(np.float32)

def add_angle_velocity(ang):
    """(T,6) -> (T,12) [angles | velocities] (centered finite-diff)."""
    T_, C = ang.shape
    vel = np.zeros_like(ang)
    if T_ >= 3:
        vel[1:-1] = 0.5 * (ang[2:] - ang[:-2])
        vel[0]    = ang[1] - ang[0]
        vel[-1]   = ang[-1] - ang[-2]
    return np.concatenate([ang, vel], axis=-1).astype(np.float32)

def standardize_and_clip(x):
    return np.clip((x - X_MEAN) / X_STD, -CLIP_RANGE, CLIP_RANGE).astype(np.float32)

# =============================================================
# 4) IDENTIFY TFLITE OUTPUTS  (probs / recon / mahal / motion / ood)
# =============================================================
def fake_perfect_angles(T_):
    t = np.linspace(0, np.pi, T_)
    a = np.zeros((T_,6), dtype=np.float32); arc = 90*np.sin(t)
    a[:,0] = 10 + arc; a[:,1] = 10 + arc
    a[:,2] = 170; a[:,3] = 170; a[:,4] = 0; a[:,5] = 175
    return a

def identify_outputs():
    """Map each output index to its semantic role using a known synthetic sample."""
    ang  = fake_perfect_angles(T)
    full = add_angle_velocity(ang)
    xn   = standardize_and_clip(full)[None]
    itp.set_tensor(KP_IDX, xn); itp.invoke()

    # First pass: try name-based mapping
    role = {}
    for d in OUT_DET:
        n = d['name'].lower()
        if 'probs' in n or 'cls' in n or 'softmax' in n or d['shape'][-1] == 3:
            role['probs'] = d['index']
        elif 'recon' in n:
            role['recon'] = d['index']
        elif 'mahal' in n:
            role['mahal'] = d['index']
        elif 'motion' in n:
            role['motion'] = d['index']
        elif 'ood' in n or 'flag' in n:
            role['ood']   = d['index']

    # Second pass: value-based fallback if name matching incomplete
    if len(role) < 5:
        # Collect candidates that are NOT probs (i.e. single-valued outputs)
        cands = []
        for d in OUT_DET:
            if d['index'] == role.get('probs'): continue
            v = float(itp.get_tensor(d['index']).flatten()[0])
            cands.append((d['index'], v))
        # ood: value is exactly 0 or 1
        ood_c = [c for c in cands if c[1] in (0.0, 1.0)]
        if len(ood_c) == 1 and 'ood' not in role:
            role['ood'] = ood_c[0][0]
            cands = [c for c in cands if c[0] != ood_c[0][0]]
        # remaining 3: largest = recon, middle = mahal, smallest = motion
        cands.sort(key=lambda c: c[1], reverse=True)
        if len(cands) == 3:
            if 'recon'  not in role: role['recon']  = cands[0][0]
            if 'mahal'  not in role: role['mahal']  = cands[1][0]
            if 'motion' not in role: role['motion'] = cands[2][0]

    missing = [k for k in ('probs','recon','mahal','motion','ood') if k not in role]
    if missing:
        raise RuntimeError(f"Could not identify outputs: missing {missing}. "
                           f"Output names found: {[d['name'] for d in OUT_DET]}")
    return role

ROLE = identify_outputs()
print("\n  Identified output mapping:")
for r, idx in ROLE.items():
    print(f"    {r:<7} -> tensor index {idx}")

# =============================================================
# 5) PREDICT-REP FUNCTION
# =============================================================
def predict_rep(kp_frames):
    """kp_frames: list of raw (132,) vectors. Returns dict."""
    t0 = time.time()
    kp_raw = np.stack(kp_frames)                    # (N,132)
    kp_T   = resample_T(kp_raw, T)                  # (T,132)
    ang    = compute_angles(kp_T)                   # (T,6)
    full   = add_angle_velocity(ang)                # (T,12)
    x      = standardize_and_clip(full)[None]       # (1,T,12)

    itp.set_tensor(KP_IDX, x); itp.invoke()
    probs  = itp.get_tensor(ROLE['probs'])[0]
    recon  = float(itp.get_tensor(ROLE['recon']).flatten()[0])
    mahal  = float(itp.get_tensor(ROLE['mahal']).flatten()[0])
    motion = float(itp.get_tensor(ROLE['motion']).flatten()[0])
    ood    = bool(itp.get_tensor(ROLE['ood']).flatten()[0] > 0.5)

    # Why-OOD attribution (for HUD)
    why = []
    if recon  > OOD_REC_THR:    why.append("recon")
    if mahal  > OOD_MH_THR:     why.append("mahal")
    if motion < OOD_MOTION_THR: why.append("motion")
    why_str = "+".join(why) if why else ""

    pred_idx = int(np.argmax(probs))
    conf     = float(probs[pred_idx])
    if ood:
        cls = "NOT_WORKOUT"
    elif conf < CFG["MIN_CLASS_CONFIDENCE"]:
        cls = "UNCERTAIN"
    else:
        cls = INV_LABEL[pred_idx]

    peak_R = float(ang[:,0].max())
    peak_L = float(ang[:,1].max())

    return {
        "n_raw_frames": int(kp_raw.shape[0]),
        "ood":          ood,
        "ood_why":      why_str,
        "recon":        recon,
        "mahal":        mahal,
        "motion":       motion,
        "class":        cls,
        "raw_pred":     INV_LABEL[pred_idx],
        "confidence":   conf,
        "probs":        {INV_LABEL[i]: float(probs[i]) for i in range(3)},
        "peak_R_abd":   peak_R,
        "peak_L_abd":   peak_L,
        "infer_ms":     (time.time()-t0)*1000.0,
        "kp_raw":       kp_raw,
        "angles":       ang,
        "x_norm":       x[0],
    }

# =============================================================
# 6) SELF-TEST
# =============================================================
banner("MODEL SELF-TEST  (synthetic perfect rep)")
try:
    fake_kp = []
    for tt in range(T):
        v = np.zeros(132, dtype=np.float32)
        v[24*4:24*4+3] = [0.55, 0.55, 0.0]
        v[23*4:23*4+3] = [0.45, 0.55, 0.0]
        v[12*4:12*4+3] = [0.55, 0.35, 0.0]
        v[11*4:11*4+3] = [0.45, 0.35, 0.0]
        s = np.sin(np.pi * tt / max(1, T-1))
        v[14*4:14*4+3] = [0.55 + 0.20*s, 0.40 - 0.05*s, 0.0]
        v[13*4:13*4+3] = [0.45 - 0.20*s, 0.40 - 0.05*s, 0.0]
        v[16*4:16*4+3] = [0.55 + 0.30*s, 0.45 - 0.10*s, 0.0]
        v[15*4:15*4+3] = [0.45 - 0.30*s, 0.45 - 0.10*s, 0.0]
        v[26*4:26*4+3] = [0.55, 0.85, 0.0]
        v[25*4:25*4+3] = [0.45, 0.85, 0.0]
        for i in range(33): v[i*4+3] = 1.0
        fake_kp.append(v)
    out = predict_rep(fake_kp)
    print(f"  ✓ pipeline runs  ({out['infer_ms']:.2f} ms)")
    print(f"  ✓ recon  = {out['recon']:.2f}    (thr {OOD_REC_THR:.2f})")
    print(f"  ✓ mahal  = {out['mahal']:.2f}    (thr {OOD_MH_THR:.2f})")
    print(f"  ✓ motion = {out['motion']:.3f}   (thr {OOD_MOTION_THR:.3f})")
    print(f"  ✓ ood    = {out['ood']}     why={out['ood_why'] or '-'}")
    print(f"  ✓ probs  = {out['probs']}")
    print(f"  ✓ peak   R={out['peak_R_abd']:.1f}°  L={out['peak_L_abd']:.1f}°")
except Exception as e:
    print(f"  ❌ self-test failed: {e}"); traceback.print_exc(); sys.exit(1)

# =============================================================
# 7) OPEN WEBCAM
# =============================================================
banner("OPEN WEBCAM")
cap = cv2.VideoCapture(CFG["CAMERA_INDEX"], cv2.CAP_DSHOW if os.name == "nt" else 0)
if not cap.isOpened():
    print(f"  ❌ Could not open camera index {CFG['CAMERA_INDEX']}"); sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG["CAMERA_W"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG["CAMERA_H"])
got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"  ✓ camera {CFG['CAMERA_INDEX']} opened {got_w}x{got_h}")

# =============================================================
# 8) MEDIAPIPE
# =============================================================
banner("MEDIAPIPE POSE")
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles
pose = mp_pose.Pose(
    model_complexity=CFG["MP_MODEL_COMPLEXITY"],
    min_detection_confidence=CFG["MP_MIN_DET_CONF"],
    min_tracking_confidence=CFG["MP_MIN_TRK_CONF"],
    enable_segmentation=False,
)
print(f"  ✓ MediaPipe Pose ready  (complexity={CFG['MP_MODEL_COMPLEXITY']})")

# =============================================================
# 9) MAIN LOOP
# =============================================================
banner("RUNNING — press Q to quit, R to reset rep counter")

STATE = 0  # 0=WAITING, 1=ACTIVE
buffer_kp     = []
rep_count     = 0
last_rep_t    = 0.0
last_result   = None
last_result_t = 0.0
fps_hist      = deque(maxlen=30)
prev_t        = time.time()

session_tally = {"Perfect":0,"Over_Range":0,"Lower_Range":0,
                 "NOT_WORKOUT":0,"UNCERTAIN":0}

def angles_now(landmarks):
    """Compute mean shoulder abduction angle (used for state machine)."""
    R_hip = (landmarks[24].x, landmarks[24].y, landmarks[24].z)
    R_sh  = (landmarks[12].x, landmarks[12].y, landmarks[12].z)
    R_el  = (landmarks[14].x, landmarks[14].y, landmarks[14].z)
    L_hip = (landmarks[23].x, landmarks[23].y, landmarks[23].z)
    L_sh  = (landmarks[11].x, landmarks[11].y, landmarks[11].z)
    L_el  = (landmarks[13].x, landmarks[13].y, landmarks[13].z)
    R = _angle_3pt(R_hip, R_sh, R_el)
    L = _angle_3pt(L_hip, L_sh, L_el)
    return R, L, (R + L) * 0.5

while True:
    ok, frame = cap.read()
    if not ok:
        print("  ⚠️  failed to grab frame"); break
    if CFG["FLIP_HORIZONTAL"]:
        frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = pose.process(rgb)
    rgb.flags.writeable = True

    cur_R = cur_L = cur_M = None
    have_pose = res.pose_landmarks is not None

    if have_pose:
        lms = res.pose_landmarks.landmark
        cur_R, cur_L, cur_M = angles_now(lms)
        kp_vec = landmarks_to_vec132(lms)

        # ---- rep state machine ----
        if STATE == 0 and cur_M > CFG["REP_HIGH_DEG"] and (time.time() - last_rep_t) > CFG["REP_COOLDOWN_S"]:
            STATE = 1
            buffer_kp = [kp_vec]
            if CFG["LOG_REP_DETAILS"]:
                print(f"\n[REP] start  angle={cur_M:.1f}°")
        elif STATE == 1:
            buffer_kp.append(kp_vec)
            if cur_M < CFG["REP_LOW_DEG"] or len(buffer_kp) >= CFG["REP_MAX_FRAMES"]:
                if len(buffer_kp) >= CFG["REP_MIN_FRAMES"]:
                    rep_count += 1
                    last_rep_t = time.time()
                    try:
                        last_result   = predict_rep(buffer_kp)
                        last_result_t = time.time()
                        r = last_result
                        session_tally[r['class']] = session_tally.get(r['class'], 0) + 1
                        ood_str = "YES" if r['ood'] else "no"
                        print(f"[REP #{rep_count}] frames={r['n_raw_frames']:3d}  "
                              f"peak R={r['peak_R_abd']:.1f}° L={r['peak_L_abd']:.1f}°")
                        print(f"   recon={r['recon']:7.2f} (thr {OOD_REC_THR:.0f})  "
                              f"mahal={r['mahal']:6.2f} (thr {OOD_MH_THR:.1f})  "
                              f"motion={r['motion']:6.3f} (thr {OOD_MOTION_THR:.3f})  "
                              f"ood={ood_str}{(' ['+r['ood_why']+']') if r['ood_why'] else ''}")
                        print(f"   raw_pred={r['raw_pred']} conf={r['confidence']:.2f}  "
                              f"-> {r['class']}  ({r['infer_ms']:.1f} ms)")
                        for k, v in r['probs'].items():
                            print(f"      {k:<12}: {v:.3f}")
                        if CFG["SAVE_LAST_REP_NPZ"]:
                            np.savez(os.path.join(CFG["MODEL_DIR"], "last_rep.npz"),
                                     kp_raw=r['kp_raw'],
                                     angles=r['angles'],
                                     x_norm=r['x_norm'],
                                     probs=np.array(list(r['probs'].values())),
                                     recon=r['recon'],
                                     mahal=r['mahal'],
                                     motion=r['motion'],
                                     ood=int(r['ood']))
                    except Exception as e:
                        print(f"  ❌ predict_rep failed: {e}"); traceback.print_exc()
                else:
                    if CFG["LOG_REP_DETAILS"]:
                        print(f"[REP] dropped (only {len(buffer_kp)} frames)")
                STATE = 0; buffer_kp = []

        mp_drawing.draw_landmarks(
            frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

    # ---- FPS ----
    now = time.time(); dt = now - prev_t; prev_t = now
    fps_hist.append(1.0 / max(dt, 1e-6))
    fps = float(np.mean(fps_hist))

    # ---- HUD ----
    if CFG["SHOW_DEBUG_OVERLAY"]:
        y = [25]
        def put(s, color=(255,255,255)):
            cv2.putText(frame, s, (10, y[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y[0] += 25

        put(f"FPS: {fps:5.1f}  cam={CFG['CAMERA_INDEX']}  T={T}  v={VERSION}")
        put(f"Pose: {'OK' if have_pose else 'NOT FOUND'}",
            (0,255,0) if have_pose else (0,0,255))
        if cur_M is not None:
            put(f"Abd  R={cur_R:5.1f}  L={cur_L:5.1f}  Mean={cur_M:5.1f}")
        put(f"State: {'ACTIVE' if STATE==1 else 'waiting'}  buf={len(buffer_kp):3d}",
            (0,200,255) if STATE==1 else (200,200,200))
        put(f"Reps: {rep_count}  P={session_tally['Perfect']}"  
            f"O={session_tally['Over_Range']}  L={session_tally['Lower_Range']}  "
            f"OOD={session_tally['NOT_WORKOUT']}  ?={session_tally['UNCERTAIN']}")

        if last_result and (time.time() - last_result_t) < CFG["PREDICTION_HOLD_S"]:
            r = last_result
            # Color by class
            if r['ood']:
                color = (0, 0, 255)            # red
            elif r['class'] == 'Perfect':
                color = (0, 255, 0)            # green
            elif r['class'] == 'UNCERTAIN':
                color = (200, 200, 0)          # yellow
            else:
                color = (0, 165, 255)          # orange (range error)

            put(f">> {r['class']}  conf={r['confidence']:.2f}", color)
            put(f"   raw={r['raw_pred']}  peak R={r['peak_R_abd']:.0f} L={r['peak_L_abd']:.0f}", color)
            put(f"   rec={r['recon']:.0f}/{OOD_REC_THR:.0f}  "
                f"mh={r['mahal']:.1f}/{OOD_MH_THR:.1f}  "
                f"mot={r['motion']:.2f}/{OOD_MOTION_THR:.2f}", color)
            if r['ood'] and r['ood_why']:
                put(f"   OOD reason: {r['ood_why']}", (0, 0, 255))

    cv2.imshow(f"Side Shoulder Raise — Real-Time {VERSION}", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n[QUIT] user pressed Q")
        break
    if key == ord('r'):
        rep_count = 0; STATE = 0; buffer_kp = []
        for k in session_tally: session_tally[k] = 0
        print("[RESET] rep counter cleared")
    if key == ord('s') and last_result is not None:
        # Manual save with timestamped name
        ts   = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CFG["MODEL_DIR"], f"capture_{ts}.npz")
        r = last_result
        np.savez(path,
                 kp_raw=r['kp_raw'],
                 angles=r['angles'],
                 x_norm=r['x_norm'],
                 probs=np.array(list(r['probs'].values())),
                 recon=r['recon'], mahal=r['mahal'], motion=r['motion'],
                 ood=int(r['ood']),
                 raw_pred=r['raw_pred'], cls=r['class'])
        print(f"[SAVE] {path}")

cap.release()
cv2.destroyAllWindows()
pose.close()

banner("BYE")
print(f"  total reps   : {rep_count}")
print(f"  session tally: {session_tally}")