# diagnose_model.py
#
# Run this BEFORE retraining to understand exactly what is wrong.
# It checks:
#   1. Raw model outputs on your actual training data
#   2. Whether features look different between classes (are they separable?)
#   3. Whether the label_classes.npy order matches what you expect
#   4. Min knee angle distribution per class
#
# Usage:
#   python diagnose_model.py
#
# Requirements: same as training script

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ── CONFIG — must match your training script exactly ──────────────────────
CSV_PATH      = "ds.csv"
MODEL_PATH    = "best_squat_model.keras"      # use the .keras, most accurate
TFLITE_PATH   = "squat_classifier_float32.tflite"
TARGET_FRAMES = 64
SMOOTH_ALPHA  = 0.6
MIN_REP_LEN   = 15
JOINTS = {
    "nose": 0, "l_shoulder": 11, "r_shoulder": 12,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
}
JOINT_IDX = list(JOINTS.values())
# ─────────────────────────────────────────────────────────────────────────

# ── Feature pipeline (must match training exactly) ────────────────────────
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

def torso_normalize(lm):
    mid_hip = (lm[23] + lm[24]) / 2.0
    mid_sho = (lm[11] + lm[12]) / 2.0
    scale   = np.linalg.norm(mid_sho - mid_hip)
    return (lm - mid_hip) / max(scale, 1e-6)

def resample(seq, target=TARGET_FRAMES):
    T = len(seq)
    if T == target: return seq.astype(np.float32)
    idx = np.linspace(0, T - 1, target)
    out = np.zeros((target, seq.shape[1]), dtype=np.float32)
    for f in range(seq.shape[1]):
        out[:, f] = np.interp(idx, np.arange(T), seq[:, f])
    return out

def process_rep_dual(frames):
    T = len(frames)
    frames_sm = ema_smooth(frames, SMOOTH_ALPHA)
    norm = np.array([torso_normalize(f) for f in frames_sm], dtype=np.float32)
    knee_r = angle_batch(norm[:, 24], norm[:, 26], norm[:, 28])
    knee_l = angle_batch(norm[:, 23], norm[:, 25], norm[:, 27])
    knee_ang = (knee_r + knee_l) / 2.0
    hip_y_raw = (frames_sm[:, 23, 1] + frames_sm[:, 24, 1]) / 2.0
    hip_drop = float(np.max(hip_y_raw) - np.min(hip_y_raw))
    min_hip_y = float(np.max(hip_y_raw))
    mid_sho = (norm[:, 11] + norm[:, 12]) / 2.0
    mid_hip2 = (norm[:, 23] + norm[:, 24]) / 2.0
    mid_kne = (norm[:, 25] + norm[:, 26]) / 2.0
    spine_ang = angle_batch(mid_sho, mid_hip2, mid_kne)
    min_spine = float(np.min(spine_ang))
    mean_spine = float(np.mean(spine_ang))
    min_knee = float(np.min(knee_ang))
    max_knee = float(np.max(knee_ang))
    knee_rom = max_knee - min_knee
    norm_min = float(np.clip((min_knee - 70.0) / (170.0 - 70.0), 0.0, 1.0))
    torso_v = (norm[:, 11] + norm[:, 12]) / 2.0 - (norm[:, 23] + norm[:, 24]) / 2.0
    vert    = np.array([[0.0, -1.0, 0.0]])
    dv      = np.sum(torso_v * vert, axis=-1)
    lean_a  = np.degrees(np.arccos(np.clip(dv / (np.linalg.norm(torso_v, axis=-1) + 1e-6), -1.0, 1.0)))
    max_lean  = float(np.max(lean_a));  mean_lean = float(np.mean(lean_a))
    sho_fwd   = float(np.mean(((norm[:, 11] + norm[:, 12]) / 2.0)[:, 0] - ((norm[:, 23] + norm[:, 24]) / 2.0)[:, 0]))
    depth_feat = np.array(
        [min_knee, knee_rom, hip_drop, norm_min, min_hip_y, max_knee,
         min_spine, mean_spine, max_lean, mean_lean, sho_fwd], dtype=np.float32)
    pos = norm[:, JOINT_IDX, :].reshape(T, -1)
    vel = np.concatenate([np.zeros_like(pos[:1]), np.diff(pos, axis=0)], axis=0)
    acc = np.concatenate([np.zeros_like(vel[:1]), np.diff(vel, axis=0)], axis=0)
    seq_feat = np.concatenate([pos, vel, acc, knee_ang], axis=1).astype(np.float32)
    return resample(seq_feat, TARGET_FRAMES), depth_feat

def process_rep_tiled(frames):
    """Old single-branch pipeline with depth tiled into sequence."""
    T = len(frames)
    frames_sm = ema_smooth(frames, SMOOTH_ALPHA)
    norm = np.array([torso_normalize(f) for f in frames_sm], dtype=np.float32)
    knee_r = angle_batch(norm[:, 24], norm[:, 26], norm[:, 28])
    knee_l = angle_batch(norm[:, 23], norm[:, 25], norm[:, 27])
    knee_ang = (knee_r + knee_l) / 2.0
    hip_y_raw = (frames_sm[:, 23, 1] + frames_sm[:, 24, 1]) / 2.0
    hip_drop = float(np.max(hip_y_raw) - np.min(hip_y_raw))
    min_hip_y = float(np.max(hip_y_raw))
    min_knee = float(np.min(knee_ang))
    max_knee = float(np.max(knee_ang))
    knee_rom = max_knee - min_knee
    norm_min = float(np.clip((min_knee - 70.0) / (170.0 - 70.0), 0.0, 1.0))
    depth_feat = np.array(
        [min_knee, knee_rom, hip_drop, norm_min, min_hip_y, max_knee],
        dtype=np.float32)
    pos = norm[:, JOINT_IDX, :].reshape(T, -1)
    vel = np.concatenate([np.zeros_like(pos[:1]), np.diff(pos, axis=0)], axis=0)
    acc = np.concatenate([np.zeros_like(vel[:1]), np.diff(vel, axis=0)], axis=0)
    d3 = np.tile(np.array([min_knee, knee_rom, hip_drop], dtype=np.float32), (T, 1))
    seq_feat = np.concatenate([pos, vel, acc, knee_ang, d3], axis=1).astype(np.float32)
    return resample(seq_feat, TARGET_FRAMES), depth_feat

# ── Load data ─────────────────────────────────────────────────────────────
print("="*60)
print("STEP 1: Loading data and checking label order")
print("="*60)

try:
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
except TypeError:
    df = pd.read_csv(CSV_PATH, error_bad_lines=False)
if df.shape[1] > 101:
    df = df.iloc[:, :101]

lm_cols = [f"{c}{i}" for i in range(33) for c in ["x", "y", "z"]]
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])

print(f"\nLabelEncoder classes (THIS IS WHAT THE MODEL LEARNED):")
for i, c in enumerate(le.classes_):
    count = int(df[df.label_id == i].rep_number.nunique())
    print(f"  index {i} = '{c}'  ({count} reps)")

# Check saved label_classes.npy
try:
    saved = np.load("label_classes.npy", allow_pickle=True)
    print(f"\nlabel_classes.npy order: {list(saved)}")
    match = all(a == b for a, b in zip(saved, le.classes_))
    print(f"  Matches LabelEncoder: {'YES ✓' if match else 'NO ✗ — THIS IS A BUG'}")
except FileNotFoundError:
    print("\nlabel_classes.npy not found — will be created by training script")

# ── Process reps ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Feature separability check (most important)")
print("="*60)

seq_list, depth_list, y_list, meta = [], [], [], []
for rep in tqdm(df.rep_number.unique(), desc="Processing"):
    g = df[df.rep_number == rep]
    if len(g) < MIN_REP_LEN:
        continue
    lm = g[lm_cols].values.reshape(-1, 33, 3).astype(np.float32)
    seq_d, dep_d = process_rep_dual(lm)
    seq_t, dep_t = process_rep_tiled(lm)
    seq_list.append((seq_d, seq_t))
    depth_list.append(dep_d)
    y_list.append(int(g.label_id.iloc[0]))
    meta.append({
        "rep": rep,
        "label": g.label.iloc[0],
        "min_knee": float(dep_d[0]),
        "knee_rom": float(dep_d[1]),
        "hip_drop": float(dep_d[2]),
        "min_spine": float(dep_d[6]),
        "max_lean":  float(dep_d[8]) if len(dep_d) > 8 else 0.0,
        "sho_fwd":   float(dep_d[10]) if len(dep_d) > 10 else 0.0,
    })

y = np.array(y_list)
X_seq_dual   = np.array([s[0] for s in seq_list], dtype=np.float32)
X_seq_tiled  = np.array([s[1] for s in seq_list], dtype=np.float32)
X_depth      = np.array(depth_list, dtype=np.float32)
meta_df      = pd.DataFrame(meta)

print(f"\nFeature dimensions:")
print(f"  Dual-branch  seq: {X_seq_dual.shape}   depth: {X_depth.shape}")
print(f"  Tiled-branch seq: {X_seq_tiled.shape}")

print("\n--- Per-class feature statistics ---")
print(f"{'Label':<14} {'count':>5}  {'min_knee':>9}  {'knee_rom':>9}  "
      f"{'hip_drop':>9}  {'min_spine':>10}")
print("-" * 65)
for cls_name in le.classes_:
    sub = meta_df[meta_df.label == cls_name]
    if len(sub) == 0:
        print(f"  {cls_name:<12}  NO REPS FOUND")
        continue
    print(f"  {cls_name:<12}  {len(sub):>5}  "
          f"{sub.min_knee.mean():>8.1f}°  "
          f"{sub.knee_rom.mean():>8.1f}°  "
          f"{sub.hip_drop.mean():>8.3f}   "
          f"{sub.min_spine.mean():>9.1f}°")

print()
print(">>> KEY: If min_knee is similar across classes, the model CANNOT tell them apart.")
print(">>> Perfect squat should have lower min_knee (~90°) than Shallow (~110-130°).")
print(">>> BackRounding should have lower min_spine than Perfect.")

# ── Simple threshold baseline ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Threshold baseline (what a simple rule would get)")
print("="*60)
# Two-class threshold: shallow if min_knee > 110
thresh = 110.0
baseline_preds = []
for row in meta:
    # Case-insensitive — find actual class names from encoder
    cls_map = {c.lower(): c for c in le.classes_}
    back_n    = cls_map.get("backrounding", le.classes_[0])
    perf_n    = cls_map.get("perfect",      le.classes_[1])
    shal_n    = cls_map.get("shallow",      le.classes_[2])
    row_label = row["label"].lower()
    if row["max_lean"] if "max_lean" in row else 0 > 0:  # dummy init
        pass

baseline_preds = []
for row in meta:
    cls_map = {c.lower(): c for c in le.classes_}
    back_n  = cls_map.get("backrounding", le.classes_[0])
    perf_n  = cls_map.get("perfect",      le.classes_[1])
    shal_n  = cls_map.get("shallow",      le.classes_[2])
    lean = row.get("max_lean", 0)
    if lean > thresh:
        baseline_preds.append(back_n)
    elif row["min_knee"] > thresh:
        baseline_preds.append(shal_n)
    else:
        baseline_preds.append(perf_n)

baseline_true = [m["label"] for m in meta]
print(f"\nSimple threshold rule (min_knee>{thresh}→shallow, min_spine<155→backrounding):")
print(classification_report(baseline_true, baseline_preds,
                             labels=list(le.classes_), target_names=list(le.classes_)))
print(">>> If baseline accuracy < 70%, your DATA has the problem, not the model.")
print(">>> If baseline accuracy > 80%, your MODEL is the problem.")

# ── Run model on full dataset ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Model predictions on FULL dataset")
print("="*60)

# Try TFLite first (works without full TF)
try:
    try:
        import tflite_runtime.interpreter as tflite
        Interp = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interp = tf.lite.Interpreter

    interp = Interp(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()
    out_d = interp.get_output_details()
    num_inputs = len(inp_d)
    expected_dim = int(inp_d[0]["shape"][-1])
    print(f"\nTFLite model: {num_inputs} input(s), seq_dim={expected_dim}")
    print(f"Output classes: {int(out_d[0]['shape'][-1])}")

    all_probs = []
    for i in range(len(X_seq_dual)):
        if num_inputs == 1:
            interp.set_tensor(inp_d[0]["index"],
                              X_seq_tiled[i:i+1].astype(np.float32))
        else:
            for d in inp_d:
                if len(d["shape"]) == 3:
                    interp.set_tensor(d["index"], X_seq_dual[i:i+1].astype(np.float32))
                else:
                    interp.set_tensor(d["index"], X_depth[i:i+1].astype(np.float32))
        interp.invoke()
        all_probs.append(interp.get_tensor(out_d[0]["index"])[0].copy())

    all_probs = np.array(all_probs)
    preds = np.argmax(all_probs, axis=1)

    print("\nClassification report (TFLite on full dataset):")
    print(classification_report(y, preds, target_names=le.classes_))
    print("Confusion matrix:")
    cm = confusion_matrix(y, preds)
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

    # Show per-class average confidence
    print("\nAverage confidence per predicted class:")
    for ci, cn in enumerate(le.classes_):
        mask = preds == ci
        if mask.sum() > 0:
            avg_conf = all_probs[mask, ci].mean()
            print(f"  predicted '{cn}': {mask.sum()} times, avg conf={avg_conf*100:.1f}%")

    # Show a few sample predictions
    print("\nSample predictions (first 10 reps):")
    print(f"{'True':<14}  {'Predicted':<14}  {'Conf':>5}  {'Probs'}")
    print("-" * 65)
    for i in range(min(10, len(y))):
        true_name = le.classes_[y[i]]
        pred_name = le.classes_[preds[i]]
        conf      = all_probs[i, preds[i]]
        prob_str  = "  ".join([f"{p*100:.0f}%" for p in all_probs[i]])
        flag      = "" if true_name == pred_name else "  ← WRONG"
        print(f"  {true_name:<12}  {pred_name:<12}  {conf*100:>4.0f}%  [{prob_str}]{flag}")

except Exception as e:
    print(f"TFLite test failed: {e}")

# ── Final diagnosis ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Diagnosis summary")
print("="*60)

print("""
Check the output above and find your situation:

A) Confusion matrix shows only one class predicted
   → Model collapsed. Retrain with higher FOCAL_GAMMA (try 3.0)
     and SHALLOW_WEIGHT_MULT / backrounding weight (try 10.0).

B) Classification report looks good (>70% each class) but realtime fails
   → Feature pipeline mismatch between training and realtime script.
     The feature dim in realtime doesn't match what the model was trained on.

C) Baseline threshold accuracy < 70%
   → Your DATA has overlap — shallow and perfect reps have similar knee angles.
     Check your labelling. Maybe some "shallow" reps actually go deep enough.

D) min_knee values are the same across classes in Step 2
   → Data collection problem. Shallow reps must have higher min_knee than Perfect.
     Re-record shallow examples with noticeably less depth.
""")