import numpy as np
import tensorflow.lite as tflite
from core.feedback_engine import FeedbackType
from .base_engine import BaseWorkoutEngine, FrameResult

def angle_batch(a, b, c):
    ba  = a - b;  bc = c - b
    dot = np.sum(ba * bc, axis=-1)
    n   = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-6
    return np.degrees(np.arccos(np.clip(dot / n, -1.0, 1.0)))[:, None]

def ema_smooth(seq, alpha):
    if len(seq) == 0: return seq
    out = [seq[0]]
    for i in range(1, len(seq)):
        out.append(alpha * seq[i] + (1.0 - alpha) * out[-1])
    return np.array(out, dtype=np.float32)

def torso_normalize(lm):
    mid_hip = (lm[23] + lm[24]) / 2.0
    mid_sho = (lm[11] + lm[12]) / 2.0
    scale   = np.linalg.norm(mid_sho - mid_hip)
    return (lm - mid_hip) / max(scale, 1e-6)

def resample(seq, target=64):
    T = len(seq)
    if T == target: return seq.astype(np.float32)
    idx = np.linspace(0, T - 1, target)
    out = np.zeros((target, seq.shape[1]), dtype=np.float32)
    for f in range(seq.shape[1]):
        out[:, f] = np.interp(idx, np.arange(T), seq[:, f])
    return out

JOINT_IDX = [0, 11, 12, 23, 24, 25, 26, 27, 28]

def process_rep(frames, tiled_depth=False):
    T = len(frames)
    frames_sm = ema_smooth(frames, 0.6)
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

    torso_vec = mid_sho - mid_hip2
    vertical = np.array([[0.0, -1.0, 0.0]])
    dot_v = np.sum(torso_vec * vertical, axis=-1)
    lean_ang = np.degrees(np.arccos(np.clip(dot_v / (np.linalg.norm(torso_vec, axis=-1) + 1e-6), -1.0, 1.0)))
    
    max_lean = float(np.max(lean_ang))
    mean_lean = float(np.mean(lean_ang))
    sho_fwd = float(np.mean(mid_sho[:, 0] - mid_hip2[:, 0]))

    spine_ang = angle_batch(mid_sho, mid_hip2, mid_kne)
    min_spine = float(np.min(spine_ang))
    mean_spine = float(np.mean(spine_ang))

    min_knee = float(np.min(knee_ang))
    max_knee = float(np.max(knee_ang))
    knee_rom = max_knee - min_knee
    norm_min = float(np.clip((min_knee - 70.0) / (170.0 - 70.0), 0.0, 1.0))

    depth_feat = np.array([min_knee, knee_rom, hip_drop, norm_min, min_hip_y, max_knee,
                           min_spine, mean_spine, max_lean, mean_lean, sho_fwd], dtype=np.float32)

    pos = norm[:, JOINT_IDX, :].reshape(T, -1)
    vel = np.concatenate([np.zeros_like(pos[:1]), np.diff(pos, axis=0)], axis=0)
    acc = np.concatenate([np.zeros_like(vel[:1]), np.diff(vel, axis=0)], axis=0)

    if tiled_depth:
        d3 = np.tile(np.array([min_knee, knee_rom, hip_drop], dtype=np.float32), (T, 1))
        seq_feat = np.concatenate([pos, vel, acc, knee_ang, d3], axis=1).astype(np.float32)
    else:
        seq_feat = np.concatenate([pos, vel, acc, knee_ang], axis=1).astype(np.float32)

    seq_feat = resample(seq_feat, 64)
    return seq_feat, depth_feat

def single_frame_knee_angle(lm33):
    a = np.array([[lm33[24][0], lm33[24][1], lm33[24][2]],
                  [lm33[23][0], lm33[23][1], lm33[23][2]]])
    b = np.array([[lm33[26][0], lm33[26][1], lm33[26][2]],
                  [lm33[25][0], lm33[25][1], lm33[25][2]]])
    c = np.array([[lm33[28][0], lm33[28][1], lm33[28][2]],
                  [lm33[27][0], lm33[27][1], lm33[27][2]]])
    angles = angle_batch(a, b, c)
    return float(np.mean(angles))

class SquatEngine(BaseWorkoutEngine):
    def initialize(self) -> None:
        model_path = self.config.get("model_path")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.inp_details = self.interpreter.get_input_details()
        self.out_details = self.interpreter.get_output_details()
        
        self.num_inputs = len(self.inp_details)
        self.tiled_depth = (self.num_inputs == 1)
        
        with open(self.config.get("labels_path"), 'r') as f:
            self.labels = [line.strip().split(",")[1].strip() if "," in line else line.strip() for line in f if line.strip()]
            
        self.record_start_angle = 138.0
        self.record_stop_angle = 135.0
        self.min_frames = 8
        
        self.recording = False
        self.frame_buffer = []

    def process_frame(self, landmarks: list) -> FrameResult:
        lm_array = np.array(landmarks)[:, :3]
        current_knee = single_frame_knee_angle(lm_array)
        
        feedback = FeedbackType.NONE
        conf = 0.0
        
        if not self.recording:
            if current_knee < self.record_start_angle:
                self.recording = True
                self.frame_buffer = [lm_array.copy()]
        else:
            self.frame_buffer.append(lm_array.copy())
            if current_knee >= self.record_stop_angle:
                self.recording = False
                if len(self.frame_buffer) >= self.min_frames:
                    frames = np.stack(self.frame_buffer, axis=0)
                    seq_feat, depth_feat = process_rep(frames, tiled_depth=self.tiled_depth)
                    
                    probs = self._predict(seq_feat, depth_feat)
                    pred_idx = int(np.argmax(probs))
                    conf = float(probs[pred_idx])
                    label = self.labels[pred_idx].lower()
                    
                    if label == "perfect":
                        self.rep_count_internal += 1
                        feedback = FeedbackType.PERFECT
                    elif label == "shallow":
                        feedback = FeedbackType.SHALLOW
                    elif label == "backrounding":
                        feedback = FeedbackType.BACK_ROUNDING
                        
                self.frame_buffer = []

        return FrameResult(
            rep_count=self.rep_count_internal,
            feedback=feedback,
            confidence=conf,
            is_recording=self.recording,
            details={"angle": current_knee}
        )

    def _predict(self, seq_feat, depth_feat):
        if self.num_inputs == 1:
            self.interpreter.set_tensor(self.inp_details[0]["index"], seq_feat[np.newaxis].astype(np.float32))
        else:
            for d in self.inp_details:
                if len(d["shape"]) == 3:
                    self.interpreter.set_tensor(d["index"], seq_feat[np.newaxis].astype(np.float32))
                else:
                    expected_depth = int(d["shape"][-1])
                    df = depth_feat[:expected_depth].copy()
                    self.interpreter.set_tensor(d["index"], df[np.newaxis].astype(np.float32))
        
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.out_details[0]["index"])[0].astype(np.float32)

    def draw_custom_visuals(self, render_frame, landmarks: list) -> None:
        import cv2
        h, w, _ = render_frame.shape
        
        def lm_px(idx):
            return (int(landmarks[idx][0] * w), int(landmarks[idx][1] * h))
            
        # Hip depth line
        try:
            l_hip_px = lm_px(23)
            r_hip_px = lm_px(24)
            cv2.line(render_frame, l_hip_px, r_hip_px, (220, 150, 50), 2)
            
            # Knee angle arc label
            lm_array = np.array(landmarks)[:, :3]
            current_knee = single_frame_knee_angle(lm_array)
            r_knee_px = lm_px(26)
            
            # Color logic based on depth
            t = np.clip((current_knee - 90) / (170 - 90), 0.0, 1.0)
            r = int(220 * (1 - t))
            g = int(200 * t)
            kc = (50, g, r) # BGR
            
            # Put text with black outline
            pos = (r_knee_px[0] - 30, r_knee_px[1] - 15)
            cv2.putText(render_frame, f"{current_knee:.0f}", pos, cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(render_frame, f"{current_knee:.0f}", pos, cv2.FONT_HERSHEY_DUPLEX, 0.7, kc, 2, cv2.LINE_AA)
        except Exception as e:
            print("Error drawing custom squat visuals:", e)
