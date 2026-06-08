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

def resample(seq, target):
    T = len(seq)
    if T == target:
        return seq.astype(np.float32)
    idx = np.linspace(0, T - 1, target)
    out = np.zeros((target, seq.shape[1]), dtype=np.float32)
    for f in range(seq.shape[1]):
        out[:, f] = np.interp(idx, np.arange(T), seq[:, f])
    return out

JOINT_IDX = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]

def process_rep(frames, target_frames):
    T    = len(frames)
    norm = ema_smooth(frames, 0.5)
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

class ShrugEngine(BaseWorkoutEngine):
    def initialize(self) -> None:
        self.interpreter = tflite.Interpreter(model_path=self.config.get("model_path"))
        self.interpreter.allocate_tensors()
        self.inp_details = self.interpreter.get_input_details()
        self.out_details = self.interpreter.get_output_details()
        
        self.TARGET_FRAMES = 48
        for d in self.inp_details:
            if len(d["shape"]) == 3:
                self.TARGET_FRAMES = int(d["shape"][1])
                break

        with open(self.config.get("labels_path"), 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
            self.labels = [ln.split(",", 1)[1].strip() if "," in ln else ln for ln in lines]

        # VAE setup (optional)
        self.vae_interp = None
        stats_path = self.config.get("stats_path")
        if stats_path:
            try:
                st = np.load(stats_path, allow_pickle=True)
                if "vae_kl_threshold" in st:
                    self.kl_threshold = float(st["vae_kl_threshold"][0])
                    self.vae_sc_mean = st["vae_scaler_mean"].astype(np.float32)
                    self.vae_sc_scale = st["vae_scaler_scale"].astype(np.float32)
                    
                    # Assume VAE model is next to classifier model
                    vae_path = self.config.get("model_path").replace("shrug_classifier_fp16.tflite", "shrug_vae_encoder.tflite")
                    self.vae_interp = tflite.Interpreter(model_path=vae_path)
                    self.vae_interp.allocate_tensors()
                    self.vae_inp = self.vae_interp.get_input_details()
                    self.vae_out_det = self.vae_interp.get_output_details()
            except Exception as e:
                print(f"Skipping VAE for shrugs: {e}")

        self.recording = False
        self.frame_buffer = []
        self.pre_trigger_buf = []
        
        # New Nose-anchored calibration
        self.baseline_sho_y = []
        self.baseline_nose_y = []
        self.baseline_done = False
        self.BASELINE_FRAMES = 40
        self.nose_sho_gap = 0.0
        self.smoothed_nose_y = 0.0
        self.stop_confirm_count = 0

    def draw_custom_visuals(self, render_frame, landmarks: list) -> None:
        import cv2
        h, w, _ = render_frame.shape
        def lm_px(idx):
            return (int(landmarks[idx][0] * w), int(landmarks[idx][1] * h))
            
        try:
            # Elbow spread line
            el_l_px = lm_px(13)
            el_r_px = lm_px(14)
            cv2.line(render_frame, el_l_px, el_r_px, (50, 200, 80), 3)
            
            # Threshold gap info
            if self.baseline_done:
                # We use smoothed_nose_y + nose_sho_gap
                shoulder_line_y = self.smoothed_nose_y + self.nose_sho_gap
                sho_line_px = int(shoulder_line_y * h)
                cv2.line(render_frame, (0, sho_line_px), (w, sho_line_px), (0, 220, 80) if self.recording else (0, 255, 180), 2)
        except Exception as e:
            print("Error drawing custom shrug visuals:", e)

    def process_frame(self, landmarks: list) -> FrameResult:
        lm = np.array(landmarks)[:, :3]
        
        shoulder_y = float((lm[11, 1] + lm[12, 1]) / 2.0)
        nose_y = float(lm[0, 1])
        
        feedback = FeedbackType.NONE
        conf = 0.0

        if not self.baseline_done:
            self.baseline_sho_y.append(shoulder_y)
            self.baseline_nose_y.append(nose_y)
            if len(self.baseline_sho_y) >= self.BASELINE_FRAMES:
                cal_nose_y = float(np.mean(self.baseline_nose_y))
                cal_sho_y = float(np.mean(self.baseline_sho_y))
                self.nose_sho_gap = cal_sho_y - cal_nose_y
                self.smoothed_nose_y = cal_nose_y
                self.baseline_done = True
            return FrameResult(rep_count=self.rep_count_internal, feedback=feedback, confidence=conf, is_recording=self.recording)

        # Smooth nose tracker
        NOSE_SMOOTH_ALPHA = 0.05
        self.smoothed_nose_y = (NOSE_SMOOTH_ALPHA * nose_y + (1.0 - NOSE_SMOOTH_ALPHA) * self.smoothed_nose_y)
        shoulder_line_y = self.smoothed_nose_y + self.nose_sho_gap

        if not self.recording:
            self.pre_trigger_buf.append(lm.copy())
            if len(self.pre_trigger_buf) > 10:
                self.pre_trigger_buf.pop(0)
                
            # START: shoulder rose above the rest line (sho_y decreases)
            if shoulder_y < shoulder_line_y:
                self.recording = True
                self.stop_confirm_count = 0
                self.frame_buffer = list(self.pre_trigger_buf)
        else:
            self.frame_buffer.append(lm.copy())
            
            # STOP: shoulder returned to or below the rest line
            if shoulder_y >= shoulder_line_y:
                self.stop_confirm_count += 1
                
                if self.stop_confirm_count >= 5:
                    self.recording = False
                    self.stop_confirm_count = 0
                    if len(self.frame_buffer) >= 8:
                    seq, dep = process_rep(np.stack(self.frame_buffer), self.TARGET_FRAMES)
                    
                    is_ood = False
                    if self.vae_interp is not None:
                        n = min(len(dep), len(self.vae_sc_mean))
                        x = ((dep[:n] - self.vae_sc_mean[:n]) / (self.vae_sc_scale[:n] + 1e-6)).reshape(1, -1).astype(np.float32)
                        self.vae_interp.set_tensor(self.vae_inp[0]["index"], x)
                        self.vae_interp.invoke()
                        # Extract mu/logvar
                        mu_t = self.vae_interp.get_tensor(self.vae_out_det[0]["index"])[0]
                        lv_t = self.vae_interp.get_tensor(self.vae_out_det[1]["index"])[0]
                        kl = float(0.5 * np.sum(np.square(mu_t) + np.exp(lv_t) - lv_t - 1.0))
                        if kl > self.kl_threshold:
                            is_ood = True
                    
                    if is_ood:
                        feedback = FeedbackType.NOT_WORKOUT
                    else:
                        probs = self._predict(seq, dep)
                        pred_idx = int(np.argmax(probs))
                        conf = float(probs[pred_idx])
                        label = self.labels[pred_idx].lower()
                        
                        if label == "perfect":
                            self.rep_count_internal += 1
                            feedback = FeedbackType.PERFECT
                        elif label == "bent_elbow":
                            feedback = FeedbackType.BENT_ELBOW
                            
                    self.frame_buffer = []
            else:
                self.stop_confirm_count = 0

        return FrameResult(
            rep_count=self.rep_count_internal,
            feedback=feedback,
            confidence=conf,
            is_recording=self.recording,
            details={"nose_y": nose_y, "sho_y": shoulder_y, "line_y": shoulder_line_y if self.baseline_done else 0}
        )

    def _predict(self, seq_feat, depth_feat):
        if len(self.inp_details) == 1:
            self.interpreter.set_tensor(self.inp_details[0]["index"], seq_feat[np.newaxis].astype(np.float32))
        else:
            for d in self.inp_details:
                if len(d["shape"]) == 3:
                    self.interpreter.set_tensor(d["index"], seq_feat[np.newaxis].astype(np.float32))
                else:
                    n = int(d["shape"][-1])
                    self.interpreter.set_tensor(d["index"], depth_feat[:n][np.newaxis].astype(np.float32))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.out_details[0]["index"])[0].astype(np.float32)
