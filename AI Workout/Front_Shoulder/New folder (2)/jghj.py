# ============================================================================
# FIXED REAL-TIME FITNESS COACH v2
# Key Fix: Body-relative normalization for camera invariance
# ============================================================================

import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

# ============================================================================
# ██████  TUNABLE PARAMETERS  ██████
# ============================================================================

@dataclass
class Config:
    # ── File Paths ──────────────────────────────────────────────────────
    model_dir: str = r"./exported_models"
    camera_id: int = 0

    # ── VAE Gate ────────────────────────────────────────────────────────
    vae_gate_threshold: float = 0.3086
    vae_gate_sensitivity: float = 1.0
    auto_calibrate_gate: bool = True
    calibration_reps: int = 3
    calibration_multiplier: float = 3.0

    # ── Body Normalization ──────────────────────────────────────────────
    # These control how live landmarks are transformed to match training data
    use_body_relative_coords: bool = True     # ← normalize relative to torso
    torso_center_landmarks: Tuple = (11, 12, 23, 24)  # shoulders + hips
    torso_scale_landmarks: Tuple = (11, 23)   # left shoulder to left hip for scale
    target_torso_height: float = 0.35         # ← target torso height in normalized coords
                                               #    (tune this to match your training data)

    # ── Rep Detection ───────────────────────────────────────────────────
    shoulder_angle_up_threshold: float = 70.0
    shoulder_angle_down_threshold: float = 30.0
    min_rep_frames: int = 15
    max_rep_frames: int = 200
    cooldown_frames: int = 5

    # ── Sequence ────────────────────────────────────────────────────────
    max_seq_len: int = 80
    num_landmarks: int = 33
    features_per_landmark: int = 4

    # ── Classification ──────────────────────────────────────────────────
    classification_confidence_threshold: float = 0.6
    smoothing_window: int = 3

    # ── Display ─────────────────────────────────────────────────────────
    display_width: int = 1280
    display_height: int = 720
    show_skeleton: bool = True
    show_angles: bool = True
    show_gate_status: bool = True
    show_debug_info: bool = True
    show_diagnostics: bool = True
    feedback_display_duration: float = 3.0

    # ── MediaPipe ───────────────────────────────────────────────────────
    mp_min_detection_confidence: float = 0.7
    mp_min_tracking_confidence: float = 0.7
    mp_model_complexity: int = 1

    display_angles: Dict = field(default_factory=lambda: {
        'L_shoulder': (13, 11, 23),
        'R_shoulder': (14, 12, 24),
        'L_elbow': (11, 13, 15),
        'R_elbow': (12, 14, 16),
    })

    color_perfect: Tuple = (0, 255, 0)
    color_over_range: Tuple = (0, 165, 255)
    color_low: Tuple = (0, 255, 255)
    color_bent_elbow: Tuple = (0, 0, 255)
    color_idle: Tuple = (200, 200, 200)
    color_gate_pass: Tuple = (0, 255, 0)
    color_gate_block: Tuple = (0, 0, 255)
    color_calibrating: Tuple = (255, 200, 0)


# ============================================================================
# ██████  BODY-RELATIVE NORMALIZER  ██████
# ============================================================================

class BodyNormalizer:
    """
    Transforms raw MediaPipe landmarks into body-relative coordinates
    that are invariant to camera distance, position, and angle.
    
    The idea: 
    1. Center all landmarks on the torso midpoint (avg of shoulders + hips)
    2. Scale so torso height = consistent value
    3. This makes the data look like it came from the same camera setup as training
    """

    def __init__(self, config: Config):
        self.config = config

        # Load training stats to know what the training distribution looked like
        norm_path = os.path.join(config.model_dir, 'norm_stats.npz')
        norm_data = np.load(norm_path)
        self.train_mean = norm_data['mean']  # [132]
        self.train_std = norm_data['std']    # [132]

        # Compute training data's average torso center and scale
        # from the training mean (approximate)
        x_idx = list(range(0, 132, 4))
        y_idx = list(range(1, 132, 4))

        # Training torso center from mean values
        tc_landmarks = config.torso_center_landmarks
        self.train_torso_cx = np.mean([self.train_mean[lm * 4] for lm in tc_landmarks])
        self.train_torso_cy = np.mean([self.train_mean[lm * 4 + 1] for lm in tc_landmarks])

        # Training torso scale
        ts_lms = config.torso_scale_landmarks
        shoulder_y = self.train_mean[ts_lms[0] * 4 + 1]
        hip_y = self.train_mean[ts_lms[1] * 4 + 1]
        self.train_torso_height = abs(hip_y - shoulder_y)

        print(f"  ✓ Body Normalizer initialized")
        print(f"    Training torso center: ({self.train_torso_cx:.4f}, {self.train_torso_cy:.4f})")
        print(f"    Training torso height: {self.train_torso_height:.4f}")

    def normalize_frame(self, landmarks_4d: np.ndarray) -> np.ndarray:
        """
        Transform a single frame [33, 4] to match training distribution.
        
        Steps:
        1. Compute live torso center and scale
        2. Translate so torso center matches training center
        3. Scale so torso height matches training height
        4. Keep visibility as-is
        """
        result = landmarks_4d.copy()

        # Current torso center
        tc_lms = self.config.torso_center_landmarks
        live_cx = np.mean([landmarks_4d[lm, 0] for lm in tc_lms])
        live_cy = np.mean([landmarks_4d[lm, 1] for lm in tc_lms])

        # Current torso scale
        ts_lms = self.config.torso_scale_landmarks
        live_shoulder_y = landmarks_4d[ts_lms[0], 1]
        live_hip_y = landmarks_4d[ts_lms[1], 1]
        live_torso_height = abs(live_hip_y - live_shoulder_y) + 1e-8

        # Scale factor to match training
        scale = self.train_torso_height / live_torso_height

        # Transform x, y, z (not visibility)
        for i in range(33):
            # Translate to origin (torso center)
            result[i, 0] = (landmarks_4d[i, 0] - live_cx) * scale + self.train_torso_cx
            result[i, 1] = (landmarks_4d[i, 1] - live_cy) * scale + self.train_torso_cy
            result[i, 2] = landmarks_4d[i, 2] * scale  # z: just scale
            # visibility stays the same
            result[i, 3] = landmarks_4d[i, 3]

        return result

    def normalize_rep(self, frames: np.ndarray) -> np.ndarray:
        """Normalize a sequence of frames [T, 33, 4]."""
        result = np.zeros_like(frames)
        for t in range(frames.shape[0]):
            result[t] = self.normalize_frame(frames[t])
        return result


# ============================================================================
# ██████  DIAGNOSTICS  ██████
# ============================================================================

class DataDiagnostics:
    def __init__(self, config: Config):
        self.config = config
        norm_data = np.load(os.path.join(config.model_dir, 'norm_stats.npz'))
        self.train_mean = norm_data['mean']
        self.train_std = norm_data['std']
        self.live_frames_raw = []
        self.live_frames_normalized = []
        self.diagnosed = False
        self.diagnosis_result = None

    def add_frame(self, raw_4d: np.ndarray, normalized_4d: np.ndarray):
        self.live_frames_raw.append(raw_4d.flatten())
        self.live_frames_normalized.append(normalized_4d.flatten())

    def diagnose(self) -> dict:
        if len(self.live_frames_raw) < 30:
            return {'status': 'collecting'}

        raw = np.array(self.live_frames_raw)
        normed = np.array(self.live_frames_normalized)
        raw_mean = raw.mean(axis=0)
        normed_mean = normed.mean(axis=0)

        x_idx = list(range(0, 132, 4))
        y_idx = list(range(1, 132, 4))
        z_idx = list(range(2, 132, 4))
        v_idx = list(range(3, 132, 4))

        result = {
            'status': 'diagnosed',
            'num_frames': len(self.live_frames_raw),
        }

        for name, idx in [('x', x_idx), ('y', y_idx), ('z', z_idx), ('vis', v_idx)]:
            result[f'train_{name}'] = (float(self.train_mean[idx].min()), float(self.train_mean[idx].max()))
            result[f'raw_{name}'] = (float(raw_mean[idx].min()), float(raw_mean[idx].max()))
            result[f'norm_{name}'] = (float(normed_mean[idx].min()), float(normed_mean[idx].max()))

        self.diagnosed = True
        self.diagnosis_result = result
        return result

    def print_diagnosis(self):
        if not self.diagnosis_result:
            return
        d = self.diagnosis_result
        print(f"\n{'='*70}")
        print(f"  DATA DISTRIBUTION DIAGNOSIS ({d['num_frames']} frames)")
        print(f"{'='*70}")
        print(f"  {'':8s} {'Training':>22s}  {'Raw Live':>22s}  {'Body-Normalized':>22s}")
        print(f"  {'─'*76}")
        for name in ['x', 'y', 'z', 'vis']:
            tr = d[f'train_{name}']
            rw = d[f'raw_{name}']
            nm = d[f'norm_{name}']
            print(f"  {name:8s} [{tr[0]:8.4f},{tr[1]:8.4f}]  "
                  f"[{rw[0]:8.4f},{rw[1]:8.4f}]  "
                  f"[{nm[0]:8.4f},{nm[1]:8.4f}]")
        print(f"{'='*70}")


# ============================================================================
# ██████  ANGLE CALCULATOR  ██████
# ============================================================================

class AngleCalculator:
    TRAINING_ANGLE_TRIPLETS = [
        (11, 13, 15), (12, 14, 16),
        (13, 11, 23), (14, 12, 24),
        (11, 23, 25), (12, 24, 26),
        (23, 25, 27), (24, 26, 28),
        (12, 11, 13), (11, 12, 14),
        (11, 12, 24), (12, 11, 23),
    ]

    @staticmethod
    def compute_angle_degrees(a, b, c):
        v1 = a - b; v2 = c - b
        cos_a = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1, 1)
        return np.degrees(np.arccos(cos_a))

    @staticmethod
    def compute_training_angles(landmarks_3d):
        angles = np.zeros(len(AngleCalculator.TRAINING_ANGLE_TRIPLETS), dtype=np.float32)
        for i, (a, c, b) in enumerate(AngleCalculator.TRAINING_ANGLE_TRIPLETS):
            v1 = landmarks_3d[a] - landmarks_3d[c]
            v2 = landmarks_3d[b] - landmarks_3d[c]
            cos_a = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1, 1)
            angles[i] = np.arccos(cos_a) / np.pi
        return angles

    @staticmethod
    def get_shoulder_flexion_angle(landmarks_3d):
        l = AngleCalculator.compute_angle_degrees(landmarks_3d[13], landmarks_3d[11], landmarks_3d[23])
        r = AngleCalculator.compute_angle_degrees(landmarks_3d[14], landmarks_3d[12], landmarks_3d[24])
        return (l + r) / 2.0


# ============================================================================
# ██████  REP STATE MACHINE  ██████
# ============================================================================

class RepState:
    IDLE = "idle"; GOING_UP = "going_up"; GOING_DOWN = "going_down"; COOLDOWN = "cooldown"

    def __init__(self, config: Config):
        self.config = config
        self.state = self.IDLE
        self.frame_count = 0
        self.cooldown_count = 0
        self.rep_frames = []
        self.rep_angles = []
        self.peak_angle = 0.0
        self.rep_count = 0

    def update(self, landmarks_4d, landmarks_3d, training_angles):
        angle = AngleCalculator.get_shoulder_flexion_angle(landmarks_3d)
        completed = None

        if self.state == self.COOLDOWN:
            self.cooldown_count += 1
            if self.cooldown_count >= self.config.cooldown_frames:
                self.state = self.IDLE; self.cooldown_count = 0
            return None

        if self.state == self.IDLE and angle > self.config.shoulder_angle_up_threshold:
            self.state = self.GOING_UP
            self.frame_count = 0; self.rep_frames = []; self.rep_angles = []
            self.peak_angle = angle

        if self.state in (self.GOING_UP, self.GOING_DOWN):
            self.rep_frames.append(landmarks_4d.copy())
            self.rep_angles.append(training_angles.copy())
            self.frame_count += 1
            self.peak_angle = max(self.peak_angle, angle)

            if self.state == self.GOING_UP and angle < self.peak_angle - 10:
                self.state = self.GOING_DOWN

            if self.state == self.GOING_DOWN and angle < self.config.shoulder_angle_down_threshold:
                if self.frame_count >= self.config.min_rep_frames:
                    self.rep_count += 1
                    completed = {
                        'frames': np.array(self.rep_frames),
                        'angles': np.array(self.rep_angles),
                        'frame_count': self.frame_count,
                        'peak_angle': self.peak_angle,
                        'rep_number': self.rep_count,
                    }
                self.state = self.COOLDOWN; self.cooldown_count = 0
                self.frame_count = 0; self.rep_frames = []; self.rep_angles = []

            if self.frame_count > self.config.max_rep_frames:
                self.state = self.IDLE; self.frame_count = 0
                self.rep_frames = []; self.rep_angles = []

        return completed

    def get_state_info(self):
        return {'state': self.state, 'frame_count': self.frame_count,
                'rep_count': self.rep_count, 'peak_angle': self.peak_angle}


# ============================================================================
# ██████  INFERENCE ENGINE  ██████
# ============================================================================

class InferenceEngine:
    CLASS_NAMES = ['perfect', 'over_range', 'low', 'bent_elbow']
    FEEDBACK = {
        'perfect': "Perfect form! Keep it up!",
        'over_range': "Too high! Lower your arms slightly.",
        'low': "Raise higher! Arms should reach shoulder level.",
        'bent_elbow': "Straighten your elbows! Keep arms extended.",
    }

    def __init__(self, config: Config, body_normalizer: BodyNormalizer):
        self.config = config
        self.normalizer = body_normalizer

        providers = ['CPUExecutionProvider']
        vae_path = os.path.join(config.model_dir, 'vae_encoder.onnx')
        cls_path = os.path.join(config.model_dir, 'classifier.onnx')
        self.vae_session = ort.InferenceSession(vae_path, providers=providers)
        self.cls_session = ort.InferenceSession(cls_path, providers=providers)
        print(f"  ✓ Models loaded")

        norm_data = np.load(os.path.join(config.model_dir, 'norm_stats.npz'))
        self.feat_mean = norm_data['mean']
        self.feat_std = norm_data['std']
        angle_data = np.load(os.path.join(config.model_dir, 'angle_stats.npz'))
        self.angle_mean = angle_data['mean']
        self.angle_std = angle_data['std']

        self.prediction_history = deque(maxlen=config.smoothing_window)
        self.calibration_scores = []
        self.is_calibrated = not config.auto_calibrate_gate
        self.live_gate_threshold = config.vae_gate_threshold * config.vae_gate_sensitivity

    @property
    def effective_threshold(self):
        return self.live_gate_threshold

    def preprocess_rep(self, frames, angles):
        """
        Preprocess rep with body-relative normalization.
        frames: [T, 33, 4] — already body-normalized by BodyNormalizer
        angles: [T, 12]
        """
        T = frames.shape[0]
        ml = self.config.max_seq_len

        if T >= ml:
            fp, ap = frames[:ml], angles[:ml]
            actual = ml
        else:
            fp = np.concatenate([frames, np.zeros((ml - T, 33, 4), dtype=np.float32)])
            ap = np.concatenate([angles, np.zeros((ml - T, 12), dtype=np.float32)])
            actual = T

        # Normalize keypoints using training stats
        flat = fp.reshape(ml, -1)
        flat_norm = (flat - self.feat_mean) / self.feat_std
        if T < ml:
            flat_norm[T:] = 0
        kp = flat_norm.reshape(1, ml, 33, 4).astype(np.float32)

        # Normalize angles
        an = (ap - self.angle_mean) / self.angle_std
        if T < ml:
            an[T:] = 0
        an = an.reshape(1, ml, 12).astype(np.float32)

        return kp, an

    def run_inference(self, rep_data: dict) -> dict:
        """Full inference with body-normalized input."""

        # Body-normalize the rep frames
        raw_frames = rep_data['frames']  # [T, 33, 4]
        norm_frames = self.normalizer.normalize_rep(raw_frames)

        t0 = time.time()
        kp, an = self.preprocess_rep(norm_frames, rep_data['angles'])

        # VAE
        vae_out = self.vae_session.run(None, {'input_sequence': kp})
        mu, joint_err, total_err = vae_out[0], vae_out[1], float(vae_out[2][0])
        vae_ms = (time.time() - t0) * 1000

        # Auto-calibration
        if self.config.auto_calibrate_gate and not self.is_calibrated:
            self.calibration_scores.append(total_err)
            n = len(self.calibration_scores)
            if n >= self.config.calibration_reps:
                s = np.array(self.calibration_scores)
                self.live_gate_threshold = s.mean() + self.config.calibration_multiplier * s.std()
                self.is_calibrated = True
                print(f"\n  ★ AUTO-CALIBRATED: {[f'{x:.4f}' for x in s]} → threshold={self.live_gate_threshold:.4f}")

            # During calibration, always classify
            return self._classify(kp, an, mu, joint_err, total_err, vae_ms, rep_data,
                                   is_exercise=True, calibrating=True,
                                   cal_progress=f"{n}/{self.config.calibration_reps}")

        # Normal gate
        is_ex = total_err < self.effective_threshold
        if not is_ex:
            return {
                'is_exercise': False, 'gate_score': total_err,
                'gate_threshold': self.effective_threshold, 'calibrating': False,
                'class_name': 'unknown', 'confidence': 0.0, 'probabilities': {},
                'feedback': "Not recognized as Front Shoulder Raise.",
                'vae_time_ms': vae_ms, 'cls_time_ms': 0,
                'rep_number': rep_data['rep_number'],
                'frame_count': rep_data['frame_count'],
                'peak_angle': rep_data['peak_angle'],
            }

        return self._classify(kp, an, mu, joint_err, total_err, vae_ms, rep_data,
                               is_exercise=True, calibrating=False)

    def _classify(self, kp, an, mu, joint_err, total_err, vae_ms, rep_data,
                  is_exercise, calibrating, cal_progress=""):
        """Run classifier and return result dict."""
        node_feat = np.concatenate([kp, joint_err[..., np.newaxis]], axis=-1).astype(np.float32)

        t1 = time.time()
        cls_out = self.cls_session.run(None, {
            'node_features': node_feat,
            'angle_features': an,
            'vae_latent': mu.astype(np.float32),
        })
        cls_ms = (time.time() - t1) * 1000

        probs = cls_out[1][0]
        pred_idx = int(np.argmax(probs))
        pred_name = self.CLASS_NAMES[pred_idx]

        self.prediction_history.append(pred_idx)
        if len(self.prediction_history) >= 2:
            smoothed = self.CLASS_NAMES[int(np.argmax(np.bincount(list(self.prediction_history), minlength=4)))]
        else:
            smoothed = pred_name

        return {
            'is_exercise': is_exercise,
            'gate_score': total_err,
            'gate_threshold': self.effective_threshold,
            'calibrating': calibrating,
            'calibration_progress': cal_progress,
            'class_name': pred_name,
            'class_idx': pred_idx,
            'confidence': float(probs[pred_idx]),
            'probabilities': {self.CLASS_NAMES[i]: float(probs[i]) for i in range(4)},
            'smoothed_class': smoothed,
            'feedback': self.FEEDBACK.get(pred_name, ""),
            'vae_time_ms': vae_ms,
            'cls_time_ms': cls_ms,
            'rep_number': rep_data['rep_number'],
            'frame_count': rep_data['frame_count'],
            'peak_angle': rep_data['peak_angle'],
        }


# ============================================================================
# ██████  DISPLAY RENDERER  ██████
# ============================================================================

class DisplayRenderer:
    def __init__(self, config: Config):
        self.config = config
        self.last_feedback = None
        self.feedback_start_time = 0
        self.color_map = {
            'perfect': config.color_perfect, 'over_range': config.color_over_range,
            'low': config.color_low, 'bent_elbow': config.color_bent_elbow,
            'unknown': config.color_idle,
        }

    def draw_skeleton(self, frame, landmarks, color=(0, 255, 0)):
        if landmarks is None: return
        h, w = frame.shape[:2]
        for conn in mp.solutions.pose.POSE_CONNECTIONS:
            p1, p2 = landmarks.landmark[conn[0]], landmarks.landmark[conn[1]]
            if p1.visibility > 0.5 and p2.visibility > 0.5:
                cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), color, 2)
        for lm in landmarks.landmark:
            if lm.visibility > 0.5:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, color, -1)

    def draw_angle_labels(self, frame, lm3d, landmarks):
        if landmarks is None: return
        h, w = frame.shape[:2]
        for name, (a, c, b) in self.config.display_angles.items():
            ang = AngleCalculator.compute_angle_degrees(lm3d[a], lm3d[c], lm3d[b])
            lm = landmarks.landmark[c]
            if lm.visibility > 0.5:
                cv2.putText(frame, f"{name}:{ang:.0f}", (int(lm.x*w)+10, int(lm.y*h)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def draw_gate_bar(self, frame, score, is_ex, is_cal):
        h, w = frame.shape[:2]; x, y = 20, h - 80
        if score is None: return
        th = self.config.vae_gate_threshold * self.config.vae_gate_sensitivity
        mx = max(th * 2, score * 1.2)
        bw, bh = 200, 20
        cv2.rectangle(frame, (x,y), (x+bw,y+bh), (50,50,50), -1)
        fw = int(bw * min(score/mx, 1.0))
        c = self.config.color_calibrating if is_cal else (self.config.color_gate_pass if is_ex else self.config.color_gate_block)
        cv2.rectangle(frame, (x,y), (x+fw,y+bh), c, -1)
        tx = int(x + bw * min(th/mx, 1.0))
        cv2.line(frame, (tx,y-5), (tx,y+bh+5), (255,255,255), 2)
        st = "CALIBRATING" if is_cal else ("EXERCISE" if is_ex else "NOT EXERCISE")
        cv2.putText(frame, f"Gate: {st} ({score:.3f})", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

    def draw_rep_counter(self, frame, info):
        h, w = frame.shape[:2]
        txt = f"REPS: {info['rep_count']}"
        ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        x = w - ts[0] - 30
        cv2.putText(frame, txt, (x, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
        cv2.putText(frame, f"State: {info['state'].upper()}", (x, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        if info.get('current_angle') is not None:
            cv2.putText(frame, f"Shoulder: {info['current_angle']:.0f}deg", (x, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    def draw_calibration_banner(self, frame, text):
        h, w = frame.shape[:2]
        ov = frame.copy(); cv2.rectangle(ov, (0,0), (w,55), (0,0,0), -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"CALIBRATING - Do {text} more reps", (20,38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.config.color_calibrating, 2)

    def draw_feedback(self, frame, result):
        h, w = frame.shape[:2]; now = time.time()
        if result is not None:
            self.last_feedback = result; self.feedback_start_time = now
        if self.last_feedback and (now - self.feedback_start_time) < self.config.feedback_display_duration:
            fb = self.last_feedback
            cn = fb.get('smoothed_class', fb.get('class_name', 'unknown'))
            conf = fb.get('confidence', 0)
            c = self.color_map.get(cn, self.config.color_idle)
            ov = frame.copy(); cv2.rectangle(ov, (20,10), (w-20,130), (0,0,0), -1)
            cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"Rep #{fb.get('rep_number','?')}: {cn.upper()}", (40,50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2)
            bx, by, bw2, bh2 = 40, 70, w-80, 15
            cv2.rectangle(frame, (bx,by), (bx+bw2,by+bh2), (50,50,50), -1)
            cv2.rectangle(frame, (bx,by), (bx+int(bw2*conf),by+bh2), c, -1)
            cv2.putText(frame, f"{conf:.1%}", (bx+bw2+10,by+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(frame, fb.get('feedback',''), (40,115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            probs = fb.get('probabilities', {}); py = 150
            for pn, pp in sorted(probs.items(), key=lambda x: -x[1]):
                cv2.putText(frame, f"  {pn}: {pp:.1%}", (40,py), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                           self.color_map.get(pn,(200,200,200)), 1); py += 20

    def draw_diagnostics(self, frame, d):
        if not d: return
        h, w = frame.shape[:2]; x, y0 = 20, h - 200
        cv2.putText(frame, "DIAGNOSTICS (body-normalized):", (x,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,200,0), 1)
        for i, nm in enumerate(['x','y','z']):
            tr = d.get(f'train_{nm}', (0,0)); nrm = d.get(f'norm_{nm}', (0,0))
            cv2.putText(frame, f"{nm}: train[{tr[0]:.3f},{tr[1]:.3f}] live[{nrm[0]:.3f},{nrm[1]:.3f}]",
                       (x, y0+18+i*16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

    def draw_debug(self, frame, fps, ms):
        h = frame.shape[0]
        cv2.putText(frame, f"FPS:{fps:.0f} Inf:{ms:.1f}ms", (20,h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

    def draw_keys(self, frame):
        h, w = frame.shape[:2]
        for i, t in enumerate(["Q:Quit","R:Reset","D:Debug","S:Skel","A:Ang","C:Recal"]):
            cv2.putText(frame, t, (w-140, h-10-i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)


# ============================================================================
# ██████  MAIN APP  ██████
# ============================================================================

class FitnessCoach:
    def __init__(self, config: Config):
        self.config = config
        print("="*60)
        print("  REAL-TIME FITNESS COACH v2 — Body-Relative Normalization")
        print("="*60)

        print("\n[1/5] Body normalizer...")
        self.body_norm = BodyNormalizer(config)

        print("\n[2/5] Models...")
        self.engine = InferenceEngine(config, self.body_norm)

        print("\n[3/5] MediaPipe...")
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=config.mp_min_detection_confidence,
            min_tracking_confidence=config.mp_min_tracking_confidence,
            model_complexity=config.mp_model_complexity)

        print("\n[4/5] State machine...")
        self.rep_state = RepState(config)

        print("\n[5/5] Display & diagnostics...")
        self.renderer = DisplayRenderer(config)
        self.diagnostics = DataDiagnostics(config)

        self.last_gate_score = None
        self.last_is_ex = None
        self.last_cal = False
        self.fps_q = deque(maxlen=30)
        self.last_result = None
        self.inf_ms = 0

        if config.auto_calibrate_gate:
            print(f"\n  ★ Auto-cal ON: do {config.calibration_reps} reps first")
        print(f"\n{'='*60}\n  READY\n{'='*60}")

    def run(self):
        cap = cv2.VideoCapture(self.config.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.display_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.display_height)
        if not cap.isOpened():
            print("ERROR: No camera"); return

        try:
            while True:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)

                results = self.mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                new_result = None

                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    lm4d = np.array([[l.x,l.y,l.z,l.visibility] for l in lms], dtype=np.float32)
                    lm3d = lm4d[:,:3]

                    # Body-normalize for diagnostics
                    lm4d_norm = self.body_norm.normalize_frame(lm4d)
                    if not self.diagnostics.diagnosed:
                        self.diagnostics.add_frame(lm4d, lm4d_norm)
                        if len(self.diagnostics.live_frames_raw) == 60:
                            self.diagnostics.diagnose()
                            self.diagnostics.print_diagnosis()

                    # Angles (computed from RAW landmarks — angles are invariant)
                    training_angles = AngleCalculator.compute_training_angles(lm3d)
                    current_angle = AngleCalculator.get_shoulder_flexion_angle(lm3d)

                    # Rep state uses RAW landmarks for angle detection,
                    # but stores BODY-NORMALIZED landmarks for classification
                    completed = self.rep_state.update(lm4d_norm, lm3d, training_angles)

                    if completed is not None:
                        ti = time.time()
                        result = self.engine.run_inference(completed)
                        self.inf_ms = (time.time() - ti) * 1000

                        self.last_gate_score = result['gate_score']
                        self.last_is_ex = result['is_exercise']
                        self.last_cal = result.get('calibrating', False)
                        self.last_result = result
                        new_result = result

                        tag = " [CAL]" if result.get('calibrating') else ""
                        print(f"  Rep #{result['rep_number']} | "
                              f"Gate:{'PASS' if result['is_exercise'] else 'BLOCK'} "
                              f"({result['gate_score']:.3f}/{result['gate_threshold']:.3f}) | "
                              f"{result.get('class_name','?')} ({result.get('confidence',0):.0%}) | "
                              f"{result['frame_count']}fr peak={result['peak_angle']:.0f}° | "
                              f"{result['vae_time_ms']:.0f}+{result.get('cls_time_ms',0):.0f}ms{tag}")

                    # Draw
                    if self.config.show_skeleton:
                        sc = self.config.color_idle
                        if self.last_result and self.last_result.get('is_exercise'):
                            cn = self.last_result.get('smoothed_class', 'unknown')
                            sc = self.renderer.color_map.get(cn, self.config.color_idle)
                        self.renderer.draw_skeleton(frame, results.pose_landmarks, sc)
                    if self.config.show_angles:
                        self.renderer.draw_angle_labels(frame, lm3d, results.pose_landmarks)

                    si = self.rep_state.get_state_info(); si['current_angle'] = current_angle
                    self.renderer.draw_rep_counter(frame, si)
                else:
                    self.renderer.draw_rep_counter(frame, self.rep_state.get_state_info())
                    cv2.putText(frame, "No person", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                if not self.engine.is_calibrated:
                    n = len(self.engine.calibration_scores)
                    self.renderer.draw_calibration_banner(frame, str(self.config.calibration_reps - n))

                self.renderer.draw_feedback(frame, new_result)
                if self.config.show_gate_status:
                    self.renderer.draw_gate_bar(frame, self.last_gate_score, self.last_is_ex, self.last_cal)
                if self.config.show_diagnostics and self.diagnostics.diagnosed:
                    self.renderer.draw_diagnostics(frame, self.diagnostics.diagnosis_result)

                t1 = time.time(); self.fps_q.append(t1-t0)
                fps = 1.0 / (sum(self.fps_q)/len(self.fps_q)) if self.fps_q else 0
                if self.config.show_debug_info:
                    self.renderer.draw_debug(frame, fps, self.inf_ms)
                self.renderer.draw_keys(frame)

                cv2.imshow('Fitness Coach v2', frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27): break
                elif key == ord('r'):
                    self.rep_state = RepState(self.config)
                    self.engine.prediction_history.clear()
                    self.last_result = None; self.last_gate_score = None
                    print("  *** Reset ***")
                elif key == ord('d'): self.config.show_debug_info = not self.config.show_debug_info
                elif key == ord('s'): self.config.show_skeleton = not self.config.show_skeleton
                elif key == ord('a'): self.config.show_angles = not self.config.show_angles
                elif key == ord('c'):
                    self.engine.calibration_scores = []; self.engine.is_calibrated = False
                    print("  *** Recalibrating ***")
        finally:
            cap.release(); cv2.destroyAllWindows(); self.mp_pose.close()
            print(f"\n{'='*60}\n  DONE — {self.rep_state.rep_count} reps")
            if self.engine.is_calibrated:
                print(f"  Gate threshold: {self.engine.live_gate_threshold:.4f}")
            print(f"{'='*60}")


# ============================================================================
# ██████  RUN  ██████
# ============================================================================

if __name__ == "__main__":
    config = Config(
        model_dir=r"./exported_models",
        camera_id=1,

        # Gate
        vae_gate_threshold=1.9162,
        vae_gate_sensitivity=1.0,
        auto_calibrate_gate=True,
        calibration_reps=3,
        calibration_multiplier=3.0,

        # Body normalization
        use_body_relative_coords=True,
        target_torso_height=0.35,

        # Rep detection
        shoulder_angle_up_threshold=70.0,
        shoulder_angle_down_threshold=30.0,
        min_rep_frames=15,
        max_rep_frames=200,
        cooldown_frames=5,

        # Classification
        classification_confidence_threshold=0.6,
        smoothing_window=3,

        # Display
        show_skeleton=True,
        show_angles=True,
        show_gate_status=True,
        show_debug_info=True,
        show_diagnostics=True,
        feedback_display_duration=3.0,

        # MediaPipe
        mp_min_detection_confidence=0.7,
        mp_min_tracking_confidence=0.7,
        mp_model_complexity=1,
    )

    coach = FitnessCoach(config)
    coach.run()