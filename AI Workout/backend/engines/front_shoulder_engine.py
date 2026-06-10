import json
import collections
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d

from core.feedback_engine import FeedbackType
from .base_engine import BaseWorkoutEngine, FrameResult

# ==============================================================================
# CONFIGURATION
# ==============================================================================
VAE_REJECTION_THRESHOLD = 180.0  # Adjust this to make VAE stricter/looser

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

def build_adjacency(edges, num_nodes):
    A = np.eye(num_nodes, dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = np.diag(A.sum(axis=1) ** -0.5)
    return torch.tensor(D @ A @ D, dtype=torch.float32)

class GraphConv(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.W = nn.Linear(in_f, out_f, bias=True)

    def forward(self, x, A):
        return torch.matmul(A, self.W(x))

class STGCBlock(nn.Module):
    def __init__(self, in_c, out_c, tk=3):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c)
        self.tcn = nn.Conv1d(out_c, out_c, tk, padding=tk // 2)
        self.bn  = nn.BatchNorm1d(out_c)
        self.residual = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()

    def forward(self, x, A):
        B, T, N, C = x.shape
        h = F.relu(self.gcn(x.reshape(B*T, N, C), A))
        Co = h.shape[-1]
        h = h.reshape(B, T, N, Co).permute(0, 2, 3, 1).reshape(B*N, Co, T)
        h = self.bn(self.tcn(h))
        h = h.reshape(B, N, Co, T).permute(0, 3, 1, 2)
        return F.relu(h + self.residual(x))

class Encoder(nn.Module):
    def __init__(self, in_c, hid, lat, T, N):
        super().__init__()
        self.stgc1 = STGCBlock(in_c, hid)
        self.stgc2 = STGCBlock(hid, hid)
        self.stgc3 = STGCBlock(hid, hid // 2)
        flat = T * N * (hid // 2)
        self.fc_mu     = nn.Linear(flat, lat)
        self.fc_logvar = nn.Linear(flat, lat)

    def forward(self, x, A):
        h = self.stgc3(self.stgc2(self.stgc1(x, A), A), A)
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, lat, hid, out_c, T, N):
        super().__init__()
        self.T, self.N = T, N
        half = hid // 2
        self.fc = nn.Linear(lat, T * N * half)
        self.stgc1 = STGCBlock(half, hid)
        self.stgc2 = STGCBlock(hid, hid)
        self.gcn_out = GraphConv(hid, out_c)

    def forward(self, z, A):
        h = F.relu(self.fc(z))
        half = h.shape[-1] // (self.T * self.N)
        h = h.reshape(-1, self.T, self.N, half)
        h = self.stgc2(self.stgc1(h, A), A)
        B, T, N, C = h.shape
        out = self.gcn_out(h.reshape(B*T, N, C), A)
        return out.reshape(B, T, N, -1)

class STGCVAE(nn.Module):
    def __init__(self, in_c, hid, lat, T, N):
        super().__init__()
        self.encoder = Encoder(in_c, hid, lat, T, N)
        self.decoder = Decoder(lat, hid, in_c, T, N)

    def forward(self, x, A):
        mu, lv = self.encoder(x, A)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(lv)
        return self.decoder(z, A), mu, lv

    def reconstruction_error(self, x, A):
        with torch.no_grad():
            mu, _ = self.encoder(x, A)
            xr = self.decoder(mu, A)
            return ((x - xr) ** 2).mean(dim=(1, 2, 3))


# ═══════════════════════════════════════════════════════════════════════════════
# REGION CLASSIFICATION (Nose-Anchored Lines)
# ═══════════════════════════════════════════════════════════════════════════════

class Region(Enum):
    STATIC  = 0  # Below Line 1
    LOW     = 1  # Between Line 1 and Line 2
    PERFECT = 2  # Between Line 2 and Line 3
    OVER    = 3  # Above Line 3

class BodyLines:
    def __init__(self, low_pct=0.25, perfect_pct=0.50, over_offset=0.05):
        self.calibrated = False
        self.offsets = {"low": None, "perfect": None, "over": None}
        self.low_pct = low_pct
        self.perfect_pct = perfect_pct
        self.over_offset = over_offset

    def calibrate(self, nose_y, sho_y, elb_y, wri_y):
        dist_we = wri_y - elb_y
        line1_y = wri_y - (self.low_pct * dist_we)
        
        dist_es = elb_y - sho_y
        line2_y = elb_y - (self.perfect_pct * dist_es)
        
        line3_y = sho_y - self.over_offset
        
        self.offsets["low"] = line1_y - nose_y
        self.offsets["perfect"] = line2_y - nose_y
        self.offsets["over"] = line3_y - nose_y
        self.calibrated = True

    def get_lines(self, current_nose_y):
        if not self.calibrated: return None
        return {
            "low": current_nose_y + self.offsets["low"],
            "perfect": current_nose_y + self.offsets["perfect"],
            "over": current_nose_y + self.offsets["over"]
        }

    def classify_region(self, frame: np.ndarray) -> Region:
        if not self.calibrated:
            return Region.STATIC
            
        nose_y = frame[0, 1]
        wrist_y = (frame[15, 1] + frame[16, 1]) / 2.0
        lines = self.get_lines(nose_y)

        if wrist_y > lines["low"]:
            return Region.STATIC
        elif wrist_y > lines["perfect"]:
            return Region.LOW
        elif wrist_y > lines["over"]:
            return Region.PERFECT
        else:
            return Region.OVER

# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

class FrontShoulderEngine(BaseWorkoutEngine):
    def initialize(self) -> None:
        with open(self.config.get("config_path")) as f:
            self.cfg = json.load(f)

        self.NUM_LANDMARKS   = self.cfg["num_landmarks"]
        self.FEAT_PER_LM     = self.cfg["features_per_landmark"]
        self.TEMPORAL_LENGTH = self.cfg["temporal_length"]
        self.LATENT_DIM      = self.cfg["latent_dim"]
        self.HIDDEN_C        = self.cfg["hidden_channels"]
        # Use the same fixed threshold as the original realtime script.
        # The config threshold (0.11) is the VAE training loss, NOT the rejection gate.
        self.THRESHOLD       = VAE_REJECTION_THRESHOLD

        self.GLOBAL_MEAN = np.array(self.cfg["global_mean"], dtype=np.float32)
        self.GLOBAL_STD  = np.array(self.cfg["global_std"],  dtype=np.float32)
        self.EDGES       = [tuple(e) for e in self.cfg["adjacency_edges"]]

        self.device = torch.device("cpu")
        self.A_HAT = build_adjacency(self.EDGES, self.NUM_LANDMARKS).to(self.device)

        self.model = STGCVAE(self.FEAT_PER_LM, self.HIDDEN_C, self.LATENT_DIM, self.TEMPORAL_LENGTH, self.NUM_LANDMARKS).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(self.config.get("model_path"), map_location=self.device))
            self.model.eval()
            print("Front Shoulder STGCVAE Model loaded.")
        except Exception as e:
            print(f"Warning: Could not load model from {self.config.get('model_path')}. Error: {e}")

        self.body_lines = BodyLines()
        self.calibration_data = []
        self.CALIBRATION_COUNT = 30
        
        self.PRE_BUFFER_SIZE = 10
        self.STATIC_CONFIRM = 5
        self.pre_buffer = collections.deque(maxlen=self.PRE_BUFFER_SIZE)
        self.recorded_frames = []
        
        self.state = 0 # 0=IDLE, 1=RECORDING
        self.reached_perfect = False
        self.frames_in_static = 0
        self.current_region = Region.STATIC

    def landmarks_to_array(self, landmarks) -> np.ndarray:
        arr = np.zeros((self.NUM_LANDMARKS, self.FEAT_PER_LM), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            if hasattr(lm, 'x'):
                arr[i] = [lm.x, lm.y, lm.z, lm.visibility]
            else:
                arr[i] = [lm[0], lm[1], lm[2], lm[3]]
        return arr

    def normalise_skeleton_frame(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        hip_mid = (out[23, :3] + out[24, :3]) / 2.0
        sho_mid = (out[11, :3] + out[12, :3]) / 2.0
        torso = np.linalg.norm(sho_mid - hip_mid) + 1e-6
        out[:, :3] = (out[:, :3] - hip_mid) / torso
        return out

    def resample_temporal(self, frames: np.ndarray, target: int) -> np.ndarray:
        T = frames.shape[0]
        if T == target: return frames
        if T < 2: return np.tile(frames, (target, 1, 1))
        flat = frames.reshape(T, -1)
        f = interp1d(np.linspace(0, 1, T), flat, axis=0, kind="linear")
        return f(np.linspace(0, 1, target)).reshape(target, self.NUM_LANDMARKS, self.FEAT_PER_LM).astype(np.float32)

    def z_normalise(self, frames: np.ndarray) -> np.ndarray:
        return (frames - self.GLOBAL_MEAN) / self.GLOBAL_STD

    def validate_rep_with_vae(self, frames_list) -> (bool, float):
        seq = np.stack(frames_list)
        for t in range(seq.shape[0]):
            seq[t] = self.normalise_skeleton_frame(seq[t])
        seq = self.resample_temporal(seq, self.TEMPORAL_LENGTH)
        seq = self.z_normalise(seq)
        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        error = self.model.reconstruction_error(tensor, self.A_HAT).item()
        return error <= self.THRESHOLD, error

    def process_frame(self, landmarks: list) -> FrameResult:
        lm_array = self.landmarks_to_array(landmarks)
        
        if not self.body_lines.calibrated:
            nose_y = lm_array[0, 1]
            sho_y = (lm_array[11, 1] + lm_array[12, 1]) / 2.0
            elb_y = (lm_array[13, 1] + lm_array[14, 1]) / 2.0
            wri_y = (lm_array[15, 1] + lm_array[16, 1]) / 2.0
            
            self.calibration_data.append([nose_y, sho_y, elb_y, wri_y])
            if len(self.calibration_data) >= self.CALIBRATION_COUNT:
                avg = np.mean(self.calibration_data, axis=0)
                self.body_lines.calibrate(avg[0], avg[1], avg[2], avg[3])
                print("Front Shoulder: Calibrated! Lines anchored to nose.")
            
            return FrameResult(rep_count=self.rep_count_internal, feedback=FeedbackType.NONE, confidence=0.0, is_recording=False)

        self.current_region = self.body_lines.classify_region(lm_array)
        feedback = FeedbackType.NONE
        
        if self.state == 0: # IDLE
            self.pre_buffer.append(lm_array.copy())
            # Start recording when entering LOW, PERFECT, or OVER (so we can detect partial reps!)
            if self.current_region in (Region.LOW, Region.PERFECT, Region.OVER):
                self.state = 1
                self.recorded_frames = list(self.pre_buffer)
                self.recorded_frames.append(lm_array.copy())
                self.reached_perfect = (self.current_region == Region.PERFECT)
                self.frames_in_static = 0

        elif self.state == 1: # RECORDING
            self.recorded_frames.append(lm_array.copy())
            if self.current_region == Region.PERFECT:
                self.reached_perfect = True
            elif self.current_region == Region.OVER:
                # Real-time warning: arms too high
                feedback = FeedbackType.OVER_RANGE

            if self.current_region == Region.STATIC:
                self.frames_in_static += 1
                if self.frames_in_static >= self.STATIC_CONFIRM:
                    # Evaluate rep
                    if self.reached_perfect:
                        is_valid, error = self.validate_rep_with_vae(self.recorded_frames)
                        if is_valid:
                            print(f"[Front Shoulder] VAE OK. Error={error:.4f} <= {self.THRESHOLD}")
                            self.rep_count_internal += 1
                            feedback = FeedbackType.PERFECT
                        else:
                            print(f"[Front Shoulder] VAE REJECTED. Error={error:.4f} > {self.THRESHOLD}")
                            feedback = FeedbackType.REJECTED_BY_VAE
                    else:
                        print("[Front Shoulder] Partial rep. Did not reach PERFECT.")
                        feedback = FeedbackType.LOWER_RANGE
                        
                    self.state = 0
                    self.pre_buffer.clear()
                    self.recorded_frames = []
                    self.reached_perfect = False
                    self.frames_in_static = 0  # reset after rep finalizes
            else:
                self.frames_in_static = 0

        return FrameResult(
            rep_count=self.rep_count_internal,
            feedback=feedback,
            confidence=0.95,
            is_recording=(self.state == 1)
        )

    def draw_custom_visuals(self, render_frame, landmarks: list) -> None:
        import cv2
        h, w, _ = render_frame.shape
        lm_array = self.landmarks_to_array(landmarks)
        
        try:
            if self.body_lines.calibrated:
                lines = self.body_lines.get_lines(lm_array[0, 1])
                region = self.current_region
                
                colors = {"low": (0, 165, 255), "perfect": (0, 255, 0), "over": (0, 0, 255)}
                labels = {"low": "Line 1: Low Range", "perfect": "Line 2: Perfect Start", "over": "Line 3: Over Range"}
                
                for name, y_norm in lines.items():
                    y_px = max(0, min(h - 1, int(y_norm * h)))
                    cv2.line(render_frame, (0, y_px), (w, y_px), colors[name], 2)
                    cv2.putText(render_frame, labels[name], (10, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 1)
                
                reg_colors = {Region.STATIC: (180, 180, 180), Region.LOW: (0, 165, 255), Region.PERFECT: (0, 255, 0), Region.OVER: (0, 0, 255)}
                reg_texts = {Region.STATIC: "STATIC", Region.LOW: "LOW", Region.PERFECT: "PERFECT", Region.OVER: "OVER EXTENDED"}
                
                cv2.rectangle(render_frame, (w - 200, 10), (w - 10, 50), reg_colors[region], -1)
                cv2.putText(render_frame, reg_texts[region], (w - 190, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                frames_left = self.CALIBRATION_COUNT - len(self.calibration_data)
                cv2.putText(render_frame, f"Calibrating Nose Anchor... {frames_left}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        except Exception as e:
            print("Error drawing custom front shoulder visuals:", e)
