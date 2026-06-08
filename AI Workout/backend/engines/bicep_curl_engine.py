import numpy as np
import tensorflow.lite as tflite
from core.feedback_engine import FeedbackType
from .base_engine import BaseWorkoutEngine, FrameResult

def torso_normalize(lm):
    mid_hip = (lm[23] + lm[24]) / 2
    mid_sho = (lm[11] + lm[12]) / 2
    torso_len = np.linalg.norm(mid_sho - mid_hip)
    scale = max(torso_len, 1e-6)
    return (lm - mid_hip) / scale

def ema_smooth(seq, alpha):
    if len(seq) == 0: return seq
    out = [seq[0]]
    for i in range(1, len(seq)):
        out.append(alpha * seq[i] + (1-alpha) * out[-1])
    return np.array(out)

def resample(seq, target_len=64):
    L = len(seq)
    if L == target_len: return seq
    if L == 0: return np.zeros((target_len, seq.shape[1]))
    x_old = np.linspace(0, L-1, L)
    x_new = np.linspace(0, L-1, target_len)
    resampled = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    for j in range(seq.shape[1]):
        resampled[:, j] = np.interp(x_new, x_old, seq[:, j])
    return resampled

def extract_features(frames, alpha):
    frames = ema_smooth(np.array(frames), alpha)
    norm = np.array([torso_normalize(f) for f in frames])
    
    wrist_height = (norm[:, 15, 1] - norm[:, 11, 1]).reshape(-1, 1)
    elbow_flare = np.abs(norm[:, 13, 0] - norm[:, 11, 0]).reshape(-1, 1)
    geo_feats = norm[:, [13, 15], 1:3].reshape(len(norm), -1) 
    
    feat = np.concatenate([wrist_height, elbow_flare, geo_feats], axis=1)
    return resample(feat, 64)

class BicepCurlEngine(BaseWorkoutEngine):
    def initialize(self) -> None:
        model_path = self.config.get("model_path")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.is_quantized = self.input_details[0]['dtype'] == np.int8
        if self.is_quantized:
            self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
            self.output_scale, self.output_zero_point = self.output_details[0]['quantization']
            
        self.labels = ['partial_curl', 'perfect', 'wider_elbow']
        self.rep_buffer = []
        self.is_recording = False
        self.smooth_alpha = 0.7

    def process_frame(self, landmarks: list) -> FrameResult:
        lms = np.array(landmarks)[:, :3] # We only need x, y, z
        
        s = np.array([lms[11][0], lms[11][1]])
        e = np.array([lms[13][0], lms[13][1]])
        w_c = np.array([lms[15][0], lms[15][1]])
        
        radians = np.arctan2(w_c[1]-e[1], w_c[0]-e[0]) - np.arctan2(s[1]-e[1], s[0]-e[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0: angle = 360 - angle
        
        feedback = FeedbackType.NONE
        conf = 0.0
        
        if angle < 160 and not self.is_recording:
            self.is_recording = True
            self.rep_buffer = []
            
        if self.is_recording:
            self.rep_buffer.append(lms)
            
            if angle > 165 and len(self.rep_buffer) > 10:
                self.is_recording = False
                
                feats = extract_features(self.rep_buffer, self.smooth_alpha)
                probs = self._predict(feats)
                predicted_idx = np.argmax(probs)
                conf = float(probs[predicted_idx])
                label = self.labels[predicted_idx]
                
                if label == 'perfect':
                    self.rep_count_internal += 1
                    feedback = FeedbackType.PERFECT
                elif label == 'wider_elbow':
                    feedback = FeedbackType.WIDER_ELBOW
                else:
                    feedback = FeedbackType.PARTIAL_CURL

        return FrameResult(
            rep_count=self.rep_count_internal,
            feedback=feedback,
            confidence=conf,
            is_recording=self.is_recording,
            details={"angle": float(angle)}
        )
        
    def _predict(self, input_data):
        input_tensor = np.expand_dims(input_data, axis=0)
        
        if self.is_quantized:
            input_tensor = input_tensor / self.input_scale + self.input_zero_point
            input_tensor = input_tensor.astype(np.int8)
        else:
            input_tensor = input_tensor.astype(np.float32)
            
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        if self.is_quantized:
            output_data = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale
            
        exp_output = np.exp(output_data - np.max(output_data))
        probabilities = exp_output / exp_output.sum()
        return probabilities[0]
