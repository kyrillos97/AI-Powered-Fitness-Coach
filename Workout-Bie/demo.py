import cv2
import mediapipe as mp
import numpy as np
import tensorflow.lite as tflite
import face_recognition
import json
import os
import math

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = 'bicep_coach_3class.tflite'
LABELS = ['partial_curl', 'perfect', 'wider_elbow']  # Alphabetical order (matches LabelEncoder)
DB_FILE = "user_db.json"
MATCH_THRESHOLD = 0.5  # Strict match
TARGET_FRAMES = 64
SMOOTH_ALPHA = 0.7

# State Machine Constants
STATE_SEARCHING = 0    # Waiting for user to come close
STATE_VERIFIED = 1     # User recognized, waiting for them to step back
STATE_WORKOUT = 2      # Full body visible, counting reps

# =========================
# HELPER CLASSES
# =========================
class SmartTracker:
    """
    Tracks the user by their HIP center. 
    Once we know who the user is, we just follow their hips.
    """
    def __init__(self):
        self.active = False
        self.last_hip_x = 0.5
        self.last_hip_y = 0.5
        self.missed_frames = 0
    
    def lock_on(self, landmarks):
        """Initial lock when face is verified"""
        self.active = True
        self.missed_frames = 0
        self.update(landmarks)
        
    def update(self, landmarks):
        """Update position based on new frame landmarks"""
        # Midpoint of hips (Landmarks 23 and 24)
        hip_x = (landmarks[23].x + landmarks[24].x) / 2
        hip_y = (landmarks[23].y + landmarks[24].y) / 2
        self.last_hip_x = hip_x
        self.last_hip_y = hip_y
        self.missed_frames = 0

    def is_user(self, landmarks, threshold=0.2):
        """
        Checks if the skeleton in the current frame is close enough 
        to where the user was in the last frame.
        """
        hip_x = (landmarks[23].x + landmarks[24].x) / 2
        hip_y = (landmarks[23].y + landmarks[24].y) / 2
        
        dist = math.hypot(hip_x - self.last_hip_x, hip_y - self.last_hip_y)
        return dist < threshold

# =========================
# WORKOUT LOGIC (UPDATED TO MATCH TRAINING)
# =========================
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

def extract_features(frames):
    """UPDATED: Now matches training features exactly"""
    frames = ema_smooth(np.array(frames), SMOOTH_ALPHA)
    norm = np.array([torso_normalize(f) for f in frames])
    
    # Feature 1: WRIST HEIGHT (For Partial Curls)
    wrist_height = (norm[:, 15, 1] - norm[:, 11, 1]).reshape(-1, 1)
    
    # Feature 2: ELBOW FLARE (For Wider Elbows) - FIXED!
    elbow_flare = np.abs(norm[:, 13, 0] - norm[:, 11, 0]).reshape(-1, 1)
    
    # Feature 3: ELBOW/WRIST DYNAMICS (Y and Z coordinates)
    geo_feats = norm[:, [13, 15], 1:3].reshape(len(norm), -1) 
    
    # Combined: [Height, Flare, Elbow_Y, Elbow_Z, Wrist_Y, Wrist_Z]
    feat = np.concatenate([wrist_height, elbow_flare, geo_feats], axis=1)
    
    return resample(feat, TARGET_FRAMES)

class TFLiteClassifier:
    def __init__(self, model_path):
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.is_quantized = self.input_details[0]['dtype'] == np.int8
            if self.is_quantized:
                self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
                self.output_scale, self.output_zero_point = self.output_details[0]['quantization']
            self.model_loaded = True
            print(f"Model loaded successfully. Quantized: {self.is_quantized}")
            print(f"Input shape: {self.input_details[0]['shape']}")
            print(f"Expected features: {self.input_details[0]['shape'][1]}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
    
    def predict(self, input_data):
        if not self.model_loaded: 
            return np.array([0.0, 0.0, 0.0])
        
        # Debug: Check feature dimensions
        if input_data.shape != (TARGET_FRAMES, 6):
            print(f"WARNING: Input shape {input_data.shape} doesn't match expected (64, 6)")
        
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
        
        # Apply softmax to ensure probabilities sum to 1
        exp_output = np.exp(output_data - np.max(output_data))
        probabilities = exp_output / exp_output.sum()
        
        return probabilities[0]

# =========================
# MAIN APP
# =========================
def main():
    # Setup
    classifier = TFLiteClassifier(MODEL_PATH)
    mp_pose = mp.solutions.pose
    pose_tracker = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    user_tracker = SmartTracker()

    cap = cv2.VideoCapture(1)  # Changed to 0 for default camera
    
    # State Variables
    app_state = STATE_SEARCHING
    enrolled_encoding = None
    captured_encodings = []
    
    # Workout Variables
    rep_buffer = []
    is_recording = False
    last_prediction = "READY"
    rep_count = 0  # Only count perfect reps
    feedback_text = "Step closer to camera"
    feedback_color = (0, 255, 255)
    debug_info = ""  # For debugging
    
    # Load enrolled user if exists
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f:
                data = json.load(f)
                if "User" in data:
                    enrolled_encoding = np.array(data["User"])
                    print("User loaded from DB.")
        except Exception as e:
            print(f"Error loading DB: {e}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -----------------------------
        # STATE 0: ENROLL / SEARCHING
        # -----------------------------
        if app_state == STATE_SEARCHING:
            if enrolled_encoding is None:
                # ENROLL MODE
                feedback_text = f"Taking Photos: {len(captured_encodings)}/3 (Press 'C')"
                feedback_color = (255, 200, 0)
                
                # Use FULL SIZE frame for detection
                face_locs = face_recognition.face_locations(rgb_frame)
                
                for (top, right, bottom, left) in face_locs:
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                
                if face_locs:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('c'):
                        try:
                            enc = face_recognition.face_encodings(rgb_frame, face_locs)[0]
                            captured_encodings.append(enc)
                            if len(captured_encodings) >= 3:
                                enrolled_encoding = np.mean(captured_encodings, axis=0)
                                # Save to DB
                                with open(DB_FILE, 'w') as f:
                                    json.dump({"User": enrolled_encoding.tolist()}, f)
                                print("Enrolled!")
                        except Exception as e:
                            print(f"Error capturing face: {e}")
            else:
                # VERIFICATION MODE
                feedback_text = "Come closer to verify..."
                feedback_color = (0, 165, 255)
                
                # Check for faces (Full resolution for accuracy)
                face_locs = face_recognition.face_locations(rgb_frame)
                if face_locs:
                    face_encs = face_recognition.face_encodings(rgb_frame, face_locs)
                    for i, enc in enumerate(face_encs):
                        dist = face_recognition.face_distance([enrolled_encoding], enc)[0]
                        top, right, bottom, left = face_locs[i]
                        
                        # Draw Box
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                        
                        if dist < MATCH_THRESHOLD:
                            # SUCCESS: Handover to Pose Tracker
                            app_state = STATE_VERIFIED
                            feedback_text = "Verified! Don't move..."
                            
                            # We need to initialize the SmartTracker with the skeleton 
                            # that matches this face box.
                            results = pose_tracker.process(rgb_frame)
                            if results.pose_landmarks:
                                lms = results.pose_landmarks.landmark
                                nose_x = int(lms[0].x * w)
                                nose_y = int(lms[0].y * h)
                                
                                # Verify skeleton aligns with face
                                if left < nose_x < right and top < nose_y < bottom:
                                    user_tracker.lock_on(lms)
                                    break

        # -----------------------------
        # STATE 1 & 2: TRACKING & WORKOUT
        # -----------------------------
        elif app_state == STATE_VERIFIED or app_state == STATE_WORKOUT:
            # We NO LONGER run face_recognition here (Saves speed & works at distance)
            
            results = pose_tracker.process(rgb_frame)
            
            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                
                # 1. Check if this is our user (Continuity Check)
                if user_tracker.is_user(lms):
                    user_tracker.update(lms) # Update position
                    
                    # Draw Skeleton
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # 2. Check Visibility (Are ankles visible?)
                    left_ankle_vis = lms[27].visibility
                    right_ankle_vis = lms[28].visibility
                    
                    if app_state == STATE_VERIFIED:
                        if left_ankle_vis > 0.5 and right_ankle_vis > 0.5:
                            app_state = STATE_WORKOUT
                            feedback_text = "Target Locked. GO!"
                            feedback_color = (0, 255, 0)
                        else:
                            feedback_text = "Verified! Step BACK until I see your feet."
                            feedback_color = (0, 255, 255)
                    
                    # 3. WORKOUT LOGIC
                    if app_state == STATE_WORKOUT:
                        # Angle Logic
                        s = np.array([lms[11].x, lms[11].y])    # Left shoulder
                        e = np.array([lms[13].x, lms[13].y])    # Left elbow
                        w_c = np.array([lms[15].x, lms[15].y])  # Left wrist
                        
                        radians = np.arctan2(w_c[1]-e[1], w_c[0]-e[0]) - np.arctan2(s[1]-e[1], s[0]-e[0])
                        angle = np.abs(radians * 180.0 / np.pi)
                        if angle > 180.0: angle = 360 - angle
                        
                        # Visualize angle
                        cv2.putText(frame, f"Angle: {angle:.1f}°", (20, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Feedback loop
                        frame_lms = [[lm.x, lm.y, lm.z] for lm in lms]
                        if angle < 160 and not is_recording:
                            is_recording = True
                            rep_buffer = []
                            print("Started recording rep...")
                        
                        if is_recording:
                            rep_buffer.append(np.array(frame_lms))
                            
                            # Visualize recording state
                            cv2.circle(frame, (30, h - 30), 15, (0, 0, 255), -1)
                            cv2.putText(frame, "RECORDING", (55, h - 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            if angle > 165 and len(rep_buffer) > 10:
                                is_recording = False
                                print(f"Processing rep with {len(rep_buffer)} frames...")
                                
                                try:
                                    feats = extract_features(rep_buffer)
                                    print(f"Extracted features shape: {feats.shape}")
                                    probs = classifier.predict(feats)
                                    predicted_idx = np.argmax(probs)
                                    last_prediction = LABELS[predicted_idx].upper().replace("_", " ")
                                    
                                    # Debug: Print probabilities
                                    debug_info = f"P:{probs[0]:.2f} T:{probs[1]:.2f} W:{probs[2]:.2f}"
                                    print(f"Probabilities: {debug_info}")
                                    print(f"Predicted: {LABELS[predicted_idx]}")
                                    
                                    # Only count if perfect
                                    if LABELS[predicted_idx] == 'perfect':
                                        rep_count += 1
                                        feedback_text = "GOOD REP!"
                                        feedback_color = (0, 255, 0)
                                    elif LABELS[predicted_idx] == 'wider_elbow':
                                        feedback_text = "KEEP ELBOWS IN!"
                                        feedback_color = (0, 165, 255)
                                    else:  # partial_curl
                                        feedback_text = "FULL RANGE OF MOTION!"
                                        feedback_color = (255, 165, 0)
                                    
                                except Exception as e:
                                    print(f"Error predicting: {e}")
                                    last_prediction = "ERROR"

                else:
                    # Skeleton found, but it's not the user (too far from last position)
                    cv2.putText(frame, "Ignoring Background Person", (20, h-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            else:
                # Lost Tracking
                user_tracker.missed_frames += 1
                if user_tracker.missed_frames > 30: # Lost for 1 second
                    app_state = STATE_SEARCHING # Force re-verification
                    feedback_text = "Lost you. Come closer."
                    feedback_color = (0, 0, 255)

        # -----------------------------
        # UI OVERLAY
        # -----------------------------
        # Top Bar
        cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
        
        # Status Text
        cv2.putText(frame, feedback_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2)
        
        # Rep Counter (Only in Workout Mode)
        if app_state == STATE_WORKOUT:
            cv2.putText(frame, f"REPS: {rep_count}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, last_prediction, (w - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show probabilities
            cv2.putText(frame, debug_info, (w - 250, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # State indicator
        state_names = ["ENROLL/VERIFY", "VERIFIED", "WORKOUT"]
        cv2.putText(frame, f"State: {state_names[app_state]}", (20, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('AI Trainer - Bicep Curl Form Checker', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # Reset rep count
            rep_count = 0
        elif key == ord('s'):  # Skip to workout state (for testing)
            if app_state == STATE_SEARCHING and enrolled_encoding is not None:
                app_state = STATE_WORKOUT
                feedback_text = "TEST MODE - GO!"
                feedback_color = (0, 255, 0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()