import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# --- CONFIGURATION ---
FILE_NAME = 'last12_cleaned_partial_curl_training_data.csv'
LABEL = 'wider_elbow'  # You can change this for different exercises
MIN_ANGLE = 165         # Angle to trigger start/stop of recording

# --- INITIALIZE MEDIAPIPE ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def calculate_angle(a, b, c):
    """Calculates angle at point b given three points a, b, c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_start_info(filename):
    """Checks the CSV and returns the next rep_id to use."""
    # Define headers
    headers = ['rep_id']
    for i in range(33):
        headers.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
    headers.append('label')

    if not os.path.exists(filename) or os.stat(filename).st_size == 0:
        with open(filename, mode='w', newline='') as f:
            csv.writer(f).writerow(headers)
        return 0 
    else:
        with open(filename, mode='r') as f:
            rows = list(csv.reader(f))
            if len(rows) <= 1: # Only headers exist
                return 0
            # Read the rep_id from the last row in the file
            try:
                last_rep_id = int(rows[-1][0])
                return last_rep_id
            except (IndexError, ValueError):
                return 0

# --- PREPARE STATE ---
current_rep_id = get_start_info(FILE_NAME)
recording_active = False
frame_buffer = []

print(f"File: {FILE_NAME} initialized.")
print(f"Resuming from Rep ID: {current_rep_id}")

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    
    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 1. Get coordinates for Left Arm (Shoulder 11, Elbow 13, Wrist 15)
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        
        angle = calculate_angle(shoulder, elbow, wrist)

        # 2. Logic for recording sequences
        # START: Angle drops below 175
        if angle < MIN_ANGLE and not recording_active:
            recording_active = True
            current_rep_id += 1
            frame_buffer = [] # Reset buffer for the new rep
            print(f"Started Rep {current_rep_id}")

        if recording_active:
            # Flatten all 33 landmarks [x, y, z, v]
            row = [current_rep_id]
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            row.append(LABEL)
            
            frame_buffer.append(row)

            # STOP: Angle returns to 175+
            if angle >= MIN_ANGLE:
                recording_active = False
                # Save the entire sequence of frames for this rep
                with open(FILE_NAME, mode='a', newline='') as f:
                    csv.writer(f).writerows(frame_buffer)
                print(f"Saved Rep {current_rep_id} ({len(frame_buffer)} frames)")

        # 3. Visualization
        # Draw skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Status Box
        cv2.rectangle(image, (0,0), (250, 100), (245, 117, 16), -1)
        
        cv2.putText(image, 'ANGLE', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(int(angle)), (10,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        
        cv2.putText(image, 'REP ID', (15,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(current_rep_id), (10,90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        if recording_active:
            cv2.circle(image, (600, 40), 15, (0, 0, 255), -1)
            cv2.putText(image, "REC", (540, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('MediaPipe Data Logger', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()