import time
import cv2
import numpy as np
from typing import Tuple, Dict, List
from ultralytics import YOLO

exercise_plan = {
    "biceps_curl": (20, 20, 20, 1),
}

# YOLO Pose Keypoints (17 keypoints)
# 0: nose, 1: eyes(L), 2: eyes(R), 3: ears(L), 4: ears(R)
# 5: shoulders(L), 6: shoulders(R)
# 7: elbows(L), 8: elbows(R)
# 9: wrists(L), 10: wrists(R)
# 11: hips(L), 12: hips(R)
# 13: knees(L), 14: knees(R)
# 15: ankles(L), 16: ankles(R)

class DefaultFeedback:
    """Fallback feedback handler (simple print)."""
    def give_feedback(self, exercise_name: str, issue: str):
        print(f"[Feedback] {exercise_name}: {issue}")

class Workout:
    def __init__(self, video_path=1, visual=True,
                 model_path="yolo11n-pose.pt", feedback_handler=None, 
                 rest_between_sets=60, rest_between_exercises=120):
        self.video_path = video_path
        self.visual = visual
        self.model = YOLO(model_path)
        self.feedback_handler = feedback_handler if feedback_handler else DefaultFeedback()
        self.rest_between_sets = rest_between_sets
        self.rest_between_exercises = rest_between_exercises
        self.cap = None
        
        # Person detection readiness controls
        self.detection_confirmation_frames = 5
        self._detected_frames = 0
        self.person_confirmed = False
        
        # Biceps curl angle ranges
        self.biceps_curl_angle_range = (40, 80)  # elbow flexion range
        self.biceps_extended_angle_range = (160, 180)  # full extension
        
        # Alternating arm tracking
        self.active_arm = None  # 'left' or 'right'
        self.last_counted_arm = None  # Track which arm was last counted
        self.shoulder_reference_y = None  # Reference Y position of shoulders

    # -------------------- Geometry Utilities --------------------
    def angle_between(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculate angle ABC (angle at point B)"""
        ba = a - b
        bc = c - b
        dot = np.dot(ba, bc)
        norm = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norm == 0:
            return 0.0
        cos_angle = np.clip(dot / norm, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Euclidean distance between two points"""
        return np.linalg.norm(p1 - p2)
    
    def point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, 
                               line_end: np.ndarray) -> float:
        """Calculate perpendicular distance from point to line"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return np.linalg.norm(point_vec)
        
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        proj_point = line_start + proj_length * line_unitvec
        return np.linalg.norm(point - proj_point)

    # -------------------- Landmarks --------------------
    def _get_landmarks(self, results) -> Dict[str, Tuple[float, float]]:
        """Extract landmarks with confidence checking"""
        lm = {}
        if not results or len(results) == 0:
            return lm
        
        try:
            res_conf = results[0].keypoints.conf.cpu().numpy()
        except Exception:
            return lm
        
        finalres = res_conf[0] if res_conf.shape[0] > 0 else []
        
        # Critical keypoints for biceps curl
        critical_keypoints = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        if len(finalres) != 0:
            for i in critical_keypoints:
                if finalres[i] < 0.70:
                    return lm
        
        kpts = results[0].keypoints.xy.cpu().numpy()
        if kpts.shape[0] == 0:
            return lm
        
        kp = kpts[0]
        for idx, (x, y) in enumerate(kp):
            lm[str(idx)] = (float(x), float(y))
        
        return lm

    def _get_point(self, lm: Dict, idx: int) -> np.ndarray:
        """Get point coordinates or return None"""
        if str(idx) not in lm:
            return None
        return np.array(lm[str(idx)], dtype=np.float32)

    # -------------------- Drawing Utilities --------------------
    def _draw_line(self, frame: np.ndarray, p1: np.ndarray, p2: np.ndarray, 
                   color: Tuple[int, int, int], thickness: int = 2, label: str = ""):
        """Draw line between two points"""
        if p1 is None or p2 is None:
            return
        pt1 = tuple(map(int, p1))
        pt2 = tuple(map(int, p2))
        cv2.line(frame, pt1, pt2, color, thickness)
        if label:
            mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.putText(frame, label, (mid[0] + 5, mid[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_angle_at_point(self, frame: np.ndarray, point: np.ndarray, angle: float, 
                            label: str, color: Tuple[int, int, int]):
        """Draw angle value at a specific point with clear visualization"""
        if point is None:
            return
        pt = tuple(map(int, point))
        text = f"{label}: {angle:.1f}°"
        
        # Draw circle at joint
        cv2.circle(frame, pt, 10, color, 3)
        cv2.circle(frame, pt, 6, (255, 255, 255), 2)
        
        # Draw angle text with background for clarity
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Position text near the joint
        text_x = pt[0] + 15
        text_y = pt[1] - 15
        
        # Draw background rectangle for text
        padding = 5
        cv2.rectangle(frame, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     color, -1)
        
        # Draw text
        cv2.putText(frame, text, (text_x, text_y),
                   font, font_scale, (255, 255, 255), thickness)
    
    def _print_all_angles(self, lm: Dict):
        """Print all presented angles to terminal"""
        if not lm:
            return
        
        angles_data = {}
        
        # LEFT SIDE ANGLES
        shoulder_l = self._get_point(lm, 5)
        elbow_l = self._get_point(lm, 7)
        wrist_l = self._get_point(lm, 9)
        hip_l = self._get_point(lm, 11)
        ankle_l = self._get_point(lm, 15)
        
        # LEFT ELBOW ANGLE
        if all(p is not None for p in [shoulder_l, elbow_l, wrist_l]):
            angle = self.angle_between(shoulder_l, elbow_l, wrist_l)
            angles_data["L-Elbow (shoulder-elbow-wrist)"] = angle
        
        # LEFT SHOULDER ANGLE
        if all(p is not None for p in [wrist_l, shoulder_l, hip_l]):
            angle = self.angle_between(wrist_l, shoulder_l, hip_l)
            angles_data["L-Shoulder (wrist-shoulder-hip)"] = angle
        
        # RIGHT SIDE ANGLES
        shoulder_r = self._get_point(lm, 6)
        elbow_r = self._get_point(lm, 8)
        wrist_r = self._get_point(lm, 10)
        hip_r = self._get_point(lm, 12)
        ankle_r = self._get_point(lm, 16)
        
        # RIGHT ELBOW ANGLE
        if all(p is not None for p in [shoulder_r, elbow_r, wrist_r]):
            angle = self.angle_between(shoulder_r, elbow_r, wrist_r)
            angles_data["R-Elbow (shoulder-elbow-wrist)"] = angle
        
        # RIGHT SHOULDER ANGLE
        if all(p is not None for p in [wrist_r, shoulder_r, hip_r]):
            angle = self.angle_between(wrist_r, shoulder_r, hip_r)
            angles_data["R-Shoulder (wrist-shoulder-hip)"] = angle
        
        # Print formatted output
        print("\n" + "="*60)
        print(f"{'ANGLE MEASUREMENTS':^60}")
        print("="*60)
        for angle_name, angle_value in angles_data.items():
            print(f"{angle_name:.<45} {angle_value:>6.2f}°")
        print("="*60 + "\n")

    def _draw_biceps_curl_overlays(self, frame: np.ndarray, lm: Dict):
        """Draw validation lines for biceps curl"""
        shoulder_l = self._get_point(lm, 5)
        shoulder_r = self._get_point(lm, 6)
        elbow_l = self._get_point(lm, 7)
        elbow_r = self._get_point(lm, 8)
        wrist_l = self._get_point(lm, 9)
        wrist_r = self._get_point(lm, 10)
        hip_l = self._get_point(lm, 11)
        hip_r = self._get_point(lm, 12)
        ankle_l = self._get_point(lm, 15)
        ankle_r = self._get_point(lm, 16)
        
        # 1. Standing posture line (shoulder-hip-ankle)
        if all(p is not None for p in [shoulder_l, shoulder_r, hip_l, hip_r, ankle_l, ankle_r]):
            shoulder_mid = (shoulder_l + shoulder_r) / 2
            hip_mid = (hip_l + hip_r) / 2
            ankle_mid = (ankle_l + ankle_r) / 2
            
            self._draw_line(frame, shoulder_mid, hip_mid, (0, 255, 0), 3)
            self._draw_line(frame, hip_mid, ankle_mid, (0, 255, 0), 3, "Posture")
        
        # 2. Draw elbow angles - only for active arm
        if self.active_arm == 'left':
            if all(p is not None for p in [shoulder_l, elbow_l, wrist_l]):
                angle_l = self.angle_between(wrist_l, elbow_l, shoulder_l)
                self._draw_angle_at_point(frame, elbow_l, angle_l, "L-Curl", (255, 0, 0))
                cv2.line(frame, tuple(map(int, wrist_l)), tuple(map(int, elbow_l)), (255, 100, 100), 3)
                cv2.line(frame, tuple(map(int, elbow_l)), tuple(map(int, shoulder_l)), (255, 100, 100), 3)
        elif self.active_arm == 'right':
            if all(p is not None for p in [shoulder_r, elbow_r, wrist_r]):
                angle_r = self.angle_between(wrist_r, elbow_r, shoulder_r)
                self._draw_angle_at_point(frame, elbow_r, angle_r, "R-Curl", (0, 0, 255))
                cv2.line(frame, tuple(map(int, wrist_r)), tuple(map(int, elbow_r)), (100, 100, 255), 3)
                cv2.line(frame, tuple(map(int, elbow_r)), tuple(map(int, shoulder_r)), (100, 100, 255), 3)
        
        # 3. Draw elbow-hip distance reference
        if all(p is not None for p in [elbow_l, elbow_r, hip_l, hip_r]):
            elbow_avg = (elbow_l + elbow_r) / 2
            hip_avg = (hip_l + hip_r) / 2
            self._draw_line(frame, elbow_avg, hip_avg, (255, 255, 0), 2, "Elbow-Hip")
        
        # 4. Joint markers
        joints = [
            (shoulder_l, "SL", (100, 100, 255)),
            (shoulder_r, "SR", (100, 100, 255)),
            (elbow_l, "EL", (255, 100, 100)),
            (elbow_r, "ER", (255, 100, 100)),
            (wrist_l, "WL", (100, 255, 100)),
            (wrist_r, "WR", (100, 255, 100)),
            (hip_l, "HL", (100, 255, 255)),
            (hip_r, "HR", (100, 255, 255)),
            (ankle_l, "AL", (255, 100, 255)),
            (ankle_r, "AR", (255, 100, 255)),
        ]
        
        for joint, label, color in joints:
            if joint is not None:
                pt = tuple(map(int, joint))
                cv2.circle(frame, pt, 6, color, -1)
                cv2.circle(frame, pt, 8, (255, 255, 255), 2)

    def _draw_validation_overlays(self, frame: np.ndarray, lm: Dict, exercise_name: str = "biceps_curl"):
        """Draw all validation lines, markers, and angles on frame"""
        self._draw_biceps_curl_overlays(frame, lm)

    # -------------------- Biceps Curl Validation --------------------
    def _check_biceps_elbow_angle(self, lm: Dict) -> Tuple[float, bool, str]:
        """Check biceps curl elbow angle with alternating arm tracking"""
        shoulder_l = self._get_point(lm, 5)
        shoulder_r = self._get_point(lm, 6)
        elbow_l = self._get_point(lm, 7)
        elbow_r = self._get_point(lm, 8)
        wrist_l = self._get_point(lm, 9)
        wrist_r = self._get_point(lm, 10)
        
        if any(p is None for p in [shoulder_l, elbow_l, wrist_l, shoulder_r, elbow_r, wrist_r]):
            return 0.0, False, "Missing arm landmarks"
        
        # Calculate angles for both arms
        angle_l = self.angle_between(wrist_l, elbow_l, shoulder_l)
        angle_r = self.angle_between(wrist_r, elbow_r, shoulder_r)
        
        # Determine which arm is actively curling (angle < 50 degrees)
        left_curling = angle_l < 50
        right_curling = angle_r < 50
        
        # Initialize active arm on first curl
        if self.active_arm is None:
            if left_curling:
                self.active_arm = 'left'
            elif right_curling:
                self.active_arm = 'right'
            else:
                return 0.0, False, "Start curling"
        
        # Check if we need to switch arms
        if self.active_arm == 'left':
            if self.last_counted_arm == 'left' and right_curling and not left_curling:
                self.active_arm = 'right'
            current_angle = angle_l
        else:  # active_arm == 'right'
            if self.last_counted_arm == 'right' and left_curling and not right_curling:
                self.active_arm = 'left'
            current_angle = angle_r
        
        # Validate the active arm's curl
        min_angle, max_angle = self.biceps_curl_angle_range
        
        if current_angle < min_angle:
            self.feedback_handler.give_feedback("biceps_curl", f"Lower the weight - curl too high ({self.active_arm})")
            self.last_counted_arm = self.active_arm
            return current_angle, True, f"Valid curl - {self.active_arm}"
        elif current_angle > max_angle:
            self.feedback_handler.give_feedback("biceps_curl", f"Curl higher - bring weight up ({self.active_arm})")
            return current_angle, False, "Not enough curl"
        
        return current_angle, True, f"Valid curl - {self.active_arm}"
    
    def _check_biceps_extended_angle(self, lm: Dict) -> Tuple[float, bool, str]:
        """Check if arms are fully extended (160-180 degrees)"""
        shoulder_l = self._get_point(lm, 5)
        shoulder_r = self._get_point(lm, 6)
        elbow_l = self._get_point(lm, 7)
        elbow_r = self._get_point(lm, 8)
        wrist_l = self._get_point(lm, 9)
        wrist_r = self._get_point(lm, 10)
        
        if any(p is None for p in [shoulder_l, elbow_l, wrist_l, shoulder_r, elbow_r, wrist_r]):
            return 0.0, False, "Missing arm landmarks"
        
        angle_l = self.angle_between(wrist_l, elbow_l, shoulder_l)
        angle_r = self.angle_between(wrist_r, elbow_r, shoulder_r)
        avg_angle = (angle_l + angle_r) / 2
        
        min_angle, max_angle = self.biceps_extended_angle_range
        
        if avg_angle < min_angle:
            self.feedback_handler.give_feedback("biceps_curl", "Extend arms fully at the bottom")
            return avg_angle, False, "Not fully extended"
        
        return avg_angle, True, "Extended"
    
    def _check_elbow_hip_distance(self, lm: Dict) -> Tuple[float, bool, str]:
        """Check elbow stays close to hips (elbow shouldn't move forward)"""
        elbow_l = self._get_point(lm, 7)
        elbow_r = self._get_point(lm, 8)
        hip_l = self._get_point(lm, 11)
        hip_r = self._get_point(lm, 12)
        
        if any(p is None for p in [elbow_l, elbow_r, hip_l, hip_r]):
            return 0.0, False, "Missing body landmarks"
        
        
        # Distance in x-axis (horizontal distance)
        distance_l = abs(elbow_l[0] - hip_l[0])
        distance_r = abs(elbow_r[0] - hip_r[0])
        print('*'*50)
        print(distance_l)
        print(distance_r)
        print('*'*50)
        if distance_l > 45 or distance_r > 45:
            self.feedback_handler.give_feedback("biceps_curl", "Keep elbows back - don't swing forward")
            return distance_l, False, "Elbows too far"
        
        return distance_l, True, "Elbows stable"
    
    def _check_standing_posture(self, lm: Dict) -> Tuple[bool, str]:
        """Check full body is straight using shoulder-hip distance from reference"""
        shoulder_l = self._get_point(lm, 5)
        shoulder_r = self._get_point(lm, 6)
        hip_l = self._get_point(lm, 11)
        hip_r = self._get_point(lm, 12)
        
        if any(p is None for p in [shoulder_l, shoulder_r, hip_l, hip_r]):
            return False, "Missing body landmarks"
        
        if self.shoulder_reference_y is None:
            return False, "No shoulder reference set"
        
        # Calculate current shoulder and hip midpoints
        shoulder_mid = (shoulder_l + shoulder_r) / 2
        hip_mid = (hip_l + hip_r) / 2
        
        # Calculate current shoulder-hip distance
        current_shoulder_hip_dist = hip_mid[1] - shoulder_mid[1]
        
        # Calculate reference shoulder-hip distance (from initial standing position)
        reference_shoulder_hip_dist = hip_mid[1] - self.shoulder_reference_y
        
        # Calculate deviation from reference
        distance_deviation = abs(current_shoulder_hip_dist - reference_shoulder_hip_dist)
        
        print("@"*50)
        print(f"Reference shoulder Y: {self.shoulder_reference_y:.2f}")
        print(f"Current shoulder-hip distance: {current_shoulder_hip_dist:.2f}")
        print(f"Reference shoulder-hip distance: {reference_shoulder_hip_dist:.2f}")
        print(f"Deviation: {distance_deviation:.2f}")
        print("@"*50)
        
        max_deviation = 10  # +/- 10 pixels tolerance
        
        if distance_deviation > max_deviation:
            if current_shoulder_hip_dist < reference_shoulder_hip_dist:
                self.feedback_handler.give_feedback("biceps_curl", "Don't lean forward - stand upright")
            else:
                self.feedback_handler.give_feedback("biceps_curl", "Don't lean back - stand straight")
            return False, f"Leaning ({distance_deviation:.1f}px)"
        
        return True, "Straight posture"

    def _validate_biceps_curl_rep(self, lm: Dict) -> Tuple[bool, Dict]:
        """Comprehensive biceps curl validation"""
        checks = {}
        
        # Check elbow angle for curl
        checks["elbow_curl_angle"] = self._check_biceps_elbow_angle(lm)
        
        # Check elbow-hip distance
        checks["elbow_hip_distance"] = self._check_elbow_hip_distance(lm)
        
        # Check standing posture
        posture_check = self._check_standing_posture(lm)
        checks["standing_posture"] = (0, posture_check[0], posture_check[1])
        
        # All checks must pass for valid curl
        all_valid = all(check[1] for check in checks.values())
        return all_valid, checks

    # -------------------- Rest Period --------------------
    def rest_period(self, duration, numset, tarset, is_exercise_rest=False, exercise_name="biceps_curl"):
        label = "exercise" if is_exercise_rest else "set"
        end_time = time.time() + duration
        
        while time.time() < end_time:
            remaining = int(end_time - time.time()) + 1
            
            ret, frame = self.cap.read()
            if not ret:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            txt = f"Rest between {label}s: {remaining} sec"
            cv2.putText(frame, txt, (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 3)
            
            if label != "exercise":
                display_name = exercise_name.replace('_', ' ').title()
                txt = f"{display_name} | Set {numset}/{tarset} and remaining sets: {tarset - numset}"
                cv2.putText(frame, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 255, 0), 2)
            
            cv2.imshow("Workout (Press Q to quit)", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    # -------------------- Training Loop --------------------
    def train(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not opened")
        
        print("Looking for person... please stand in front of the camera.")
        while not self.person_confirmed:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Camera disconnected during person detection")
            
            results = self.model(frame, verbose=False)
            lm = self._get_landmarks(results)
            
            if lm:
                self._detected_frames += 1
            else:
                self._detected_frames = 0
            
            if self._detected_frames >= self.detection_confirmation_frames:
                self.person_confirmed = True
                # SET SHOULDER REFERENCE
                shoulder_l = self._get_point(lm, 5)
                shoulder_r = self._get_point(lm, 6)
                shoulder_mid = (shoulder_l + shoulder_r) / 2
                self.shoulder_reference_y = shoulder_mid[1]
                
                print(f"Person detected reliably — starting workout.")
                print(f"Shoulder reference Y set to: {self.shoulder_reference_y:.2f}")
                cv2.putText(frame, "Person confirmed. Starting...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if self.visual:
                    cv2.imshow("Workout (Press Q to quit)", frame)
                    cv2.waitKey(500)
                break
            
            cv2.putText(frame, f"Waiting for person... ({self._detected_frames}/{self.detection_confirmation_frames})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.visual:
                cv2.imshow("Workout (Press Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User aborted before detection.")
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return
        
        # Main exercise loop
        phase = "up"
        for ex_idx, (exercise_name, rep_groups) in enumerate(exercise_plan.items()):
            print(f"\n{'='*60}")
            print(f"Starting {exercise_name.upper().replace('_', ' ')}...")
            print(f"{'='*60}\n")
            
            for group_idx, target_reps in enumerate(rep_groups):
                if group_idx == len(rep_groups) - 1:
                    break
                
                rep_count = 0
                print(f"Set {group_idx + 1}: {target_reps} reps")
                
                while rep_count < target_reps:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Camera disconnected")
                        break
                    
                    results = self.model(frame, verbose=False)
                    try:
                        frame = results[0].plot(kpt_line=True, kpt_radius=5)
                    except Exception:
                        pass
                    
                    lm = self._get_landmarks(results)
                    
                    # Draw validation overlays based on exercise type
                    self._draw_validation_overlays(frame, lm, exercise_name)
                    
                    # Print all angles to terminal
                    if lm:
                        self._print_all_angles(lm)
                    
                    # Biceps curl validation
                    if lm:
                        # Check if arms are extended (bottom position)
                        extension_angle, is_extended, _ = self._check_biceps_extended_angle(lm)
                        
                        if phase == "up":
                            # At top position, waiting to lower
                            if is_extended:
                                phase = "down"
                                cv2.putText(frame, "PHASE: CURL UP", (10, 150),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        
                        elif phase == "down":
                            # At bottom position, check if curling up
                            is_valid, checks = self._validate_biceps_curl_rep(lm)
                            curl_angle = checks["elbow_curl_angle"][0]
                            
                            if is_valid and curl_angle <= 50:
                                rep_count += 1
                                print(f"Rep {rep_count}/{target_reps}")
                                phase = "up"
                                cv2.putText(frame, f"REP {rep_count} COMPLETE!", (10, 150),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Display info
                    txt = f"Biceps Curl | Rep {rep_count}/{target_reps}"
                    cv2.putText(frame, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, (0, 255, 0), 2)
                    
                    if lm:
                        _, checks = self._validate_biceps_curl_rep(lm)
                        y_offset = 90
                        for check_name, (value, is_valid, msg) in checks.items():
                            color = (0, 255, 0) if is_valid else (0, 0, 255)
                            status = "✓" if is_valid else "✗"
                            display_value = f"{value:.1f}" if value > 0 else ""
                            cv2.putText(frame, f"{status} {check_name}: {msg} {display_value}", 
                                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            y_offset += 20
                    if self.visual:
                        cv2.imshow("Workout (Press Q to quit)", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            rep_count = target_reps
                            break
                
                l = len(rep_groups)
                if group_idx < l - 1:
                    self.rest_period(self.rest_between_sets, group_idx + 1, l - 1, 
                                   is_exercise_rest=False, exercise_name=exercise_name)
            
            if ex_idx < len(exercise_plan) - 1:
                self.rest_period(self.rest_between_exercises, is_exercise_rest=True)
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Training session complete!")

# -------------------- Example --------------------
if __name__ == "__main__":
    w = Workout(video_path=1,
                visual=True,
                model_path="yolo11n-pose.pt")
    w.train()
