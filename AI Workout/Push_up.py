import time
import cv2
import numpy as np
from typing import Tuple, Dict, List
from ultralytics import YOLO

exercise_plan = {
    "push_up": (10, 2, 1, 1),
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
        
        # Angle error ranges (in degrees)
        self.elbow_angle_range = (35, 55)  # 45 ± 10
        self.hand_shoulder_hip_range = (90, 180)  # > 90
        self.chest_to_ground_range = (0, 10)  # ≤ 10 degrees

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
        
        # Critical keypoints for push-ups
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

    # -------------------- Validation Criteria --------------------
    def _check_feet_hands_alignment(self, lm: Dict) -> Tuple[bool, str]:
        """Check 1: Feet and hands on the same line (with tolerance)"""
        wrist_l = self._get_point(lm, 9)
        wrist_r = self._get_point(lm, 10)
        ankle_l = self._get_point(lm, 15)
        ankle_r = self._get_point(lm, 16)
        
        if any(p is None for p in [wrist_l, wrist_r, ankle_l, ankle_r]):
            return False, "Cannot detect hands or feet"
        
        # Use wrists and ankles to form alignment line
        wrist_avg = (wrist_l + wrist_r) / 2
        ankle_avg = (ankle_l + ankle_r) / 2
        
        # Check if body is roughly horizontal (small y-distance relative to x-distance)
        y_diff = abs(wrist_avg[1] - ankle_avg[1])
        x_diff = abs(wrist_avg[0] - ankle_avg[0])
        
        if x_diff == 0:
            return False, "Invalid body position"
        
        slope = y_diff / x_diff
        tolerance = 0.2  # 20% tolerance
        
        if slope > tolerance:
            self.feedback_handler.give_feedback("push_up", "Keep your body horizontal - hands and feet aligned")
            return False, "Not aligned"
        
        return True, "Aligned"

    def _check_back_straight(self, lm: Dict) -> Tuple[bool, str]:
        """Check 2: Back is straight (shoulder-hip-knee-ankle alignment)"""
        shoulder_l = self._get_point(lm, 5)
        shoulder_r = self._get_point(lm, 6)
        hip_l = self._get_point(lm, 11)
        hip_r = self._get_point(lm, 12)
        knee_l = self._get_point(lm, 13)
        knee_r = self._get_point(lm, 14)
        ankle_l = self._get_point(lm, 15)
        ankle_r = self._get_point(lm, 16)
        
        if any(p is None for p in [shoulder_l, shoulder_r, hip_l, hip_r, knee_l, knee_r, ankle_l, ankle_r]):
            return False, "Missing body landmarks"
        
        # Midpoints
        shoulder_mid = (shoulder_l + shoulder_r) / 2
        hip_mid = (hip_l + hip_r) / 2
        knee_mid = (knee_l + knee_r) / 2
        ankle_mid = (ankle_l + ankle_r) / 2
        
        # Check distances from the back line
        back_line_start = shoulder_mid
        back_line_end = ankle_mid
        
        hip_dist = self.point_to_line_distance(hip_mid, back_line_start, back_line_end)
        knee_dist = self.point_to_line_distance(knee_mid, back_line_start, back_line_end)
        
        max_deviation = 30  # pixels tolerance
        
        if hip_dist > max_deviation:
            if hip_mid[1] > shoulder_mid[1] + 20:  # sagging hips
                self.feedback_handler.give_feedback("push_up", "Don't sag your hips - keep core tight")
            else:
                self.feedback_handler.give_feedback("push_up", "Don't pike your hips - keep body straight")
            return False, "Hips misaligned"
        
        if knee_dist > max_deviation:
            self.feedback_handler.give_feedback("push_up", "Keep your legs straight - no bending")
            return False, "Knees misaligned"
        
        return True, "Straight"

    def _check_head_up(self, lm: Dict) -> Tuple[bool, str]:
        """Check 3: Head is up (eye-shoulder-hip alignment)"""
        nose = self._get_point(lm, 0)
        eye_l = self._get_point(lm, 1)
        eye_r = self._get_point(lm, 2)
        shoulder_l = self._get_point(lm, 5)
        shoulder_r = self._get_point(lm, 6)
        hip_l = self._get_point(lm, 11)
        hip_r = self._get_point(lm, 12)
        
        if any(p is None for p in [nose, shoulder_l, shoulder_r, hip_l, hip_r]):
            return False, "Missing head/body landmarks"
        
        shoulder_mid = (shoulder_l + shoulder_r) / 2
        hip_mid = (hip_l + hip_r) / 2
        
        # Head should be slightly ahead of shoulders (not down)
        head_to_shoulder_y = nose[1] - shoulder_mid[1]
        
        # If nose is significantly below shoulder, head is down
        if head_to_shoulder_y > 50:  # threshold in pixels
            self.feedback_handler.give_feedback("push_up", "Look forward - don't drop your head")
            return False, "Head down"
        
        return True, "Head up"

    def _check_elbow_angle(self, lm: Dict) -> Tuple[float, bool, str]:
        """Check 4.a: Elbow angle is ~45 degrees"""
        shoulder_l = self._get_point(lm, 5)
        shoulder_r = self._get_point(lm, 6)
        elbow_l = self._get_point(lm, 7)
        elbow_r = self._get_point(lm, 8)
        wrist_l = self._get_point(lm, 9)
        wrist_r = self._get_point(lm, 10)
        
        if any(p is None for p in [shoulder_l, elbow_l, wrist_l, shoulder_r, elbow_r, wrist_r]):
            return 0.0, False, "Missing arm landmarks"
        
        # Calculate angles for both arms
        angle_l = self.angle_between(shoulder_l, elbow_l, wrist_l)
        angle_r = self.angle_between(shoulder_r, elbow_r, wrist_r)
        avg_angle = (angle_l + angle_r) / 2
        
        min_angle, max_angle = self.elbow_angle_range
        
        if avg_angle < min_angle:
            self.feedback_handler.give_feedback("push_up", "Bend your elbows more - lower your body")
            return avg_angle, False, "Too straight"
        elif avg_angle > max_angle:
            self.feedback_handler.give_feedback("push_up", "Extend your elbows more")
            return avg_angle, False, "Too bent"
        
        return avg_angle, True, "Valid"

    def _check_hand_shoulder_hip_angle(self, lm: Dict) -> Tuple[float, bool, str]:
        """Check 4.b: Angle between hand, shoulder, hip > 90 degrees"""
        shoulder_l = self._get_point(lm, 5)
        shoulder_r = self._get_point(lm, 6)
        wrist_l = self._get_point(lm, 9)
        wrist_r = self._get_point(lm, 10)
        hip_l = self._get_point(lm, 11)
        hip_r = self._get_point(lm, 12)
        
        if any(p is None for p in [shoulder_l, wrist_l, hip_l, shoulder_r, wrist_r, hip_r]):
            return 0.0, False, "Missing body landmarks"
        
        # Calculate angles for both sides
        angle_l = self.angle_between(wrist_l, shoulder_l, hip_l)
        angle_r = self.angle_between(wrist_r, shoulder_r, hip_r)
        avg_angle = (angle_l + angle_r) / 2
        
        min_angle, max_angle = self.hand_shoulder_hip_range
        
        if avg_angle < min_angle:
            self.feedback_handler.give_feedback("push_up", "Widen your stance or bring hands closer to your body")
            return avg_angle, False, "Angle too small"
        
        return avg_angle, True, "Valid"

    def _check_chest_to_ground(self, lm: Dict) -> Tuple[float, bool, str]:
        """Check 4.c: Chest close to ground (shoulder-ankle-hand angle ≤ 10°)"""
        shoulder_l = self._get_point(lm, 5)
        shoulder_r = self._get_point(lm, 6)
        ankle_l = self._get_point(lm, 15)
        ankle_r = self._get_point(lm, 16)
        wrist_l = self._get_point(lm, 9)
        wrist_r = self._get_point(lm, 10)
        
        if any(p is None for p in [shoulder_l, ankle_l, wrist_l, shoulder_r, ankle_r, wrist_r]):
            return 0.0, False, "Missing landmarks"
        
        # Calculate angles for both sides
        angle_l = self.angle_between(ankle_l, shoulder_l, wrist_l)
        angle_r = self.angle_between(ankle_r, shoulder_r, wrist_r)
        avg_angle = (angle_l + angle_r) / 2
        
        min_angle, max_angle = self.chest_to_ground_range
        
        if avg_angle > max_angle:
            self.feedback_handler.give_feedback("push_up", "Go deeper - chest closer to ground")
            return avg_angle, False, "Not deep enough"
        
        return avg_angle, True, "Valid"

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
        """Draw angle value at a specific point"""
        if point is None:
            return
        pt = tuple(map(int, point))
        text = f"{label}: {angle:.1f}°"
        cv2.putText(frame, text, (pt[0] + 10, pt[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, pt, 8, color, 2)
    
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
        
        # LEFT CHEST ANGLE
        if all(p is not None for p in [ankle_l, shoulder_l, wrist_l]):
            angle = self.angle_between(ankle_l, shoulder_l, wrist_l)
            angles_data["L-Chest (ankle-shoulder-wrist)"] = angle
        
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
        
        # RIGHT CHEST ANGLE
        if all(p is not None for p in [ankle_r, shoulder_r, wrist_r]):
            angle = self.angle_between(ankle_r, shoulder_r, wrist_r)
            angles_data["R-Chest (ankle-shoulder-wrist)"] = angle
        
        # Print formatted output
        print("\n" + "="*60)
        print(f"{'ANGLE MEASUREMENTS':^60}")
        print("="*60)
        for angle_name, angle_value in angles_data.items():
            print(f"{angle_name:.<45} {angle_value:>6.2f}°")
        print("="*60 + "\n")

    def _draw_validation_overlays(self, frame: np.ndarray, lm: Dict):
        """Draw all validation lines, markers, and angles on frame"""
        # 1. Feet-hands alignment line
        wrist_l = self._get_point(lm, 9)
        wrist_r = self._get_point(lm, 10)
        ankle_l = self._get_point(lm, 15)
        ankle_r = self._get_point(lm, 16)
        
        if all(p is not None for p in [wrist_l, wrist_r, ankle_l, ankle_r]):
            wrist_avg = (wrist_l + wrist_r) / 2
            ankle_avg = (ankle_l + ankle_r) / 2
            self._draw_line(frame, wrist_avg, ankle_avg, (0, 255, 255), 2, "Alignment")
        
        # 2. Back line (shoulder-hip-knee-ankle)
        shoulder_l = self._get_point(lm, 5)
        shoulder_r = self._get_point(lm, 6)
        hip_l = self._get_point(lm, 11)
        hip_r = self._get_point(lm, 12)
        knee_l = self._get_point(lm, 13)
        knee_r = self._get_point(lm, 14)
        ankle_l = self._get_point(lm, 15)
        ankle_r = self._get_point(lm, 16)
        
        if all(p is not None for p in [shoulder_l, shoulder_r, hip_l, hip_r, knee_l, knee_r, ankle_l, ankle_r]):
            shoulder_mid = (shoulder_l + shoulder_r) / 2
            hip_mid = (hip_l + hip_r) / 2
            knee_mid = (knee_l + knee_r) / 2
            ankle_mid = (ankle_l + ankle_r) / 2
            
            self._draw_line(frame, shoulder_mid, hip_mid, (0, 255, 0), 2)
            self._draw_line(frame, hip_mid, knee_mid, (0, 255, 0), 2)
            self._draw_line(frame, knee_mid, ankle_mid, (0, 255, 0), 2, "Back Line")
        
        # 3. Draw all critical angles
        if lm:
            # LEFT SIDE ANGLES
            elbow_l = self._get_point(lm, 7)
            wrist_l = self._get_point(lm, 9)
            shoulder_l = self._get_point(lm, 5)
            hip_l = self._get_point(lm, 11)
            ankle_l = self._get_point(lm, 15)
            
            # LEFT ELBOW ANGLE (shoulder-elbow-wrist)
            if all(p is not None for p in [shoulder_l, elbow_l, wrist_l]):
                angle = self.angle_between(shoulder_l, elbow_l, wrist_l)
                self._draw_angle_at_point(frame, elbow_l, angle, "L-Elbow", (255, 0, 0))
                # Draw lines forming the angle
                cv2.line(frame, tuple(map(int, shoulder_l)), tuple(map(int, elbow_l)), (255, 0, 0), 1)
                cv2.line(frame, tuple(map(int, elbow_l)), tuple(map(int, wrist_l)), (255, 0, 0), 1)
            
            # LEFT SHOULDER ANGLE (wrist-shoulder-hip)
            if all(p is not None for p in [wrist_l, shoulder_l, hip_l]):
                angle = self.angle_between(wrist_l, shoulder_l, hip_l)
                self._draw_angle_at_point(frame, shoulder_l, angle, "L-Shoulder", (255, 100, 0))
                cv2.line(frame, tuple(map(int, wrist_l)), tuple(map(int, shoulder_l)), (255, 100, 0), 1)
                cv2.line(frame, tuple(map(int, shoulder_l)), tuple(map(int, hip_l)), (255, 100, 0), 1)
            
            # LEFT CHEST ANGLE (ankle-shoulder-wrist)
            if all(p is not None for p in [ankle_l, shoulder_l, wrist_l]):
                angle = self.angle_between(ankle_l, shoulder_l, wrist_l)
                self._draw_angle_at_point(frame, shoulder_l, angle, "L-Chest", (255, 255, 0))
                cv2.line(frame, tuple(map(int, ankle_l)), tuple(map(int, shoulder_l)), (255, 255, 0), 1)
                cv2.line(frame, tuple(map(int, shoulder_l)), tuple(map(int, wrist_l)), (255, 255, 0), 1)
            
            # RIGHT SIDE ANGLES
            elbow_r = self._get_point(lm, 8)
            wrist_r = self._get_point(lm, 10)
            shoulder_r = self._get_point(lm, 6)
            hip_r = self._get_point(lm, 12)
            ankle_r = self._get_point(lm, 16)
            
            # RIGHT ELBOW ANGLE (shoulder-elbow-wrist)
            if all(p is not None for p in [shoulder_r, elbow_r, wrist_r]):
                angle = self.angle_between(shoulder_r, elbow_r, wrist_r)
                self._draw_angle_at_point(frame, elbow_r, angle, "R-Elbow", (0, 0, 255))
                cv2.line(frame, tuple(map(int, shoulder_r)), tuple(map(int, elbow_r)), (0, 0, 255), 1)
                cv2.line(frame, tuple(map(int, elbow_r)), tuple(map(int, wrist_r)), (0, 0, 255), 1)
            
            # RIGHT SHOULDER ANGLE (wrist-shoulder-hip)
            if all(p is not None for p in [wrist_r, shoulder_r, hip_r]):
                angle = self.angle_between(wrist_r, shoulder_r, hip_r)
                self._draw_angle_at_point(frame, shoulder_r, angle, "R-Shoulder", (100, 100, 255))
                cv2.line(frame, tuple(map(int, wrist_r)), tuple(map(int, shoulder_r)), (100, 100, 255), 1)
                cv2.line(frame, tuple(map(int, shoulder_r)), tuple(map(int, hip_r)), (100, 100, 255), 1)
            
            # RIGHT CHEST ANGLE (ankle-shoulder-wrist)
            if all(p is not None for p in [ankle_r, shoulder_r, wrist_r]):
                angle = self.angle_between(ankle_r, shoulder_r, wrist_r)
                self._draw_angle_at_point(frame, shoulder_r, angle, "R-Chest", (0, 255, 255))
                cv2.line(frame, tuple(map(int, ankle_r)), tuple(map(int, shoulder_r)), (0, 255, 255), 1)
                cv2.line(frame, tuple(map(int, shoulder_r)), tuple(map(int, wrist_r)), (0, 255, 255), 1)

    # -------------------- Rep Validation --------------------
    def _validate_push_up_rep(self, lm: Dict) -> Tuple[bool, Dict]:
        """Comprehensive push-up validation"""
        checks = {}
        
        # Checks that return (bool, str)
        feet_hands = self._check_feet_hands_alignment(lm)
        back = self._check_back_straight(lm)
        head = self._check_head_up(lm)
        
        checks["feet_hands_aligned"] = (0, feet_hands[0], feet_hands[1])
        checks["back_straight"] = (0, back[0], back[1])
        checks["head_up"] = (0, head[0], head[1])
        
        # Checks that return (float, bool, str)
        checks["elbow_angle"] = self._check_elbow_angle(lm)
        checks["hand_shoulder_hip"] = self._check_hand_shoulder_hip_angle(lm)
        checks["chest_to_ground"] = self._check_chest_to_ground(lm)
        
        # All checks must pass for valid rep
        all_valid = all(check[1] for check in checks.values())
        return all_valid, checks

    # -------------------- Rest Period --------------------
    def rest_period(self, duration, numset, tarset, is_exercise_rest=False):
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
                txt = f"Push Up | Set {numset}/{tarset} and remaining sets: {tarset - numset}"
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
                print("Person detected reliably — starting workout.")
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
            print(f"Starting {exercise_name}...")
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
                    
                    # Draw validation overlays
                    self._draw_validation_overlays(frame, lm)
                    
                    # Print all angles to terminal
                    if lm:
                        self._print_all_angles(lm)
                    
                    if lm:
                        is_valid, checks = self._validate_push_up_rep(lm)
                        
                        if phase == "up":
                            # At top position, check if going down
                            if is_valid:
                                phase = "down"
                                cv2.putText(frame, "PHASE: DOWN", (10, 150),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        
                        elif phase == "down":
                            # At bottom position, check if going up
                            elbow_angle_val = checks["elbow_angle"][0]
                            chest_to_ground_val = checks["chest_to_ground"][0]
                            
                            if elbow_angle_val < 60 and chest_to_ground_val < 12:
                                rep_count += 1
                                print(f"Rep {rep_count}/{target_reps}")
                                phase = "up"
                                cv2.putText(frame, f"REP {rep_count} COMPLETE!", (10, 150),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Display info
                    txt = f"Push Up | Rep {rep_count}/{target_reps}"
                    cv2.putText(frame, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, (0, 255, 0), 2)
                    
                    if lm:
                        _, checks = self._validate_push_up_rep(lm)
                        y_offset = 90
                        for check_name, (value, is_valid, msg) in checks.items():
                            color = (0, 255, 0) if is_valid else (0, 0, 255)
                            status = "✓" if is_valid else "✗"
                            cv2.putText(frame, f"{status} {check_name}: {msg}", (10, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            y_offset += 20
                    
                    if self.visual:
                        cv2.imshow("Workout (Press Q to quit)", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            rep_count = target_reps
                            break
                
                l = len(rep_groups)
                if group_idx < l - 1:
                    self.rest_period(self.rest_between_sets, group_idx + 1, l - 1, is_exercise_rest=False)
            
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
