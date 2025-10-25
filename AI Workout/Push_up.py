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
        self.elbow_angle_range = (10, 90)  # Adjusted for bottom position
        self.hand_shoulder_hip_range = (90, 180)  # > 90
        self.chest_to_ground_range = (0,90) # ≤ 90 degrees
        
        # Thresholds for rep transitions
        self.bottom_elbow_threshold = 40
        self.bottom_chest_threshold = 40
        self.top_elbow_threshold = 140
        
        # Lost detection threshold
        self.lost_detection_threshold = 5
        
        # Confirmation frames for bottom position
        self.bottom_confirmation_frames =3

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
                if finalres[i] < 0.5:  # Lowered confidence threshold for better detection robustness
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
        tolerance = 0.4  # 20% tolerance
        
        if slope > tolerance:
            return False, "Keep your body horizontal - hands and feet aligned"
        
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
                return False, "Don't sag your hips - keep core tight"
            else:
                return False, "Don't pike your hips - keep body straight"
        
        if knee_dist > max_deviation:
            return False, "Keep your legs straight - no bending"
        
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
            return False, "Look forward - don't drop your head"
        
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
        
        
        min_angle, max_angle = self.elbow_angle_range
        
        if angle_l > max_angle and angle_r> max_angle:
            return max(angle_l,angle_r), False, "Bend your elbows more - lower your body"
        elif angle_l < min_angle and angle_r< min_angle:
            return min(angle_l,angle_r), False, "Extend your elbows more - don't over-bend"
        
        return min(angle_l,angle_r), True, "Valid"

    # def _check_hand_shoulder_hip_angle(self, lm: Dict) -> Tuple[float, bool, str]:
    #     """Check 4.b: Angle between hand, shoulder, hip > 90 degrees"""
    #     shoulder_l = self._get_point(lm, 5)
    #     shoulder_r = self._get_point(lm, 6)
    #     wrist_l = self._get_point(lm, 9)
    #     wrist_r = self._get_point(lm, 10)
    #     hip_l = self._get_point(lm, 11)
    #     hip_r = self._get_point(lm, 12)
        
    #     if any(p is None for p in [shoulder_l, wrist_l, hip_l, shoulder_r, wrist_r, hip_r]):
    #         return 0.0, False, "Missing body landmarks"
        
    #     # Calculate angles for both sides
    #     angle_l = self.angle_between(wrist_l, shoulder_l, hip_l)
    #     angle_r = self.angle_between(wrist_r, shoulder_r, hip_r)
    #     avg_angle = (angle_l + angle_r) / 2
        
    #     min_angle, max_angle = self.hand_shoulder_hip_range
        
    #     if avg_angle < min_angle:
    #         return avg_angle, False, "Widen your stance or bring hands closer to your body"
        
    #     return avg_angle, True, "Valid"

    def _check_chest_to_ground(self, lm: Dict) -> Tuple[float, bool, str]:
        """Check 4.c: Chest close to ground (shoulder-ankle-hand angle ≤ 60°)"""
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
            return avg_angle, False, "Go deeper - chest closer to ground"
        
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
        
        # 3. Draw all critical angles with clear organization
        if lm:
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
            
            # ========== LEFT SIDE ANGLES (Blue tones) ==========
            # LEFT ELBOW ANGLE (shoulder-elbow-wrist)
            if all(p is not None for p in [shoulder_l, elbow_l, wrist_l]):
                angle = self.angle_between(shoulder_l, elbow_l, wrist_l)
                self._draw_angle_at_point(frame, elbow_l, angle, "L-Elbow", (255, 0, 0))
                cv2.line(frame, tuple(map(int, shoulder_l)), tuple(map(int, elbow_l)), (255, 100, 100), 2)
                cv2.line(frame, tuple(map(int, elbow_l)), tuple(map(int, wrist_l)), (255, 100, 100), 2)
            
            # LEFT SHOULDER ANGLE (wrist-shoulder-hip)
            if all(p is not None for p in [wrist_l, shoulder_l, hip_l]):
                angle = self.angle_between(wrist_l, shoulder_l, hip_l)
                self._draw_angle_at_point(frame, shoulder_l, angle, "L-Shoulder", (255, 165, 0))
                cv2.line(frame, tuple(map(int, wrist_l)), tuple(map(int, shoulder_l)), (255, 200, 150), 2)
                cv2.line(frame, tuple(map(int, shoulder_l)), tuple(map(int, hip_l)), (255, 200, 150), 2)
            
            # LEFT CHEST ANGLE (ankle-shoulder-wrist)
            if all(p is not None for p in [ankle_l, shoulder_l, wrist_l]):
                angle = self.angle_between(ankle_l, shoulder_l, wrist_l)
                self._draw_angle_at_point(frame, shoulder_l, angle, "L-Chest", (255, 255, 0))
                cv2.line(frame, tuple(map(int, ankle_l)), tuple(map(int, shoulder_l)), (200, 200, 100), 2)
                cv2.line(frame, tuple(map(int, shoulder_l)), tuple(map(int, wrist_l)), (200, 200, 100), 2)
            
            # ========== RIGHT SIDE ANGLES (Red/Cyan tones) ==========
            # RIGHT ELBOW ANGLE (shoulder-elbow-wrist)
            if all(p is not None for p in [shoulder_r, elbow_r, wrist_r]):
                angle = self.angle_between(shoulder_r, elbow_r, wrist_r)
                self._draw_angle_at_point(frame, elbow_r, angle, "R-Elbow", (0, 0, 255))
                cv2.line(frame, tuple(map(int, shoulder_r)), tuple(map(int, elbow_r)), (100, 100, 255), 2)
                cv2.line(frame, tuple(map(int, elbow_r)), tuple(map(int, wrist_r)), (100, 100, 255), 2)
            
            # RIGHT SHOULDER ANGLE (wrist-shoulder-hip)
            if all(p is not None for p in [wrist_r, shoulder_r, hip_r]):
                angle = self.angle_between(wrist_r, shoulder_r, hip_r)
                self._draw_angle_at_point(frame, shoulder_r, angle, "R-Shoulder", (0, 165, 255))
                cv2.line(frame, tuple(map(int, wrist_r)), tuple(map(int, shoulder_r)), (150, 200, 255), 2)
                cv2.line(frame, tuple(map(int, shoulder_r)), tuple(map(int, hip_r)), (150, 200, 255), 2)
            
            # RIGHT CHEST ANGLE (ankle-shoulder-wrist)
            if all(p is not None for p in [ankle_r, shoulder_r, wrist_r]):
                angle = self.angle_between(ankle_r, shoulder_r, wrist_r)
                self._draw_angle_at_point(frame, shoulder_r, angle, "R-Chest", (0, 255, 255))
                cv2.line(frame, tuple(map(int, ankle_r)), tuple(map(int, shoulder_r)), (100, 200, 200), 2)
                cv2.line(frame, tuple(map(int, shoulder_r)), tuple(map(int, wrist_r)), (100, 200, 200), 2)
            
            # ========== JOINT MARKERS ==========
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

    # -------------------- Rep Validation --------------------
    def _validate_form(self, lm: Dict) -> Tuple[bool, Dict]:
        """Validate general form (alignment, back, head)"""
        checks = {}
        
        feet_hands = self._check_feet_hands_alignment(lm)
        back = self._check_back_straight(lm)
        head = self._check_head_up(lm)
        
        checks["feet_hands_aligned"] = (0, feet_hands[0], feet_hands[1])
        checks["back_straight"] = (0, back[0], back[1])
        checks["head_up"] = (0, head[0], head[1])
        
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
            
            if not is_exercise_rest:
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
        for ex_idx, (exercise_name, rep_groups) in enumerate(exercise_plan.items()):
            print(f"Starting {exercise_name}...")
            for group_idx, target_reps in enumerate(rep_groups):
                if group_idx == len(rep_groups) - 1:
                    break
                
                rep_count = 0
                is_at_bottom = False
                lost_detection_counter = 0
                bottom_counter = 0
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
                    #self._draw_validation_overlays(frame, lm)
                    
                    # Print all angles to terminal
                    if lm:
                        self._print_all_angles(lm)
                    
                    if lm:
                        lost_detection_counter = 0
                        form_valid, form_checks = self._validate_form(lm)
                        elbow_val, elbow_ok, elbow_msg = self._check_elbow_angle(lm)
                        chest_val, chest_ok, chest_msg = self._check_chest_to_ground(lm)
                        
                        # Give form feedback always
                        for name, (_, valid, msg) in form_checks.items():
                            if not valid:
                                self.feedback_handler.give_feedback("push_up", msg)
                        
                        # Give angle feedback only when not at bottom
                        if not is_at_bottom:
                            if not elbow_ok:
                                self.feedback_handler.give_feedback("push_up", elbow_msg)
                            if not chest_ok:
                                self.feedback_handler.give_feedback("push_up", chest_msg)
                        
                        # Check for bottom position
                        bottom_valid = form_valid and elbow_ok and chest_ok and elbow_val < self.bottom_elbow_threshold and chest_val < self.bottom_chest_threshold
                        
                        if bottom_valid:
                            bottom_counter += 1
                            if bottom_counter >= self.bottom_confirmation_frames and not is_at_bottom:
                                is_at_bottom = True
                                rep_count += 1
                                print(f"Rep {rep_count}/{target_reps}")
                                cv2.putText(frame, "PHASE: DOWN", (10, 150),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                                cv2.putText(frame, f"REP {rep_count} COMPLETE!", (10, 180),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            bottom_counter = 0
                            if is_at_bottom:
                                is_at_bottom = False
                    
                    else:
                        lost_detection_counter += 1
                        bottom_counter = 0
                        if lost_detection_counter > self.lost_detection_threshold and is_at_bottom:
                            is_at_bottom = False
                            print("[Info] Detection lost while at bottom, assuming moved up")
                    
                    # Display info
                    txt = f"Push Up | Rep {rep_count}/{target_reps}"
                    cv2.putText(frame, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, (0, 255, 0), 2)
                    
                    if lm:
                        all_checks = {**form_checks}
                        all_checks["elbow_angle"] = (elbow_val, elbow_ok, elbow_msg)
                        all_checks["chest_to_ground"] = (chest_val, chest_ok, chest_msg)
                        y_offset = 90
                        for check_name, (value, is_valid, msg) in all_checks.items():
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
