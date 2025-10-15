import time
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple

exercise_plan = {
    "squat": (10, 3),  # Example: 3 sets of 10 reps (fixed from unrealistic 400)
}

class DefaultFeedback:
    """Fallback feedback handler (simple print)."""
    def give_feedback(self, exercise_name: str, issue: str):
        print(f"[Feedback] {exercise_name}: {issue}")

class Workout:
    def __init__(self, video_path=1, visual=True,
                 model_path="yolo11n-pose.pt", feedback_handler=None,
                 rest_between_sets=60, rest_between_exercises=120):
        # video_path=0 → default camera
        self.video_path = video_path
        self.visual = visual
        self.model = YOLO(model_path)
        self.feedback_handler = feedback_handler if feedback_handler else DefaultFeedback()
        self.rest_between_sets = rest_between_sets
        self.rest_between_exercises = rest_between_exercises
        self.cap = None

        # Person detection readiness controls
        self.detection_confirmation_frames = 5   # consecutive frames to confirm person
        self._detected_frames = 0
        self.person_confirmed = False

        # Feet stability params
        self.feet_stability_thresh_ratio_max = 0.75
        self.feet_stability_thresh_ratio_min =  -0.02 # 10% deviation allowed (fixed from 0.6)
        self.unstable_confirmation_frames = 10   # Consecutive unstable frames to trigger (debounced)
        self._unstable_frames = 0
        self.baseline_ankle_dist = 0.0  # Baseline ankle distance (calibrated at start)
        self.baseline_hip_dist = 0.0
        self.baseline_ratio = 0.0

        # Hip-knee y checker params
        self.hip_knee_y_threshold_ratio = 0.4  # Max allowed y-diff ratio for valid squat (40% of standing leg height)
        self.baseline_leg_height_left = 0.0
        self.baseline_leg_height_right = 0.0

        # Angle smoothing params
        self.prev_angle_left = 180.0
        self.prev_angle_right = 180.0
        self.smoothing_factor = 0.8  # For jitter reduction

        # Phase debouncing
        self.squat_confirm_frames = 2  # Consecutive frames needed to enter squat phase
        self.stand_confirm_frames = 2  # Consecutive frames needed to return to stand
        self._squat_confirm_count = 0
        self._stand_confirm_count = 0

    # -------------------- Landmarks --------------------
    def angle_between(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        dot = np.dot(ba, bc)
        norm = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norm == 0:
            return 0.0
        cos_angle = np.clip(dot / norm, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def _get_landmarks(self, results):
        lm = {}
        if not results or len(results) == 0:
            return lm
        try:
            res_conf = results[0].keypoints.conf.cpu().numpy()
        except Exception:
            return lm

        finalres = res_conf[0] if res_conf.shape[0] > 0 else []
        if len(finalres) != 0:
            for i in [11,12,13,14,15,16]:  # Critical keypoints
                if finalres[i] < 0.7:  # Lowered threshold for better detection
                    return lm  

        kpts = results[0].keypoints.xy.cpu().numpy()
        if kpts.shape[0] == 0:
            return lm
        kp = kpts[0]
        for idx, (x, y) in enumerate(kp):
            lm[str(idx)] = (float(x), float(y))
        return lm

    def _compute_angle_for_triplet(self, lm, triplet: Tuple[int, int, int]) -> float:
        if str(triplet[0]) not in lm or str(triplet[1]) not in lm or str(triplet[2]) not in lm:
            return 0.0
        a = np.array(lm[str(triplet[0])], dtype=float)
        b = np.array(lm[str(triplet[1])], dtype=float)
        c = np.array(lm[str(triplet[2])], dtype=float)
        return self.angle_between(a, b, c)

    def _compute_dist(self, lm, pt1: int, pt2: int) -> float:
        if str(pt1) not in lm or str(pt2) not in lm:
            return 0.0
        p1 = np.array(lm[str(pt1)], dtype=float)
        p2 = np.array(lm[str(pt2)], dtype=float)
        return np.linalg.norm(p1 - p2)

    # -------------------- Feet stability validation --------------------
    def _validate_feet_stable(self, lm, frame) -> bool:
        """
        Returns True if feet are stable: ankle distance within deviation of baseline.
        """
        required_keys = ["11", "12", "15", "16"]
        if any(key not in lm for key in required_keys):
            self._unstable_frames = 0
            return True  # Assume stable if can't check

        dist_ankles = self._compute_dist(lm, 15, 16)
        dist_hips = self._compute_dist(lm, 11, 12)
        if dist_hips == 0:
            return True
        
        b1 = 1
        
        # Calibrate baseline if not set (e.g., at startup)
        deviation = dist_ankles / dist_hips
        if deviation > 2.3 or deviation < 0.8:
            b1 = 0

       
        print("%"*50)
        print(deviation)
        print("%"*50)
        print("#"*50)
        print(dist_ankles)
        print("#"*50)

        # Visualize ankle line
        if self.visual and "15" in lm and "16" in lm:
            color = (0, 255, 0) if b1 else (0, 0, 255)
            cv2.line(frame, (int(lm["15"][0]), int(lm["15"][1])), (int(lm["16"][0]), int(lm["16"][1])), color, 2)

        return b1

    # -------------------- Validation --------------------
    def _validate_squat(self, angle: float, angle2: float, lm) -> bool:
        # Check angles
        min_angle = min(angle, angle2)
        if min_angle > 135:
            self.feedback_handler.give_feedback("squat", "Bend your knees more")
            return False
        if min_angle < 75:
            self.feedback_handler.give_feedback("squat", "Don't squat too deep")
            return False

        # Additional hip-knee y checker for valid squat depth
        required_keys = ["11", "12", "13", "14"]
        if any(key not in lm for key in required_keys):
            return False  # Can't check, invalid

        # Compute current y-diffs (assuming y increases downward)
        current_diff_left = lm["13"][1] - lm["11"][1]  # knee_y - hip_y
        current_diff_right = lm["14"][1] - lm["12"][1]

        # Check if within allowed range (not exactly same, with error tolerance)
        if self.baseline_leg_height_left > 0 and self.baseline_leg_height_right > 0:
            is_valid_left = current_diff_left <= self.baseline_leg_height_left * self.hip_knee_y_threshold_ratio
            is_valid_right = current_diff_right <= self.baseline_leg_height_right * self.hip_knee_y_threshold_ratio
            if not (is_valid_left and is_valid_right):
                self.feedback_handler.give_feedback("squat", "Lower your hips more to align with knees")
                return False
        else:
            return False  # No baseline, invalid

        # If angles in range and y-check passes
        if 75 <= min_angle <= 135:  # Widened range for tolerance
            return True
        return False

    # -------------------- Rest Period --------------------
    def rest_period(self, duration, numset=None, tarset=None, is_exercise_rest=False):
        label = "exercise" if is_exercise_rest else "set"
        end_time = time.time() + duration
        while time.time() < end_time:
            remaining = int(end_time - time.time()) + 1
            ret, frame = self.cap.read()
            if not ret:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            txt = f"Rest between {label}s: {remaining} sec"
            cv2.putText(frame, txt, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            if not is_exercise_rest and numset is not None and tarset is not None:
                txt = f"Squat | Set {numset}/{tarset}, remaining sets: {tarset - numset}"
                cv2.putText(frame, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Workout (Press Q to quit)", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    # -------------------- Training Loop --------------------
    def train(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not opened")

        # Person detection and baseline calibration
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
                # Calibrate baseline ankle dist
                self.baseline_ankle_dist = self._compute_dist(lm, 15, 16)
                self.baseline_hip_dist = self._compute_dist(lm, 11, 12)
                if self.baseline_hip_dist > 0:
                    self.baseline_ratio = self.baseline_ankle_dist / self.baseline_hip_dist
                # Calibrate baseline leg heights (y-diffs in standing)
                if "11" in lm and "13" in lm:
                    self.baseline_leg_height_left = lm["13"][1] - lm["11"][1]
                if "12" in lm and "14" in lm:
                    self.baseline_leg_height_right = lm["14"][1] - lm["12"][1]
                cv2.putText(frame, "Person confirmed. Starting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if self.visual:
                    cv2.imshow("Workout (Press Q to quit)", frame)
                    cv2.waitKey(500)
                break

            cv2.putText(frame, f"Waiting for person... ({self._detected_frames}/{self.detection_confirmation_frames})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.visual:
                cv2.imshow("Workout (Press Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User aborted before detection.")
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return

        # Main exercise loop
        phase = "stand"
        for ex_idx, (exercise_name, rep_groups) in enumerate(exercise_plan.items()):
            print(f"Starting {exercise_name}...")
            total_sets = len(rep_groups)
            for group_idx, target_reps in enumerate(rep_groups):
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

                    # Feet stability with debouncing
                    is_stable = self._validate_feet_stable(lm, frame)
                    if lm and not is_stable:
                        self._unstable_frames += 1
                        if self._unstable_frames >= self.unstable_confirmation_frames:
                            self.feedback_handler.give_feedback("squat", "Feet must stay stable – do not move them.")
                            cv2.putText(frame, "Feet not stable!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                            if self.visual:
                                cv2.imshow("Workout (Press Q to quit)", frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    rep_count = target_reps
                                    break
                            continue  # Skip if confirmed unstable
                    else:
                        self._unstable_frames = 0

                    angle = 0.0
                    angle2 = 0.0
                    if lm:
                        new_angle = self._compute_angle_for_triplet(lm, (11, 13, 15))
                        new_angle2 = self._compute_angle_for_triplet(lm, (12, 14, 16))
                        # Smooth angles
                        angle = self.smoothing_factor * self.prev_angle_left + (1 - self.smoothing_factor) * new_angle
                        angle2 = self.smoothing_factor * self.prev_angle_right + (1 - self.smoothing_factor) * new_angle2
                        self.prev_angle_left = angle
                        self.prev_angle_right = angle2

                    if phase == "stand":
                        if self._validate_squat(angle, angle2, lm):
                            self._squat_confirm_count += 1
                            if self._squat_confirm_count >= self.squat_confirm_frames:
                                phase = "squat"
                                self._squat_confirm_count = 0
                        else:
                            self._squat_confirm_count = 0
                    elif phase == "squat":
                        if min(angle, angle2) > 135:  # Softened stand threshold
                            self._stand_confirm_count += 1
                            if self._stand_confirm_count >= self.stand_confirm_frames:
                                rep_count += 1
                                print(f"Rep {rep_count}/{target_reps}")
                                phase = "stand"
                                self._stand_confirm_count = 0
                                # Optional: Update baseline if needed, but avoid to prevent drift
                        else:
                            self._stand_confirm_count = 0

                    txt = f"Squat | Rep {rep_count}/{target_reps}"
                    cv2.putText(frame, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Knee Angles: {angle:.1f}, {angle2:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    ready_txt = "CONFIRMED" if self.person_confirmed else "NOT CONFIRMED"
                    cv2.putText(frame, f"Person: {ready_txt}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    if self.visual:
                        cv2.imshow("Workout (Press Q to quit)", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            rep_count = target_reps
                            break

                if group_idx < total_sets - 1:
                    self.rest_period(self.rest_between_sets, group_idx + 1, total_sets - 1, is_exercise_rest=False)

            if ex_idx < len(exercise_plan) - 1:
                self.rest_period(self.rest_between_exercises, is_exercise_rest=True)

        self.cap.release()
        cv2.destroyAllWindows()
        print("Training session complete!")

# -------------------- Example --------------------
if __name__ == "__main__":
    w = Workout(video_path=1,  # 0 for default camera
                visual=True,
                model_path="yolo11n-pose.pt")
    w.train()
