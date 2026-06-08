import time
import cv2
import numpy as np
from typing import Tuple
from ultralytics import YOLO

exercise_plan = {
    "plank": (30,),
    # مثال: مجموعة واحدة 30 ثانية
}

class DefaultFeedback:
    """Fallback feedback handler (simple print)."""
    def give_feedback(self, exercise_name: str, issue: str):
        print(f"[Feedback] {exercise_name}: {issue}")

class Workout:
    def __init__(self, video_path=1, visual=True,
                 model_path="yolo11n-pose.pt", feedback_handler=None, rest_between_sets=60, rest_between_exercises=120):
        self.video_path = video_path
        self.visual = visual
        self.model = YOLO(model_path)  # Suggestion: For better accuracy, consider using "yolo11m-pose.pt" if available
        self.feedback_handler = feedback_handler if feedback_handler else DefaultFeedback()
        self.rest_between_sets = rest_between_sets
        self.rest_between_exercises = rest_between_exercises
        self.cap = None
        # --- EDIT: person-detection readiness controls ---
        # Will require a few consecutive frames at startup to confirm person presence
        self.detection_confirmation_frames = 5  # number of consecutive frames required to confirm person presence
        self._detected_frames = 0  # current consecutive detection count
        self.person_confirmed = False  # True once person is confirmed (detected once at start)
        # -------------------------------------------------
        # New: For temporal smoothing of angles
        self.recent_elbow_angles = []  # List to store recent elbow angles (up to 5)
        self.recent_upper_angles = []  # For back straight upper angle
        self.recent_lower_angles = []  # For back straight lower angle

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
        # results can be empty or have no keypoints; guard accordingly
        if not results or len(results) == 0:
            return lm
        # If the model returned keypoints, check confidences and positions
        try:
            res_conf = results[0].keypoints.conf.cpu().numpy()
        except Exception:
            return lm
        finalres = res_conf[0] if res_conf.shape[0] > 0 else []
        # require the critical keypoints to have sufficient confidence
        critical_keypoints = [0,5,6,7,8,9,10,11,12,15,16]
        if len(finalres) != 0:
            for i in critical_keypoints:
                # if any of those keypoints is low confidence → treat as "no reliable person"
                if finalres[i] < 0.60:  # Lowered from 0.80 to 0.60 for more forgiveness in side views
                    return lm
        # New: Calculate dominant side based on average confidence
        left_conf = np.mean([finalres[5], finalres[7], finalres[9], finalres[11], finalres[15]]) if len(finalres) > 0 else 0
        right_conf = np.mean([finalres[6], finalres[8], finalres[10], finalres[12], finalres[16]]) if len(finalres) > 0 else 0
        dominant_side = 'left' if left_conf > right_conf else 'right'
        lm['dominant_side'] = dominant_side  # Store in lm for easy access

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

    # def _is_body_straight(self, lm, threshold=50):  # Increased from 30 to 50 for more tolerance
    #     if not all(str(k) in lm for k in [5,6,11,12,15,16]):
    #         return False
    #     y_s = (lm['5'][1] + lm['6'][1]) / 2
    #     y_h = (lm['11'][1] + lm['12'][1]) / 2
    #     y_a = (lm['15'][1] + lm['16'][1]) / 2
    #     y_avg = (y_s + y_a) / 2
    #     dev = abs(y_h - y_avg)
    #     if dev > threshold:
    #         if y_h > y_avg:
    #             self.feedback_handler.give_feedback("plank", "Don't sag your hips - keep core tight")
    #         else:
    #             self.feedback_handler.give_feedback("plank", "Don't pike your hips - keep body straight")
    #         return False
    #     return True

    # -------------------- Elbow Angle Check --------------------
    def check_elbow_angle(self, lm) -> Tuple[bool, float]:
        """
        Check if elbow angle is approximately 90 degrees (±20 degree threshold, increased from 15).
        For plank, we check the elbow angles, preferring the dominant side in side views.
        Returns: (is_valid, arm_angle_average)
        """
        elbow_target = 90
        elbow_threshold = 30  # Increased from 15 for more tolerance

        # Right arm: shoulder(6) -> elbow(8) -> wrist(10)
        right_elbow_angle = self._compute_angle_for_triplet(lm, (6, 8, 10))
        # Left arm: shoulder(5) -> elbow(7) -> wrist(9)
        left_elbow_angle = self._compute_angle_for_triplet(lm, (5, 7, 9))

        # New: Use dominant side logic
        dominant_side = lm.get('dominant_side', 'right')  # Default to right if not set
        if dominant_side == 'left' and left_elbow_angle > 0:
            elbow_angle = left_elbow_angle
            print("Using left side elbow")
        elif dominant_side == 'right' and right_elbow_angle > 0:
            elbow_angle = right_elbow_angle
            print("Using right side elbow")
        elif right_elbow_angle > 0 and left_elbow_angle > 0:
            elbow_angle = (right_elbow_angle + left_elbow_angle) / 2
        elif right_elbow_angle > 0:
            elbow_angle = right_elbow_angle
        elif left_elbow_angle > 0:
            elbow_angle = left_elbow_angle
        else:
            elbow_angle = 0

        # New: Temporal smoothing
        self.recent_elbow_angles.append(elbow_angle)
        if len(self.recent_elbow_angles) > 5:
            self.recent_elbow_angles.pop(0)
        avg_elbow_angle = np.mean(self.recent_elbow_angles) if self.recent_elbow_angles else 0

        # Check if within threshold
        is_valid = abs(avg_elbow_angle - elbow_target) <= elbow_threshold

        # Print arm angles for monitoring
        print(f"Right Elbow: {right_elbow_angle:.1f}° | Left Elbow: {left_elbow_angle:.1f}° | Smoothed Average: {avg_elbow_angle:.1f}°")

        if not is_valid:
            if avg_elbow_angle < elbow_target:
                self.feedback_handler.give_feedback("plank", "Elbows too straight - bend more to reach 90°")
            else:
                self.feedback_handler.give_feedback("plank", "Elbows too bent - straighten slightly to reach 90°")

        return is_valid, avg_elbow_angle

    # -------------------- Back Straight Check --------------------
    def check_back_straight(self, lm, angle_threshold=30) -> bool:
        """
        Check if the back is straight (180 degrees along the body line).
        Uses angle calculations to validate body alignment, preferring dominant side.
        Returns: is_back_straight (bool)
        """
        if not all(str(k) in lm for k in [0, 5, 6, 11, 12, 15, 16]):
            return False

        dominant_side = lm.get('dominant_side', 'right')

        # Use dominant side for shoulders, hips, ankles
        if dominant_side == 'left':
            shoulder_pos = np.array(lm['5'], dtype=float)
            hip_pos = np.array(lm['11'], dtype=float)
            ankle_pos = np.array(lm['15'], dtype=float)
        else:
            shoulder_pos = np.array(lm['6'], dtype=float)
            hip_pos = np.array(lm['12'], dtype=float)
            ankle_pos = np.array(lm['16'], dtype=float)

        head_pos = np.array(lm['0'], dtype=float)

        # Calculate angles for back alignment
        # Angle between head-shoulder-hip
        #upper_angle = self.angle_between(head_pos, shoulder_pos, hip_pos)
        # Angle between shoulder-hip-ankle
        lower_angle = self.angle_between(shoulder_pos, hip_pos, ankle_pos)
        
        print('*'*50)
        print(lower_angle)
        print('*'*50)

        # New: Temporal smoothing
        # self.recent_upper_angles.append(upper_angle)
        # if len(self.recent_upper_angles) > 5:
        #     self.recent_upper_angles.pop(0)
        # smoothed_upper = np.mean(self.recent_upper_angles) if self.recent_upper_angles else 0

        self.recent_lower_angles.append(lower_angle)
        if len(self.recent_lower_angles) > 5:
            self.recent_lower_angles.pop(0)
        smoothed_lower = np.mean(self.recent_lower_angles) if self.recent_lower_angles else 0

        # For a straight back, both angles should be close to 180°
        #upper_is_straight = abs(smoothed_upper - 180) <= angle_threshold
        lower_is_straight = abs(smoothed_lower - 180) <= angle_threshold

        is_straight = lower_is_straight

        if not is_straight:
            # if smoothed_upper < 180 - angle_threshold:
            #     self.feedback_handler.give_feedback("plank", "Hips sagging - engage your core and lift hips up")
            # elif smoothed_upper > 180 + angle_threshold:
            #     self.feedback_handler.give_feedback("plank", "Hips piking too high - lower hips to align with shoulders")
            if smoothed_lower < 180 - angle_threshold:
                self.feedback_handler.give_feedback("plank", "Ankles not aligned - keep legs straight")
            elif smoothed_lower > 180 + angle_threshold:
                self.feedback_handler.give_feedback("plank", "Lower body bent - straighten your legs")
            else:
                self.feedback_handler.give_feedback("plank", "Body misaligned - keep a straight line from head to ankles")

        return is_straight

    # -------------------- Validation --------------------
    def _validate_plank(self,elbow_valid: bool, back_straight: bool) -> bool:
        """
        Validate overall plank form based on key metrics.
        Returns True only if all form requirements are met.
        """
      
        if not elbow_valid:
            return False
        if not back_straight:
            return False
        return True

    # -------------------- Rest Period --------------------
    def rest_period(self, duration, current_set=None, total_sets=None, is_exercise_rest=False):
        label = "exercise" if is_exercise_rest else "set"
        end_time = time.time() + duration  # when the rest should end
        while time.time() < end_time:
            remaining = int(end_time - time.time()) + 1  # seconds left
            # Try to get a frame from the camera
            ret, frame = self.cap.read()
            if not ret:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)  # black fallback
            txt = f"Rest between {label}s: {remaining} sec"
            cv2.putText(frame, txt, (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 3)
            if not is_exercise_rest and current_set is not None and total_sets is not None:
                txt = f"Plank | Set {current_set}/{total_sets} | Remaining sets: {total_sets - current_set}"
                cv2.putText(frame, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)
            cv2.imshow("Workout (Press Q to quit)", frame)
            # Wait only ~30 ms, so the loop is ~33 fps; no big block
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    # -------------------- Training Loop --------------------
    def train(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not opened")
        # ---------------- EDIT: single-time person detection at startup ----------------
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
                # small visual confirmation frame for user
                cv2.putText(frame, "Person confirmed. Starting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if self.visual:
                    cv2.imshow("Workout (Press Q to quit)", frame)
                    cv2.waitKey(500)
                break
            # show waiting overlay
            cv2.putText(frame, f"Waiting for person... ({self._detected_frames}/{self.detection_confirmation_frames})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.visual:
                cv2.imshow("Workout (Press Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User aborted before detection.")
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return
        # -------------------------------------------------------------------------------
        # Main exercise loop — now assume person_confirmed stays True for the whole session
        for ex_idx, (exercise_name, hold_durations) in enumerate(exercise_plan.items()):
            print(f"Starting {exercise_name}...")
            for group_idx, target_hold in enumerate(hold_durations):
                hold_start = None
                hold_elapsed = 0.0
                print(f"Set {group_idx + 1}: Hold for {target_hold} sec")
                while hold_elapsed < target_hold:
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
                    body_straight = False
                    elbow_valid = False
                    avg_elbow_angle = 0.0
                    back_straight = False
                    # Since we confirmed person once at start, proceed to compute angles regardless
                    if lm:
                        # body_straight = self._is_body_straight(lm)  # Uncommented to include in validation
                        # Check elbow angle (90° ± 20°)
                        elbow_valid, avg_elbow_angle = self.check_elbow_angle(lm)
                        # Check back alignment (180°)
                        back_straight = self.check_back_straight(lm)
                    current_time = time.time()
                    is_good_form = self._validate_plank(elbow_valid, back_straight)
                    if is_good_form:
                        if hold_start is None:
                            hold_start = current_time
                            print("Good form detected - holding...")
                        hold_elapsed = current_time - hold_start
                    else:
                        if hold_start is not None:
                            self.feedback_handler.give_feedback("plank", "Form broke - reset hold")
                            hold_start = None
                            hold_elapsed = 0.0
                    remaining = max(0, target_hold - hold_elapsed)
                    txt = f"Plank | Remaining: {remaining:.1f} sec"
                    cv2.putText(frame, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)

                    # Display elbow angle check result
                    elbow_color = (0, 255, 0) if elbow_valid else (0, 0, 255)
                    cv2.putText(frame, f"Elbow: {avg_elbow_angle:.1f}° {'✓' if elbow_valid else '✗'}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, elbow_color, 2)
                    # Display back alignment check result
                    back_color = (0, 255, 0) if back_straight else (0, 0, 255)
                    cv2.putText(frame, f"Back: {'STRAIGHT' if back_straight else 'NOT STRAIGHT'} {'✓' if back_straight else '✗'}", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, back_color, 2)
                    # # Display body alignment check result
                    # body_color = (0, 255, 0) if body_straight else (0, 0, 255)
                    # cv2.putText(frame, f"Body: {'ALIGNED' if body_straight else 'MISALIGNED'} {'✓' if body_straight else '✗'}", (10, 130),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, body_color, 2)
                    # Display overall form status
                    # form_color = (0, 255, 0) if is_good_form else (0, 0, 255)
                    # cv2.putText(frame, f"Form: {'GOOD' if is_good_form else 'NEEDS ADJUSTMENT'}", (10, 150),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, form_color, 2)
                    if self.visual:
                        cv2.imshow("Workout (Press Q to quit)", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            hold_elapsed = target_hold
                            break
                print(f"Set {group_idx + 1} completed!")
                l = len(hold_durations)
                if group_idx < l - 1:
                    self.rest_period(self.rest_between_sets, group_idx + 1, l, is_exercise_rest=False)

            if ex_idx < len(exercise_plan) - 1:
                self.rest_period(self.rest_between_exercises, is_exercise_rest=True)
        self.cap.release()
        cv2.destroyAllWindows()
        print("Training session complete!")

# -------------------- Example --------------------
if __name__ == "__main__":
    w = Workout()
    w.train()
