import time
import cv2
import numpy as np
from typing import Tuple
from ultralytics import YOLO


exercise_plan = {
    "squat": (2,2,1,1), 
          # مثال: مجموعة واحدة 10 عدات
}

class DefaultFeedback:
    """Fallback feedback handler (simple print)."""
    def give_feedback(self, exercise_name: str, issue: str):
        print(f"[Feedback] {exercise_name}: {issue}")

class Workout:
    def __init__(self, video_path=1, visual=True,
                 model_path="yolo11n-pose.pt", feedback_handler=None, rest_between_sets=60, rest_between_exercises=120):
        # video_path=0 → الكاميرا
        self.video_path = video_path
        self.visual = visual
        self.model = YOLO(model_path)
        self.feedback_handler = feedback_handler if feedback_handler else DefaultFeedback()
        self.rest_between_sets = rest_between_sets
        self.rest_between_exercises = rest_between_exercises
        self.cap = None

        # --- EDIT: person-detection readiness controls ---
        # Will require a few consecutive frames at startup to confirm person presence
        self.detection_confirmation_frames = 5   # number of consecutive frames required to confirm person presence
        self._detected_frames = 0                # current consecutive detection count
        self.person_confirmed = False            # True once person is confirmed (detected once at start)
        # -------------------------------------------------

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
        # require the critical keypoints (11..16) to have sufficient confidence
        if len(finalres) != 0:
            for i in range(11, 17):
                # if any of those keypoints is low confidence → treat as "no reliable person"
                if finalres[i] < 0.80:
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

    # -------------------- Validation --------------------
    def _validate_squat(self, angle: float,angle2:float) -> bool:
        if 80 <= angle <= 110 and 80 <= angle2 <= 110:
            return True
        if angle > 110 and angle2 > 110:
            self.feedback_handler.give_feedback("squat", "Bend your knees more")
            return False
        if angle < 80 and angle2 < 80:
            self.feedback_handler.give_feedback("squat", "Raise your knees more")
            return False
        return False  # explicit fallback

    # -------------------- Rest Period --------------------
    
    def rest_period(self, duration,numset,tarset, is_exercise_rest=False):
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
            if label !="exercise":
              txt = f"Squat | Set {numset}/{tarset} and the remaining set : {tarset-numset}"
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
        phase = "stand"
        for ex_idx, (exercise_name, rep_groups) in enumerate(exercise_plan.items()):
            print(f"Starting {exercise_name}...")
            for group_idx, target_reps in enumerate(rep_groups):
                if group_idx ==len(rep_groups)-1:
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

                    angle = 0.0
                    angle2 = 0.0
                    # Since we confirmed person once at start, proceed to compute angles regardless
                    if lm:
                        angle = self._compute_angle_for_triplet(lm, (11, 13, 15))
                        angle2 = self._compute_angle_for_triplet(lm, (12, 14, 16))

                    if phase == "stand":
                        if self._validate_squat(angle,angle2):
                            phase = "squat"
                    elif phase == "squat":
                        if angle > 150 and angle2 > 150:
                            rep_count += 1
                            print(f"Rep {rep_count}/{target_reps}")
                            phase = "stand"

                    txt = f"Squat | Rep {rep_count}/{target_reps}"
                    cv2.putText(frame, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Knee Angle: {angle:.1f}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    ready_txt = "CONFIRMED" if self.person_confirmed else "NOT CONFIRMED"
                    cv2.putText(frame, f"Person: {ready_txt}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    if self.visual:
                        cv2.imshow("Workout (Press Q to quit)", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            rep_count = target_reps
                            break
                l=len(rep_groups)
                if group_idx < l - 1:
                    self.rest_period(self.rest_between_sets,group_idx+1,l - 1, is_exercise_rest=False)
                    

            if ex_idx < len(exercise_plan) - 1:
                self.rest_period(self.rest_between_exercises, is_exercise_rest=True)

        self.cap.release()
        cv2.destroyAllWindows()
        print("Training session complete!")

# -------------------- Example --------------------
if __name__ == "__main__":
    w = Workout(video_path=1,  # 0 معناها افتح الكاميرا الافتراضية
                visual=True,
                model_path="yolo11n-pose.pt")
    w.train()
