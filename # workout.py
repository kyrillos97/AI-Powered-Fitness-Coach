# workout.py (Colab-ready, squat validation with range of motion)
# Install first in Colab:
# !pip install ultralytics opencv-python numpy

import time
import cv2
import numpy as np
from typing import Tuple
from ultralytics import YOLO
from google.colab.patches import cv2_imshow  # Import cv2_imshow for Colab

exercise_plan = {
    "squat": (1, 8, 7, 2),  # 4 sets
}


class DefaultFeedback:
    """Fallback feedback handler (simple print)."""
    def give_feedback(self, exercise_name: str, issue: str):
        print(f"[Feedback] {exercise_name}: {issue}")


class Workout:
    def __init__(self, video_path=None, visual=True,
                 model_path="yolo11n-pose.pt", feedback_handler=None, rest_between_sets=60, rest_between_exercises=120):
        self.video_path = video_path
        self.visual = visual
        self.model = YOLO(model_path)
        self.feedback_handler = feedback_handler if feedback_handler else DefaultFeedback()
        self.rest_between_sets = rest_between_sets
        self.rest_between_exercises = rest_between_exercises
        self.cap = None

    # -------------------- Landmarks --------------------
    def angle_between(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Return angle at point b formed by points a-b-c in degrees."""
        ba = a - b
        bc = c - b
        dot = np.dot(ba, bc)
        norm = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norm == 0:
            return 0.0
        cos_angle = np.clip(dot / norm, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def _get_landmarks(self, results):
        """Extract landmarks from YOLOv11 pose results (COCO 17 keypoints)."""
        lm = {}
        if not results or len(results) == 0:
            return lm
        kpts = results[0].keypoints.xy.cpu().numpy()
        if kpts.shape[0] == 0:
            return lm
        kp = kpts[0]  # first person
        for idx, (x, y) in enumerate(kp):
            lm[str(idx)] = (float(x), float(y))
        return lm

    def _compute_angle_for_triplet(self, lm, triplet: Tuple[int, int, int]) -> float:
        if str(triplet[0]) not in lm or str(triplet[1]) not in lm or str(triplet[2]) not in lm:
            return 0.0
        a = np.array(lm[str(triplet[0])], dtype=float)
        b = np.array(lm[str(triplet[1])], dtype=float)
        c = np.array(lm[str(triplet[2])], dtype=float)
        return self.angle_between(a, b, c)  # Fixed: added self.

    # -------------------- Validation --------------------
    def _validate_squat(self, angle: float) -> bool:
        """
        Valid squat if knee angle between 80° and 120°.
        """
        if 80 <= angle <= 110:
            return True
        if angle > 110:
            self.feedback_handler.give_feedback("squat", "Bend your knees more")
            return False
        if angle < 80:
            self.feedback_handler.give_feedback("squat", "Raise your knees more")
            return False
        
    # -------------------- Rest Period --------------------
    def rest_period(self, duration, is_exercise_rest=False):
        """Handle rest period with countdown timer."""
        label = "exercise" if is_exercise_rest else "set"
        print(f"Rest for {duration} seconds between {label}s...")
        start_rest = time.time()
        while time.time() - start_rest < duration:
            if self.visual:
                ret, frame = self.cap.read()
                if not ret:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Black frame if video ends
                remaining = int(duration - (time.time() - start_rest))
                cv2.putText(frame, f"Rest: {remaining} seconds", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2_imshow(frame)  # Use cv2_imshow for Colab
            # time.sleep(0.5)

    # -------------------- Training Loop --------------------
    def train(self):
        self.cap = cv2.VideoCapture(self.video_path)  # Fixed: use self.cap
        if not self.cap.isOpened():
            raise RuntimeError("Video not opened")

        phase = "stand"  # can be 'stand' or 'squat'
        for ex_idx, (exercise_name, rep_groups) in enumerate(exercise_plan.items()):
            if exercise_name not in exercise_plan:
                print(f"Unsupported exercise: {exercise_name}")
                continue
            print(f"Starting {exercise_name}...")
            total_sets = len(rep_groups)
            for group_idx, target_reps in enumerate(rep_groups):
                rep_count = 0
                print(f"Set {group_idx + 1}: {target_reps} reps")
                while rep_count < target_reps:
                    ret, frame = self.cap.read()  # Fixed: use self.cap
                    if not ret:
                        print("Video ended or camera disconnected")
                        break

                    results = self.model(frame, verbose=False)
                    lm = self._get_landmarks(results)

                    # compute knee angle (hip=11, knee=13, ankle=15)
                    angle = 0.0
                    if lm:
                        angle = self._compute_angle_for_triplet(lm, (11, 13, 15))

                    # squat logic
                    if phase == "stand":
                        if self._validate_squat(angle):
                            phase = "squat"
                          
                    elif phase == "squat":
                        if angle > 150:  # back to standing
                            rep_count += 1
                            print(f"Rep {rep_count}/{target_reps}")
                            phase = "stand"

                    # draw info
                    txt = f"Squat | Rep {rep_count}/{target_reps}"
                    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Knee Angle: {angle:.1f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    if self.visual:
                        cv2_imshow(frame)  # Use cv2_imshow for Colab

                # Rest between sets (except after last set)
                if group_idx < len(rep_groups) - 1:
                    self.rest_period(self.rest_between_sets, is_exercise_rest=False)

            # Rest between exercises (except after last exercise)
            if ex_idx < len(exercise_plan) - 1:
                self.rest_period(self.rest_between_exercises, is_exercise_rest=True)

        self.cap.release()  # Fixed: use self.cap
        print("Training session complete!")


# -------------------- Example --------------------
if __name__ == "__main__":
    # Example: test with squat video
    w = Workout(video_path="/content/STOP doing your SQUATS like this!.mp4", visual=True,
                model_path="/content/yolo11n-pose.pt")
    w.train()  # Fixed: removed reps_target parameter