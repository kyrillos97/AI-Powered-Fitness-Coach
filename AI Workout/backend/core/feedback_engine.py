import time
import random
from models.enums import FeedbackType

class FeedbackEngine:
    def __init__(self):
        self.last_audio_cue_time = 0
        self.cooldown_seconds = 2.5  # Prevent spamming audio
        self.last_feedback = FeedbackType.NONE
        self.consecutive_perfects = 0
        self.total_perfects = 0
        self.total_reps_processed = 0
        self.last_rep_count_seen = 0

        # ── Encouragement pools ──────────────────────────────────────────
        self._perfect_phrases = [
            "Perfect form!",
            "Nailed it!",
            "Excellent rep!",
            "Great execution!",
            "Textbook form!",
            "Spot on!",
            "Beautiful rep!",
            "That's how it's done!",
            "Clean rep!",
            "Outstanding form!",
        ]

        self._streak_phrases = [
            "{n} perfect reps in a row! You're on fire!",
            "{n} in a row, keep that streak going!",
            "That's {n} perfect ones straight! Unstoppable!",
            "{n} consecutive perfects! Machine mode!",
        ]

        self._recovery_phrases = [
            "Great correction! That one was perfect!",
            "You fixed it! Perfect rep!",
            "Nice adjustment! That's the way!",
            "There you go! Back to perfect form!",
        ]

        self._milestone_phrases = [
            "{n} reps done! Keep pushing!",
            "{n} reps! You're crushing it!",
            "{n} reps completed! Stay strong!",
        ]

        self._almost_done_phrases = [
            "{done} out of {total}! Almost there!",
            "{done} of {total} reps! Just {left} more!",
            "Only {left} reps to go! You got this!",
            "{done} down, {left} to go! Finish strong!",
        ]

        self._halfway_phrases = [
            "Halfway there! Keep the energy up!",
            "Half done! You're doing amazing!",
            "50 percent complete! Stay focused!",
        ]

    # ── Correction messages with actionable advice ───────────────────
    def get_feedback_message(self, feedback_type: FeedbackType) -> str:
        messages = {
            FeedbackType.NONE: "",
            FeedbackType.PERFECT: "Perfect form!",
            FeedbackType.PARTIAL_CURL: "Full range of motion! Curl all the way up.",
            FeedbackType.WIDER_ELBOW: "Keep your elbows pinned to your body!",
            FeedbackType.OVER_RANGE: "You're going too high! Lower down slightly.",
            FeedbackType.LOWER_RANGE: "Raise higher! You need more range to hit perfect.",
            FeedbackType.BACK_ROUNDING: "Keep your back straight! Engage your core.",
            FeedbackType.SHALLOW: "Go deeper! Full squat depth for a complete rep.",
            FeedbackType.BENT_ELBOW: "Keep your arms straight! Lock those elbows.",
            FeedbackType.NOT_WORKOUT: "That doesn't look like the exercise. Check your form.",
            FeedbackType.REJECTED_BY_VAE: "Form rejected. Try a cleaner rep.",
            FeedbackType.DID_NOT_REACH_PERFECT: "Didn't fully extend. Reach the full height.",
        }
        return messages.get(feedback_type, str(feedback_type))

    def _corrective_message(self, feedback_type: FeedbackType) -> str:
        """Give corrective coaching with specific instructions on how to fix."""
        corrections = {
            FeedbackType.PARTIAL_CURL: [
                "Curl all the way up! Full range of motion!",
                "Bring the weight higher! Complete the curl!",
                "Don't stop halfway, curl to the top!",
            ],
            FeedbackType.WIDER_ELBOW: [
                "Elbows too wide! Tuck them to your sides!",
                "Pin your elbows to your body!",
                "Keep those elbows close, don't flare out!",
            ],
            FeedbackType.OVER_RANGE: [
                "Too high! Come down a bit to hit the sweet spot!",
                "You went past perfect! Lower the range slightly!",
                "Over-extending! Aim for shoulder height!",
            ],
            FeedbackType.LOWER_RANGE: [
                "Not high enough! Raise your arms more!",
                "Almost there but raise higher for a perfect rep!",
                "Lift a bit more to reach the perfect zone!",
            ],
            FeedbackType.BACK_ROUNDING: [
                "Straighten your back! Keep your core tight!",
                "Your back is rounding! Stand tall!",
                "Engage your core, keep the spine neutral!",
            ],
            FeedbackType.SHALLOW: [
                "Go deeper! Break parallel for a full squat!",
                "Not deep enough! Drop those hips lower!",
                "Get lower for a complete rep!",
            ],
            FeedbackType.BENT_ELBOW: [
                "Straighten your arms! Lock the elbows!",
                "Keep your arms fully extended!",
                "Don't bend the elbows during this exercise!",
            ],
            FeedbackType.DID_NOT_REACH_PERFECT: [
                "Extend fully! You stopped before the peak!",
                "Push through to the full extension!",
                "Almost there but finish the movement!",
            ],
        }
        phrases = corrections.get(feedback_type)
        if phrases:
            return random.choice(phrases)
        return self.get_feedback_message(feedback_type)

    def generate_audio_cue(self, feedback_type: FeedbackType, rep_count: int = 0,
                           target_reps: int = 0, current_set: int = 0, target_sets: int = 0) -> str:
        """
        Generate a context-aware audio cue.
        Tracks streaks, progress, and gives corrective + encouraging feedback.
        """
        current_time = time.time()
        audio_cue = ""

        # Only skip truly empty feedback
        if feedback_type == FeedbackType.NONE:
            return ""

        # VAE rejection / not-workout: give corrective feedback (don't swallow silently!)
        if feedback_type in (FeedbackType.NOT_WORKOUT, FeedbackType.REJECTED_BY_VAE):
            if current_time - self.last_audio_cue_time > self.cooldown_seconds:
                self.consecutive_perfects = 0
                self.last_feedback = feedback_type
                self.last_audio_cue_time = current_time
                if feedback_type == FeedbackType.REJECTED_BY_VAE:
                    return random.choice([
                        "Form rejected. Try a cleaner, more controlled rep.",
                        "That rep didn't look right. Focus on smooth movement.",
                        "Rep not counted. Slow down and control the motion.",
                    ])
                else:
                    return random.choice([
                        "That doesn't look like the exercise. Check your form.",
                        "Movement not recognized. Make sure you're doing the right exercise.",
                    ])
            return ""

        # Update rep counts only if it increased
        if rep_count > self.last_rep_count_seen:
            self.last_rep_count_seen = rep_count
            self.total_reps_processed += 1

        if feedback_type == FeedbackType.PERFECT:
            self.total_perfects += 1
            was_bad_before = self.last_feedback not in (FeedbackType.PERFECT, FeedbackType.NONE)
            self.consecutive_perfects += 1

            # Priority 1: Recovery encouragement (fixed bad form)
            if was_bad_before and self.consecutive_perfects == 1:
                audio_cue = random.choice(self._recovery_phrases)

            # Priority 2: Streak milestones
            elif self.consecutive_perfects >= 3 and self.consecutive_perfects % 3 == 0:
                audio_cue = random.choice(self._streak_phrases).format(n=self.consecutive_perfects)

            # Priority 3: Almost done (within 2 reps of target)
            elif target_reps > 0 and rep_count >= target_reps - 2 and rep_count < target_reps:
                left = target_reps - rep_count
                audio_cue = random.choice(self._almost_done_phrases).format(
                    done=rep_count, total=target_reps, left=left
                )

            # Priority 4: Halfway milestone
            elif target_reps > 0 and rep_count == target_reps // 2 and target_reps >= 4:
                audio_cue = random.choice(self._halfway_phrases)

            # Priority 5: First rep
            elif rep_count == 1:
                audio_cue = "Good start! Keep it up!"

            # Priority 6: Every 5th rep milestone
            elif rep_count > 0 and rep_count % 5 == 0:
                audio_cue = random.choice(self._milestone_phrases).format(n=rep_count)

            # Priority 7: Random encouragement (30% chance)
            elif random.random() < 0.3:
                audio_cue = random.choice(self._perfect_phrases)

        else:
            # Bad rep — reset streak and give corrective coaching
            self.consecutive_perfects = 0

            # Apply cooldown only to corrective messages so we don't spam them if the engine sends multiple
            if current_time - self.last_audio_cue_time > self.cooldown_seconds:
                audio_cue = self._corrective_message(feedback_type)

        # Update tracking
        self.last_feedback = feedback_type

        if audio_cue:
            if current_time - self.last_audio_cue_time > self.cooldown_seconds:
                self.last_audio_cue_time = current_time
            else:
                audio_cue = "" # Suppress to prevent spam and overlapping cues

        return audio_cue

    def generate_set_rest_cue(self, set_num: int, is_completed: bool = False,
                              target_sets: int = 0) -> str:
        if is_completed:
            cues = [
                "Workout completed! You crushed it!",
                "All sets done! Amazing work!",
                "That's a wrap! Great session!",
            ]
            return random.choice(cues)

        if target_sets > 0:
            remaining = target_sets - set_num
            if remaining == 1:
                return f"Set {set_num} complete! One more set to go! Take a rest."
            return f"Set {set_num} complete! {remaining} sets remaining. Take a rest."

        return f"Set {set_num} complete. Take a rest."

    def get_workout_tips(self, workout_type: str) -> str:
        """Return form tips for the current workout."""
        tips = {
            "bicep_curl": (
                "Bicep Curl tips: Keep your elbows pinned to your sides. "
                "Curl the weight all the way up to your shoulders, then lower slowly. "
                "Don't swing your body — use only your biceps."
            ),
            "squat": (
                "Squat tips: Stand with feet shoulder-width apart. "
                "Push your hips back and bend your knees. Go below parallel. "
                "Keep your chest up and back straight. Drive through your heels."
            ),
            "side_shoulder": (
                "Side Shoulder Raise tips: Stand straight with arms at your sides. "
                "Raise both arms out to the sides until shoulder height. "
                "Keep your elbows slightly bent but not excessively. Lower slowly."
            ),
            "front_shoulder": (
                "Front Shoulder Raise tips: Hold weights in front of your thighs. "
                "Raise arms forward to shoulder height with a slight elbow bend. "
                "Lower slowly and controlled. Don't swing or use momentum."
            ),
            "shrug": (
                "Shrug tips: Stand with arms at your sides holding weights. "
                "Raise your shoulders straight up towards your ears. "
                "Hold at the top for a second, then lower slowly. Keep arms straight."
            ),
        }
        return tips.get(workout_type, "Focus on controlled movements and proper form.")
