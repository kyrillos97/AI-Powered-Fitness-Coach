from enum import Enum

class WorkoutType(str, Enum):
    BICEP_CURL = "bicep_curl"
    FRONT_SHOULDER = "front_shoulder"
    SHRUG = "shrug"
    SIDE_SHOULDER = "side_shoulder"
    SQUAT = "squat"

class FeedbackType(str, Enum):
    NONE = "none"
    PERFECT = "perfect"
    PARTIAL_CURL = "partial_curl"
    WIDER_ELBOW = "wider_elbow"
    OVER_RANGE = "over_range"
    LOWER_RANGE = "lower_range"
    BACK_ROUNDING = "back_rounding"
    SHALLOW = "shallow"
    BENT_ELBOW = "bent_elbow"
    NOT_WORKOUT = "not_workout"
    REJECTED_BY_VAE = "rejected_by_vae"
    DID_NOT_REACH_PERFECT = "did_not_reach_perfect"

class VoiceCommand(str, Enum):
    STOP_SESSION = "stop_session"
    PAUSE = "pause"
    RESUME = "resume"
    SKIP_REST = "skip_rest"
    NEXT_SET = "next_set"
    OPEN_CHATBOT = "open_chatbot"
    RESET_REPS = "reset_reps"
    HOW_MANY_REPS = "how_many_reps"
    WHAT_TIME = "what_time"
    WORKOUT_INFO = "workout_info"
    START_WORKOUT = "start_workout"
    UNKNOWN = "unknown"

class SessionState(str, Enum):
    IDLE = "idle"
    CALIBRATING = "calibrating"
    ACTIVE = "active"
    PAUSED = "paused"
    REST = "rest"
    COMPLETED = "completed"
