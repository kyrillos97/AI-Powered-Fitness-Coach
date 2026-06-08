from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .enums import WorkoutType, FeedbackType, SessionState

class StartSessionRequest(BaseModel):
    user_name: str
    workout_type: WorkoutType
    face_embedding: List[float] # 128 elements usually
    target_sets: int
    target_reps_per_set: int

class StartSessionResponse(BaseModel):
    session_id: str
    state: SessionState
    message: str

class VoiceCommandRequest(BaseModel):
    command_text: str
    audio_base64: Optional[str] = None

class VoiceEnrollRequest(BaseModel):
    audio_base64: str

class VoiceCommandResponse(BaseModel):
    success: bool
    command_interpreted: str
    message: str
    audio_cue: Optional[str] = None
    session_id: Optional[str] = None
    workout_started: Optional[str] = None

class RepResult(BaseModel):
    rep_number: int
    feedback: FeedbackType
    confidence: float
    details: Dict[str, Any] = {}

class FrameResponse(BaseModel):
    session_id: str
    state: SessionState
    current_set: int
    rep_count: int
    is_recording: bool
    feedback: FeedbackType
    feedback_message: str
    audio_cue: str
    landmarks: Optional[List[Dict[str, float]]] = None
    rep_history: List[RepResult] = []

class SessionStatusResponse(BaseModel):
    session_id: str
    user_name: str
    workout_type: WorkoutType
    state: SessionState
    total_reps: int
    current_set: int
    target_sets: int
    target_reps: int
    elapsed_seconds: float
    history: List[RepResult]

class SessionReportResponse(BaseModel):
    session_id: str
    user_name: str
    workout_type: WorkoutType
    target_sets: int
    target_reps: int
    actual_sets: int
    actual_reps: int
    perfect_reps: int
    total_failures: int
    failure_breakdown: Dict[str, int]
    duration_seconds: float
    message: str
