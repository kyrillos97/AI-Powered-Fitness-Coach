import time
from typing import Dict, Optional, List
from models.schemas import StartSessionRequest, SessionStatusResponse, RepResult, SessionReportResponse
from models.enums import SessionState, FeedbackType
from core.feedback_engine import FeedbackEngine
from core.face_verification import SmartTracker

class WorkoutSession:
    def __init__(self, session_id: str, request: StartSessionRequest, engine):
        self.session_id = session_id
        self.request = request
        self.engine = engine
        
        self.state = SessionState.CALIBRATING
        self.current_set = 1
        self.target_sets = request.target_sets
        self.target_reps = request.target_reps_per_set
        
        self.total_reps_session = 0
        self.history: List[RepResult] = []
        
        self.start_time = time.time()
        self.rest_start_time = 0
        self.rest_duration_sec = 60 # Default rest
        
        self.feedback_engine = FeedbackEngine()
        self.tracker = SmartTracker(request.face_embedding)
        
        print(f"Session {session_id} created for {request.user_name} doing {request.workout_type}")

    def update_state(self):
        """Called every frame or loop to check state transitions like rest timers."""
        if self.state == SessionState.REST:
            if time.time() - self.rest_start_time > self.rest_duration_sec:
                self.start_next_set()
                return "Rest over. Next set starting!"

        # Check if set is complete
        is_completed = False
        if self.state == SessionState.ACTIVE and self.engine.rep_count_internal >= self.target_reps:
            is_completed = True
            if self.current_set >= self.target_sets:
                self.state = SessionState.COMPLETED
                self.total_reps_session += self.engine.rep_count_internal
            else:
                self.state = SessionState.REST
                self.rest_start_time = time.time()
                self.total_reps_session += self.engine.rep_count_internal
                
            return self.feedback_engine.generate_set_rest_cue(self.current_set, self.state == SessionState.COMPLETED)
        return None

    def start_next_set(self):
        if self.current_set < self.target_sets:
            self.current_set += 1
            self.state = SessionState.ACTIVE
            self.engine.reset() # Assuming engine has a reset method for rep counts

    def generate_report(self) -> SessionReportResponse:
        perfect = sum(1 for r in self.history if r.feedback == FeedbackType.PERFECT)
        failures = len(self.history) - perfect
        
        breakdown = {}
        for r in self.history:
            if r.feedback != FeedbackType.PERFECT:
                breakdown[r.feedback.value] = breakdown.get(r.feedback.value, 0) + 1

        return SessionReportResponse(
            session_id=self.session_id,
            user_name=self.request.user_name,
            workout_type=self.request.workout_type,
            target_sets=self.target_sets,
            target_reps=self.target_reps,
            actual_sets=self.current_set,
            actual_reps=self.total_reps_session + (self.engine.rep_count_internal if self.state != SessionState.COMPLETED else 0),
            perfect_reps=perfect,
            total_failures=failures,
            failure_breakdown=breakdown,
            duration_seconds=time.time() - self.start_time,
            message="Great session!" if perfect > failures else "Good effort! Check the breakdown to improve form."
        )

class SessionManager:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        self.sessions: Dict[str, WorkoutSession] = {}
        self.active_session_id: Optional[str] = None # Assuming one active session handled by camera

    def create_session(self, session_id: str, request: StartSessionRequest, engine) -> WorkoutSession:
        session = WorkoutSession(session_id, request, engine)
        self.sessions[session_id] = session
        self.active_session_id = session_id
        return session

    def get_session(self, session_id: str) -> Optional[WorkoutSession]:
        return self.sessions.get(session_id)
        
    def get_active_session(self) -> Optional[WorkoutSession]:
        if self.active_session_id:
            return self.sessions.get(self.active_session_id)
        return None
        
    def end_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].state = SessionState.COMPLETED
            if self.active_session_id == session_id:
                self.active_session_id = None
