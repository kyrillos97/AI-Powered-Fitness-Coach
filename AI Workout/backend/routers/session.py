from fastapi import APIRouter, HTTPException
import uuid
from models.schemas import StartSessionRequest, StartSessionResponse, SessionStatusResponse, SessionReportResponse
from models.enums import SessionState, WorkoutType
from core.session_manager import SessionManager
from config import CONFIG
import engines

router = APIRouter(prefix="/session", tags=["Session"])

def get_engine_for_workout(workout_type: WorkoutType):
    if workout_type == WorkoutType.BICEP_CURL:
        from engines.bicep_curl_engine import BicepCurlEngine
        return BicepCurlEngine(CONFIG["MODELS"]["bicep_curl"])
    elif workout_type == WorkoutType.SQUAT:
        from engines.squat_engine import SquatEngine
        return SquatEngine(CONFIG["MODELS"]["squat"])
    elif workout_type == WorkoutType.SIDE_SHOULDER:
        from engines.side_shoulder_engine import SideShoulderEngine
        return SideShoulderEngine(CONFIG["MODELS"]["side_shoulder"])
    elif workout_type == WorkoutType.SHRUG:
        from engines.shrug_engine import ShrugEngine
        return ShrugEngine(CONFIG["MODELS"]["shrug"])
    elif workout_type == WorkoutType.FRONT_SHOULDER:
        from engines.front_shoulder_engine import FrontShoulderEngine
        return FrontShoulderEngine(CONFIG["MODELS"]["front_shoulder"])
    
    raise ValueError("Unknown workout type")

@router.post("/start", response_model=StartSessionResponse)
def start_session(request: StartSessionRequest):
    session_id = str(uuid.uuid4())
    engine = get_engine_for_workout(request.workout_type)
    
    manager = SessionManager.get_instance()
    session = manager.create_session(session_id, request, engine)
    
    return StartSessionResponse(
        session_id=session_id,
        state=session.state,
        message="Session created. Stand in frame for calibration."
    )

@router.get("/{session_id}/status", response_model=SessionStatusResponse)
def get_status(session_id: str):
    session = SessionManager.get_instance().get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return SessionStatusResponse(
        session_id=session_id,
        user_name=session.request.user_name,
        workout_type=session.request.workout_type,
        state=session.state,
        total_reps=session.total_reps_session + session.engine.rep_count_internal,
        current_set=session.current_set,
        target_sets=session.target_sets,
        target_reps=session.target_reps,
        elapsed_seconds=0, # Need to compute properly
        history=session.history
    )

@router.post("/{session_id}/stop")
def stop_session(session_id: str):
    SessionManager.get_instance().end_session(session_id)
    return {"message": "Session stopped"}

@router.post("/{session_id}/pause")
def pause_session(session_id: str):
    session = SessionManager.get_instance().get_session(session_id)
    if session and session.state == SessionState.ACTIVE:
        session.state = SessionState.PAUSED
    return {"state": session.state if session else "NOT_FOUND"}

@router.post("/{session_id}/resume")
def resume_session(session_id: str):
    session = SessionManager.get_instance().get_session(session_id)
    if session and session.state == SessionState.PAUSED:
        session.state = SessionState.ACTIVE
    return {"state": session.state if session else "NOT_FOUND"}

@router.get("/{session_id}/report", response_model=SessionReportResponse)
def get_report(session_id: str):
    session = SessionManager.get_instance().get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.generate_report()
