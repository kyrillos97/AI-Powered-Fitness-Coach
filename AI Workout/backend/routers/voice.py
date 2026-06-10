import time
import uuid
from fastapi import APIRouter, HTTPException
from models.schemas import VoiceCommandRequest, VoiceCommandResponse, StartSessionRequest, VoiceEnrollRequest
from core.voice_commands import VoiceCommandParser
from core.session_manager import SessionManager
from core.voice_identity import VoiceIdentityManager
from models.enums import VoiceCommand, SessionState, WorkoutType
import base64
import os
import tempfile

router = APIRouter(prefix="/voice", tags=["Voice Commands"])


# ══════════════════════════════════════════════════════════════════════════════
#  Voice Identity Status
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/status")
def get_voice_status():
    """Returns the current state of voice identity enrollment.
    Flutter should call this on startup to decide whether to show the enrollment screen.
    """
    identity_manager = VoiceIdentityManager.get_instance()
    return {
        "enabled": identity_manager.enabled,
        "enrolled": not identity_manager.needs_enrollment,
        "message": (
            "Voice identity is active." if not identity_manager.needs_enrollment
            else "No voice enrolled yet. Please record a short sentence to register your voice."
        )
    }

from config import CONFIG
from engines.bicep_curl_engine import BicepCurlEngine
from engines.squat_engine import SquatEngine
from engines.side_shoulder_engine import SideShoulderEngine
from engines.shrug_engine import ShrugEngine
from engines.front_shoulder_engine import FrontShoulderEngine

def _get_engine_for_workout(workout_type: WorkoutType):
    """Return the appropriate engine. Mirrors session.py logic."""
    if workout_type == WorkoutType.BICEP_CURL:
        return BicepCurlEngine(CONFIG["MODELS"]["bicep_curl"])
    elif workout_type == WorkoutType.SQUAT:
        return SquatEngine(CONFIG["MODELS"]["squat"])
    elif workout_type == WorkoutType.SIDE_SHOULDER:
        return SideShoulderEngine(CONFIG["MODELS"]["side_shoulder"])
    elif workout_type == WorkoutType.SHRUG:
        return ShrugEngine(CONFIG["MODELS"]["shrug"])
    elif workout_type == WorkoutType.FRONT_SHOULDER:
        return FrontShoulderEngine(CONFIG["MODELS"]["front_shoulder"])
    raise ValueError(f"Unknown workout type: {workout_type}")


# ══════════════════════════════════════════════════════════════════════════════
#  Global voice endpoint (no session required — handles START_WORKOUT etc.)
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/command", response_model=VoiceCommandResponse)
def global_voice_command(request: VoiceCommandRequest):
    """Handle voice commands when no session_id is known (e.g. starting a workout by voice)."""

    identity_manager = VoiceIdentityManager.get_instance()

    # ── Enrollment gate: if model is loaded but nothing enrolled yet, ask user ──
    if identity_manager.enabled and identity_manager.needs_enrollment:
        return VoiceCommandResponse(
            success=False,
            command_interpreted="needs_enrollment",
            message="Voice identity not yet enrolled.",
            audio_cue=(
                "Welcome! Before we start, I need to learn your voice. "
                "Please say a sentence out loud so I can store your voice print."
            )
        )

    # ── Speaker verification ───────────────────────────────────────────────────
    if request.audio_base64:
        if identity_manager.enabled and identity_manager.enrolled_embedding is not None:
            try:
                audio_data = base64.b64decode(request.audio_base64)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_data)
                    tmp_path = f.name
                is_user = identity_manager.verify_voice(tmp_path)
                os.remove(tmp_path)
                if not is_user:
                    return VoiceCommandResponse(
                        success=False,
                        command_interpreted="unauthorized",
                        message="Voice identity verification failed. Ignoring command.",
                        audio_cue=""
                    )
            except Exception as e:
                print(f"[VoiceID] Error decoding or verifying audio: {e}")

    command, workout_type = VoiceCommandParser.parse_command(request.command_text)
    manager = SessionManager.get_instance()

    # If there IS an active session, delegate to the session handler
    active = manager.get_active_session()
    if active and command != VoiceCommand.START_WORKOUT:
        return _handle_session_command(active, command, workout_type, request.command_text)

    if command == VoiceCommand.START_WORKOUT and workout_type:
        # End any existing session first
        if active:
            active.state = SessionState.COMPLETED
            manager.active_session_id = None

        try:
            engine = _get_engine_for_workout(workout_type)
        except Exception as e:
            return VoiceCommandResponse(
                success=False,
                command_interpreted=command.value,
                message=f"Failed to load engine: {e}",
                audio_cue="Sorry, I couldn't start that workout."
            )

        session_id = str(uuid.uuid4())
        dummy_request = StartSessionRequest(
            user_name="Voice User",
            workout_type=workout_type,
            face_embedding=[0.0] * 128,
            target_sets=3,
            target_reps_per_set=5
        )
        session = manager.create_session(session_id, dummy_request, engine)

        workout_display = workout_type.value.replace("_", " ").title()
        return VoiceCommandResponse(
            success=True,
            command_interpreted=command.value,
            message=f"Started {workout_display} session. Session ID: {session_id}",
            audio_cue=f"Starting {workout_display}! Stand in frame for calibration.",
            session_id=session_id,
            workout_started=workout_type.value
        )

    if command == VoiceCommand.START_WORKOUT and not workout_type:
        return VoiceCommandResponse(
            success=False,
            command_interpreted=command.value,
            message="Workout name not recognized.",
            audio_cue="Which workout would you like to do? Say the workout name."
        )

    return VoiceCommandResponse(
        success=False,
        command_interpreted=command.value,
        message="No active session. Say a workout name to start.",
        audio_cue="No active session. Say a workout name like bicep curl or squat to start."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Session-scoped voice endpoint
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/{session_id}/command", response_model=VoiceCommandResponse)
def handle_voice_command(session_id: str, request: VoiceCommandRequest):
    session = SessionManager.get_instance().get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # ── Speaker verification ───────────────────────────────────────────────────
    if request.audio_base64:
        identity_manager = VoiceIdentityManager.get_instance()
        if identity_manager.enabled and identity_manager.enrolled_embedding is not None:
            try:
                audio_data = base64.b64decode(request.audio_base64)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_data)
                    tmp_path = f.name
                is_user = identity_manager.verify_voice(tmp_path)
                os.remove(tmp_path)
                if not is_user:
                    return VoiceCommandResponse(
                        success=False,
                        command_interpreted="unauthorized",
                        message="Voice identity verification failed. Ignoring command.",
                        audio_cue=""
                    )
            except Exception as e:
                print(f"[VoiceID] Error decoding or verifying audio: {e}")

    command, workout_type = VoiceCommandParser.parse_command(request.command_text)
    return _handle_session_command(session, command, workout_type, request.command_text)


def _handle_session_command(session, command, workout_type, raw_text: str):
    """Core command handler shared by both endpoints."""
    success = True
    message = f"Command '{command.value}' executed."
    audio_cue = ""
    extra = {}

    if command == VoiceCommand.STOP_SESSION:
        session.state = SessionState.COMPLETED
        audio_cue = "Stopping session. Great workout!"
        SessionManager.get_instance().active_session_id = None

    elif command == VoiceCommand.PAUSE:
        if session.state in (SessionState.ACTIVE, SessionState.CALIBRATING):
            session.state = SessionState.PAUSED
            audio_cue = "Workout paused."
        else:
            success = False
            message = f"Cannot pause: currently {session.state.value}."

    elif command == VoiceCommand.RESUME:
        if session.state == SessionState.PAUSED:
            session.state = SessionState.ACTIVE
            audio_cue = "Resuming workout. Let's go!"
        else:
            success = False
            message = "Cannot resume: not paused."

    elif command == VoiceCommand.SKIP_REST:
        if session.state == SessionState.REST:
            session.start_next_set()
            audio_cue = f"Rest skipped! Starting set {session.current_set}!"
        else:
            success = False
            message = "Cannot skip rest: not in rest phase."

    elif command == VoiceCommand.NEXT_SET:
        session.start_next_set()
        audio_cue = f"Starting set {session.current_set}."

    elif command == VoiceCommand.RESET_REPS:
        session.engine.reset()
        audio_cue = "Reps reset for this set."

    elif command == VoiceCommand.HOW_MANY_REPS:
        reps = session.engine.rep_count_internal
        target = session.target_reps
        total_done = session.total_reps_session + reps
        audio_cue = (
            f"You've done {reps} reps in this set out of {target}. "
            f"Total reps this session: {total_done}. "
            f"Set {session.current_set} of {session.target_sets}."
        )
        message = audio_cue

    elif command == VoiceCommand.WHAT_TIME:
        elapsed = time.time() - session.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        if mins > 0:
            audio_cue = f"You've been working out for {mins} minutes and {secs} seconds."
        else:
            audio_cue = f"You've been working out for {secs} seconds."
        message = audio_cue

    elif command == VoiceCommand.WORKOUT_INFO:
        tips = session.feedback_engine.get_workout_tips(session.request.workout_type.value)
        audio_cue = tips
        message = tips

    elif command == VoiceCommand.OPEN_CHATBOT:
        session.state = SessionState.PAUSED
        audio_cue = "Pausing workout for chatbot."
        message = "Opened chatbot context."

    elif command == VoiceCommand.START_WORKOUT:
        # Starting a new workout mid-session
        if workout_type:
            workout_display = workout_type.value.replace("_", " ").title()
            audio_cue = f"Please stop the current session first before starting {workout_display}."
            message = f"Active session running. Stop current session first."
            success = False
        else:
            audio_cue = "Which workout would you like to do?"
            success = False

    else:
        success = False
        message = "Command not recognized."
        audio_cue = "Sorry, I didn't understand that command."

    return VoiceCommandResponse(
        success=success,
        command_interpreted=command.value,
        message=message,
        audio_cue=audio_cue
    )

# ══════════════════════════════════════════════════════════════════════════════
#  Voice Identity Enrollment
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/enroll")
def enroll_voice_identity(request: VoiceEnrollRequest):
    """Enroll a user's voice for speaker verification."""
    identity_manager = VoiceIdentityManager.get_instance()
    
    if not identity_manager.enabled:
        raise HTTPException(status_code=500, detail="Voice Identity Manager is disabled or not loaded.")
        
    try:
        audio_data = base64.b64decode(request.audio_base64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name
            
        success = identity_manager.enroll_voice(tmp_path)
        os.remove(tmp_path)
        
        if success:
            return {"success": True, "message": "Voice identity enrolled successfully."}
        else:
            raise HTTPException(status_code=500, detail="Failed to enroll voice.")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio data: {e}")
