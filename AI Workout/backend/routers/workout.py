from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
from core.camera_manager import CameraManager
from core.session_manager import SessionManager
from models.enums import SessionState

router = APIRouter(prefix="/workout", tags=["Workout"])

@router.websocket("/{session_id}/stream")
async def workout_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    session = SessionManager.get_instance().get_session(session_id)
    if not session:
        await websocket.close(code=1008, reason="Session not found")
        return
        
    camera = CameraManager.get_instance()
    
    try:
        while True:
            # Check the state of the session
            if session.state == SessionState.COMPLETED:
                report = session.generate_report()
                report_dict = report.dict() if hasattr(report, "dict") else report
                
                # Make sure workout_type is serialized to string if it's an Enum
                if hasattr(report_dict.get("workout_type"), "value"):
                    report_dict["workout_type"] = report_dict["workout_type"].value
                
                await websocket.send_text(json.dumps({
                    "state": "completed",
                    "audio_cue": "Workout completed! Generating report.",
                    "report": report_dict
                }))
                break
                
            # Get latest result from camera manager
            result = camera.get_latest_result(session_id)
            if result:
                # Optionally send landmarks if requested, here we just send standard state
                data_to_send = {
                    "session_id": result["session_id"],
                    "state": result["state"].value if hasattr(result["state"], "value") else result["state"],
                    "current_set": result["current_set"],
                    "rep_count": result.get("rep_count", 0),
                    "feedback": result.get("feedback", "none").value if hasattr(result.get("feedback", "none"), "value") else result.get("feedback", "none"),
                    "feedback_message": result.get("feedback_message", ""),
                    "audio_cue": result.get("audio_cue", ""),
                    "is_recording": result.get("is_recording", False),
                    "details": result.get("details", {})
                }
                
                # Send the data to Flutter
                await websocket.send_text(json.dumps(data_to_send))
                
            # Sleep so we send results at a reasonable frame rate (e.g., 30fps)
            await asyncio.sleep(0.033)
            
    except WebSocketDisconnect:
        print(f"Client disconnected from tracking session {session_id}")
    except Exception as e:
        print(f"Error in websocket stream: {e}")
        await websocket.close()
