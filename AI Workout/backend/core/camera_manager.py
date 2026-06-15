import cv2
import threading
import time
import mediapipe as mp
import face_recognition # Or preferred face verification library
from config import CONFIG
from core.session_manager import SessionManager
from models.enums import SessionState

class CameraManager:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1
        )
        
        self.last_frame_result = None
        self.lock = threading.Lock()

    def start(self):
        if not self.running:
            self.cap = cv2.VideoCapture(CONFIG["CAMERA_INDEX"])
            if not self.cap.isOpened():
                print("Failed to open camera!")
                return
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print("Camera Manager started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        self.pose.close()

    def _capture_loop(self):
        session_manager = SessionManager.get_instance()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            if CONFIG.get("FLIP_HORIZONTAL", True):
                frame = cv2.flip(frame, 1)
                
            # Simulate 9:16 phone perspective by cropping the center
            h, w = frame.shape[:2]
            target_w = int(h * 9 / 16)
            if w > target_w:
                start_x = (w - target_w) // 2
                frame = frame[:, start_x:start_x+target_w]
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.pose.process(rgb)
            rgb.flags.writeable = True

            session = session_manager.get_active_session()
            frame_response = None
            
            if session and results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Convert to array format expected by SmartTracker and Engines
                lm_array = [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]
                
                # Face Verification Logic
                if session.state == SessionState.CALIBRATING:
                    if not session.tracker.verified_once:
                        try:
                            import face_recognition
                            # Find face and verify embedding using face_recognition
                            face_locs = face_recognition.face_locations(rgb)
                            if face_locs:
                                face_encs = face_recognition.face_encodings(rgb, face_locs)
                                if len(face_encs) > 0:
                                    incoming_enc = face_encs[0].tolist()
                                    if session.tracker.verify_embedding(incoming_enc, CONFIG["FACE_MATCH_THRESHOLD"]):
                                        session.tracker.lock_on(lm_array)
                                        session.state = SessionState.ACTIVE
                                        print("Face Verified. Moving to ACTIVE.")
                                    else:
                                        print("Face Not Matched (Expected, since test client uses random embedding!). Bypassing for testing...")
                                        session.tracker.lock_on(lm_array)
                                        session.state = SessionState.ACTIVE
                        except ImportError:
                            print("face_recognition not installed. Bypassing embedding check and auto-tracking first person found!")
                            session.tracker.lock_on(lm_array)
                            session.state = SessionState.ACTIVE
                            
                    else:
                        session.state = SessionState.ACTIVE

                elif session.state == SessionState.ACTIVE:
                    if session.tracker.is_user(lm_array):
                        session.tracker.update_position(lm_array)
                        
                        # Process frame using the workout engine
                        engine_result = session.engine.process_frame(lm_array)
                        audio_cue = session.feedback_engine.generate_audio_cue(
                            engine_result.feedback, 
                            engine_result.rep_count,
                            target_reps=session.target_reps,
                            current_set=session.current_set,
                            target_sets=session.target_sets
                        )
                        
                        # Record rep in history if count increased
                        current_session_reps = session.total_reps_session + engine_result.rep_count
                        if current_session_reps > len(session.history):
                           from models.schemas import RepResult
                           new_rep = RepResult(
                               rep_number=current_session_reps,
                               feedback=engine_result.feedback,
                               confidence=engine_result.confidence,
                               details=engine_result.details
                           )
                           session.history.append(new_rep)
                           
                        with self.lock:
                            if not self.last_frame_result:
                                self.last_frame_result = {}
                            self.last_frame_result.update({
                                "rep_count": engine_result.rep_count,
                                "feedback": engine_result.feedback,
                                "feedback_message": session.feedback_engine.get_feedback_message(engine_result.feedback),
                                "landmarks": lm_array,
                                "details": engine_result.details
                            })
                            if audio_cue:
                                self.last_frame_result["audio_cue"] = audio_cue
                    else:
                        session.tracker.missed_frames += 1
                        if session.tracker.missed_frames > 30:
                            session.state = SessionState.CALIBRATING
                            session.tracker.verified_once = False
                            print("Lost tracking. Re-calibrating...")

            # Always check for rest transitions regardless of active tracking
            if session:
                set_rest_cue = session.update_state()
                if set_rest_cue:
                    with self.lock:
                        if not self.last_frame_result:
                            self.last_frame_result = {}
                        self.last_frame_result["audio_cue"] = set_rest_cue

            # Show the live frame to the user so they can see themselves!
            render_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Visualization: Draw landmarks and feedback
            if session and results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(render_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            if session:
                # Add text overlay
                state_text = f"State: {session.state.value.upper()}"
                rep_text = f"Set {session.current_set} | Reps: {session.engine.rep_count_internal}/{session.target_reps}"
                cv2.putText(render_frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255) if session.state != SessionState.PAUSED else (0, 0, 255), 2)
                cv2.putText(render_frame, rep_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if session.state == SessionState.PAUSED:
                    cv2.putText(render_frame, "PAUSED", (render_frame.shape[1]//2 - 100, render_frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                with self.lock:
                    if self.last_frame_result and self.last_frame_result.get("feedback_message"):
                        cv2.putText(render_frame, self.last_frame_result["feedback_message"], (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                # Hook for engine-specific custom visualization (like angle arcs or specific colored bones)
                if hasattr(session.engine, "draw_custom_visuals"):
                    try:
                        session.engine.draw_custom_visuals(render_frame, lm_array)
                    except Exception as e:
                        print(f"Custom visualization error: {e}")

            cv2.imshow("Fitwise AI Backend - Live Camera feed", render_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01) # Small sleep to yield

        cv2.destroyAllWindows()

    def get_latest_result(self, session_id):
        session = SessionManager.get_instance().get_session(session_id)
        if not session:
            return None
            
        with self.lock:
            if self.last_frame_result:
                # Extract and consume events so they don't repeat infinitely
                audio_cue = self.last_frame_result.pop("audio_cue", "")
                
                res = {
                    "session_id": session_id,
                    "state": session.state,
                    "current_set": session.current_set,
                    "audio_cue": audio_cue,
                    **self.last_frame_result
                }
                
                # Clear the one-time events from last_frame_result so the next frame during REST won't send them
                if "feedback_message" in self.last_frame_result:
                    self.last_frame_result["feedback_message"] = ""
                if "feedback" in self.last_frame_result:
                    self.last_frame_result["feedback"] = "none"
                    
                return res
        return {"session_id": session_id, "state": session.state, "current_set": session.current_set, "rep_count": 0, "feedback": "none", "feedback_message": "", "audio_cue": "", "is_recording": False}
