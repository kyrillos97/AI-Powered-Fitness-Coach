import numpy as np
import math

class SmartTracker:
    def __init__(self, face_embedding: list, threshold: float = 0.2):
        self.enrolled_embedding = np.array(face_embedding) if face_embedding else None
        self.active = False
        self.last_hip_x = 0.5
        self.last_hip_y = 0.5
        self.missed_frames = 0
        self.threshold = threshold
        self.verified_once = False

    def lock_on(self, landmarks):
        """Initial lock when face is verified"""
        self.active = True
        self.missed_frames = 0
        self.update_position(landmarks)
        self.verified_once = True

    def update_position(self, landmarks):
        """Update position based on new frame landmarks"""
        # Midpoint of hips (Landmarks 23 and 24)
        hip_x = (landmarks[23][0] + landmarks[24][0]) / 2
        hip_y = (landmarks[23][1] + landmarks[24][1]) / 2
        self.last_hip_x = hip_x
        self.last_hip_y = hip_y
        self.missed_frames = 0

    def is_user(self, landmarks):
        """
        Checks if the skeleton in the current frame is close enough 
        to where the user was in the last frame.
        """
        hip_x = (landmarks[23][0] + landmarks[24][0]) / 2
        hip_y = (landmarks[23][1] + landmarks[24][1]) / 2
        
        dist = math.hypot(hip_x - self.last_hip_x, hip_y - self.last_hip_y)
        return dist < self.threshold

    def verify_embedding(self, incoming_embedding: list, match_threshold: float = 0.5) -> bool:
        if not self.enrolled_embedding is not None or not incoming_embedding:
            return False
        
        incoming = np.array(incoming_embedding)
        
        # Cosine similarity
        dot = np.dot(self.enrolled_embedding, incoming)
        norm_a = np.linalg.norm(self.enrolled_embedding)
        norm_b = np.linalg.norm(incoming)
        
        if norm_a == 0 or norm_b == 0:
            return False
            
        similarity = dot / (norm_a * norm_b)
        # Using similarity > threshold (higher is better for cosine)
        # In demo.py it used face_distance < MATCH_THRESHOLD. 
        # For cosine similarity, > 0.5 is a common threshold for match
        return similarity > match_threshold
