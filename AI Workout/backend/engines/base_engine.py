from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel
from models.enums import FeedbackType

class FrameResult(BaseModel):
    rep_count: int
    feedback: FeedbackType
    confidence: float
    is_recording: bool
    details: Dict[str, Any] = {}

class BaseWorkoutEngine(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.rep_count_internal = 0
        self.initialize()

    @abstractmethod
    def initialize(self) -> None:
        """Load models and setup state"""
        pass
    
    @abstractmethod
    def process_frame(self, landmarks: List[List[float]]) -> FrameResult:
        """Process a single frame of landmarks representing (33, 4) array"""
        pass
    
    def reset(self) -> None:
        """Reset rep count and state for a new set"""
        self.rep_count_internal = 0

    def draw_custom_visuals(self, render_frame, landmarks: List[List[float]]) -> None:
        """Override to draw engine-specific visual indicators (e.g. angle arcs, colored bones) on the OpenCV frame."""
        pass
