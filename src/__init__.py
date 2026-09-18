"""Hand Gesture Mouse Control package."""

from .config import AppConfig, CameraConfig, GestureConfig, MappingConfig, SmoothingConfig, UIConfig
from .camera import CameraManager
from .hand_tracker import HandTracker, HandLandmarksResult
from .gesture_recognizer import GestureRecognizer, GestureType, GestureEvent
from .mouse_controller import MouseController, Win32MouseDriver, PyAutoGuiDriver, MockMouseDriver
from .smoothing import OneEuroFilter2D, AdaptiveEMA2D, DeadbandFilter
from .ui_overlay import UIOverlay

__all__ = [
    "AppConfig",
    "CameraConfig",
    "GestureConfig",
    "MappingConfig",
    "SmoothingConfig",
    "UIConfig",
    "CameraManager",
    "HandTracker",
    "HandLandmarksResult",
    "GestureRecognizer",
    "GestureType",
    "GestureEvent",
    "MouseController",
    "Win32MouseDriver",
    "PyAutoGuiDriver",
    "MockMouseDriver",
    "OneEuroFilter2D",
    "AdaptiveEMA2D",
    "DeadbandFilter",
    "UIOverlay",
]
