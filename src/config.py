"""Configuration settings for Hand Gesture Mouse Control."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CameraConfig:
    """Webcam capture configuration."""
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 60
    flip_horizontal: bool = True  # Natural mirror view
    zoom: float = 1.0  # Digital zoom factor (1.0 = normal, 1.1 - 4.0 = zoomed in)


@dataclass
class MediaPipeConfig:
    """MediaPipe Hands detector parameters."""
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.55
    max_num_hands: int = 1
    model_complexity: int = 1  # 0: Lite, 1: Full


@dataclass
class MappingConfig:
    """Coordinate transformation and active zone parameters."""
    # Active interaction box margins inside webcam frame (pixels)
    # Allows full screen traversal without stretching hand out of view
    margin_x: int = 90
    margin_y: int = 70
    
    # Cursor acceleration / non-linear sensitivity exponent (1.0 = linear 1:1 mapping)
    acceleration_exponent: float = 1.0
    sensitivity: float = 1.0


@dataclass
class SmoothingConfig:
    """Motion jitter suppression & smoothing parameters."""
    filter_type: str = "one_euro"  # "one_euro", "ema", or "none"
    
    # One Euro filter parameters
    # min_cutoff: Cutoff frequency (Hz) at low speed (lower = silky smooth stationary cursor)
    # beta: Speed coefficient (higher = low lag during rapid hand movements)
    # d_cutoff: Cutoff frequency for derivative calculation
    min_cutoff: float = 0.50
    beta: float = 0.05
    d_cutoff: float = 1.0
    
    # Exponential Moving Average fallback
    ema_alpha: float = 0.25
    
    # Sub-pixel noise floor threshold (pixels)
    deadband_threshold: float = 2.0
    
    # Click position lock duration (seconds) - freezes cursor completely while clicking
    click_freeze_sec: float = 0.35


@dataclass
class GestureConfig:
    """Deterministic gesture detection thresholds and timing."""
    # Distances are normalized by hand scale (wrist to middle MCP distance)
    left_click_threshold: float = 0.38          # Pinky tip to Thumb tip pinch (Left Click)
    right_click_threshold: float = 0.38         # Ring tip to Thumb tip pinch (Right Click)
    drag_pinch_threshold: float = 0.50          # Relaxed for natural, effortless finger gathering (Drag & Drop)
    
    # Cooldown & timing (seconds)
    click_debounce_sec: float = 0.45            # Minimum interval between consecutive clicks
    drag_hold_sec: float = 0.08                 # Quick 80ms hold to engage drag
    drag_release_grace_frames: int = 5          # Number of non-drag frames before dropping (prevents accidental drops)
    
    # Scrolling
    scroll_threshold_y: float = 6.0             # Minimum vertical movement per frame to trigger scroll
    scroll_speed_multiplier: float = 0.02   # Drastically reduced from 1.8 for smooth, controlled scrolling


@dataclass
class UIConfig:
    """HUD visual overlay parameters."""
    show_hud: bool = True
    show_landmarks: bool = True
    show_interaction_zone: bool = True
    hud_alpha: float = 0.65
    
    # Color scheme (BGR)
    color_box: Tuple[int, int, int] = (0, 220, 255)       # Amber/Cyan
    color_active: Tuple[int, int, int] = (0, 255, 128)    # Green
    color_drag: Tuple[int, int, int] = (255, 120, 0)      # Blue
    color_click: Tuple[int, int, int] = (255, 0, 255)     # Magenta
    color_text: Tuple[int, int, int] = (255, 255, 255)    # White


@dataclass
class AppConfig:
    """Master application configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    ui: UIConfig = field(default_factory=UIConfig)
