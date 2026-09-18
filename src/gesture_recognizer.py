"""Deterministic gesture classification state machine with hysteresis and debouncing."""

import enum
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from .config import GestureConfig
from .hand_tracker import (
    HandLandmarksResult,
    INDEX_TIP,
    INDEX_PIP,
    INDEX_MCP,
    MIDDLE_TIP,
    MIDDLE_PIP,
    MIDDLE_MCP,
    RING_TIP,
    RING_PIP,
    RING_MCP,
    PINKY_TIP,
    PINKY_PIP,
    PINKY_MCP,
    THUMB_TIP,
    WRIST,
)


class GestureType(enum.Enum):
    """Supported hand gestures."""
    NONE = "None"
    IDLE = "Hand Detected (Idle)"
    MOVE = "Cursor Navigation"
    LEFT_CLICK = "Left Click"
    RIGHT_CLICK = "Right Click"
    SCROLL = "Scroll Up/Down"
    DRAGGING = "Drag and Drop"


@dataclass
class GestureEvent:
    """Dispatched gesture action event."""
    gesture_type: GestureType
    # Cursor target point in normalized [0, 1] frame coordinates
    cursor_norm: Tuple[float, float]
    # Vertical scroll delta (negative = up, positive = down)
    scroll_dy: float = 0.0
    # Whether dragging mouse button is currently held down
    is_dragging: bool = False
    # Whether cursor position should be frozen/locked (e.g. during pinch/click)
    lock_cursor: bool = False
    # Human-readable state description
    description: str = ""


class GestureRecognizer:
    """State machine identifying discrete mouse gestures from 21 hand landmarks."""

    def __init__(self, config: Optional[GestureConfig] = None):
        self.config = config or GestureConfig()
        
        # Debounce, latch and timing state
        self._last_left_click_time = 0.0
        self._last_right_click_time = 0.0
        self._left_pinch_latched = False
        self._right_pinch_latched = False
        self._pinch_start_time = 0.0
        # Drag state
        self._is_dragging = False
        self._pinch_start_time = 0.0
        self._drag_release_frames = 0
        
        # Scroll tracking
        self._prev_scroll_y: Optional[float] = None
        
        # Hysteresis buffer for stabilizing gesture detection
        self._current_gesture = GestureType.NONE
        self._candidate_gesture = GestureType.NONE
        self._candidate_count = 0
        self._CONFIRMATION_FRAMES = 2

    def reset(self) -> None:
        """Reset internal state when hand leaves tracking view."""
        self._is_dragging = False
        self._pinch_start_time = 0.0
        self._drag_release_frames = 0
        self._left_pinch_latched = False
        self._right_pinch_latched = False
        self._prev_scroll_y = None
        self._current_gesture = GestureType.NONE
        self._candidate_gesture = GestureType.NONE
        self._candidate_count = 0

    def classify(self, hand: HandLandmarksResult) -> GestureEvent:
        """Classify landmarks and return corresponding GestureEvent."""
        now = time.perf_counter()
        ext = hand.get_finger_extended_states()

        index_up = ext["index"]
        middle_up = ext["middle"]
        ring_up = ext["ring"]
        pinky_up = ext["pinky"]
        thumb_up = ext["thumb"]

        # Scale-invariant normalized distances
        dist_pinky_thumb = hand.distance_norm(PINKY_TIP, THUMB_TIP)
        dist_ring_thumb = hand.distance_norm(RING_TIP, THUMB_TIP)
        dist_middle_thumb = hand.distance_norm(MIDDLE_TIP, THUMB_TIP)
        dist_index_middle = hand.distance_norm(INDEX_TIP, MIDDLE_TIP)

        # Primary tracking point: Index fingertip
        cursor_norm = (hand.landmarks_norm[INDEX_TIP][0], hand.landmarks_norm[INDEX_TIP][1])

        detected = GestureType.IDLE
        scroll_dy = 0.0

        # Detailed finger states
        index_up = ext["index"]
        middle_up = ext["middle"]
        ring_up = ext["ring"]
        pinky_up = ext["pinky"]
        thumb_up = ext["thumb"]

        # Pinch detections (touching tips)
        pinky_thumb_pinched = dist_pinky_thumb < self.config.left_click_threshold
        ring_thumb_pinched = dist_ring_thumb < self.config.right_click_threshold
        middle_thumb_pinched = dist_middle_thumb < 0.38

        # --- DRAG & DROP GESTURE ("All fingers except index") ---
        # 1. Pointing index while middle, ring, pinky are folded/curled into palm
        pointing_curled = index_up and (not middle_up and not ring_up and not pinky_up)

        # 2. Pointing index while non-index fingers gather or pinch to thumb
        gathered_drag = index_up and (
            (middle_thumb_pinched and (not ring_up or not pinky_up or dist_ring_thumb < 0.45)) or
            (dist_middle_thumb < 0.48 and dist_ring_thumb < 0.48 and not middle_up) or
            (not middle_up and not ring_up)
        )

        # 3. Closed fist fallback:
        closed_fist = not index_up and not middle_up and not ring_up and not pinky_up

        drag_gesture = pointing_curled or gathered_drag or closed_fist

        # Release click latches when pinch is broken or during drag
        if not pinky_thumb_pinched or drag_gesture:
            self._left_pinch_latched = False
        if not ring_thumb_pinched or drag_gesture:
            self._right_pinch_latched = False

        # 1. Drag & Drop logic with Release Hysteresis (prevents accidental dropping mid-motion)
        if drag_gesture:
            self._drag_release_frames = 0
            if not self._is_dragging:
                if self._pinch_start_time == 0.0:
                    self._pinch_start_time = now
                elif (now - self._pinch_start_time) >= self.config.drag_hold_sec:
                    self._is_dragging = True
                    detected = GestureType.DRAGGING
                else:
                    detected = GestureType.MOVE
            else:
                detected = GestureType.DRAGGING

            # Cursor follows index fingertip
            cursor_norm = (hand.landmarks_norm[INDEX_TIP][0], hand.landmarks_norm[INDEX_TIP][1])
        else:
            if self._is_dragging:
                # Hand may have opened deliberately, or a single frame had noise
                self._drag_release_frames += 1
                hand_clearly_opened = index_up and middle_up and ring_up
                if hand_clearly_opened or self._drag_release_frames >= self.config.drag_release_grace_frames:
                    self._is_dragging = False
                    self._drag_release_frames = 0
                    self._pinch_start_time = 0.0
                    detected = GestureType.MOVE
                else:
                    # Maintain drag during transient tracking gaps
                    detected = GestureType.DRAGGING
            else:
                self._pinch_start_time = 0.0

        can_click = not self._is_dragging and not drag_gesture

        # 2. Left Click: Pinky + Thumb pinch ONLY when middle finger is UP (hand open)
        if can_click and pinky_thumb_pinched and not ring_thumb_pinched and middle_up:
            if not self._left_pinch_latched and (now - self._last_left_click_time) >= self.config.click_debounce_sec:
                detected = GestureType.LEFT_CLICK
                self._last_left_click_time = now
                self._left_pinch_latched = True
            else:
                detected = GestureType.MOVE

        # 3. Right Click: Ring + Thumb pinch ONLY when middle finger is UP (hand open)
        elif can_click and ring_thumb_pinched and not pinky_thumb_pinched and middle_up:
            if not self._right_pinch_latched and (now - self._last_right_click_time) >= self.config.click_debounce_sec:
                detected = GestureType.RIGHT_CLICK
                self._last_right_click_time = now
                self._right_pinch_latched = True
            else:
                detected = GestureType.MOVE

        # 4. Scroll: Index + Middle fingers upright together (not pinching thumb)
        elif not self._is_dragging and not drag_gesture and not middle_thumb_pinched and (
            (index_up and middle_up and not ring_up and not pinky_up) or
            (index_up and middle_up and dist_index_middle < 0.35 and not pinky_thumb_pinched and not ring_thumb_pinched)
        ):
            detected = GestureType.SCROLL
            current_scroll_y = (hand.landmarks_norm[INDEX_TIP][1] + hand.landmarks_norm[MIDDLE_TIP][1]) / 2.0
            if self._prev_scroll_y is not None:
                dy = current_scroll_y - self._prev_scroll_y
                pixel_dy = dy * 480.0
                if abs(pixel_dy) >= self.config.scroll_threshold_y:
                    scroll_dy = -pixel_dy * self.config.scroll_speed_multiplier
            self._prev_scroll_y = current_scroll_y

        # 5. Cursor Navigation: Hand open with index finger upright
        elif not self._is_dragging and not drag_gesture and index_up and not pinky_thumb_pinched and not ring_thumb_pinched:
            detected = GestureType.MOVE
            self._prev_scroll_y = None

        else:
            # Idle hand or unrecognized configuration
            if not self._is_dragging:
                self._prev_scroll_y = None
                if detected == GestureType.IDLE and not (index_up or middle_up or ring_up or pinky_up):
                    detected = GestureType.IDLE

        # Stabilize detection with multi-frame confirmation
        if self._current_gesture in (GestureType.LEFT_CLICK, GestureType.RIGHT_CLICK):
            # Click is an instantaneous 1-frame impulse; immediately clear it to avoid repeat clicks
            self._current_gesture = detected
            self._candidate_gesture = detected
            self._candidate_count = 1
        elif detected == self._candidate_gesture:
            self._candidate_count += 1
            if self._candidate_count >= self._CONFIRMATION_FRAMES:
                self._current_gesture = detected
        else:
            self._candidate_gesture = detected
            self._candidate_count = 1

        # High priority actions trigger immediately without lag
        if detected in (GestureType.LEFT_CLICK, GestureType.RIGHT_CLICK, GestureType.DRAGGING):
            self._current_gesture = detected

        # Freeze cursor when pinching to click or during initial drag lock
        # NEVER freeze during active dragging
        lock_cursor = (
            (not self._is_dragging and (pinky_thumb_pinched or ring_thumb_pinched)) or
            (self._current_gesture in (GestureType.LEFT_CLICK, GestureType.RIGHT_CLICK))
        )

        return GestureEvent(
            gesture_type=self._current_gesture,
            cursor_norm=cursor_norm,
            scroll_dy=scroll_dy,
            is_dragging=self._is_dragging,
            lock_cursor=lock_cursor,
            description=self._current_gesture.value,
        )
