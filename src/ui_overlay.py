"""Augmented Reality HUD visualizer and feedback overlay."""

import math
from typing import Optional, Tuple
import cv2
import numpy as np
from .config import UIConfig, MappingConfig
from .gesture_recognizer import GestureType, GestureEvent
from .hand_tracker import HandLandmarksResult, INDEX_TIP, THUMB_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP


class UIOverlay:
    """Renders augmented reality HUD, interaction zones, gesture badges, and visual feedback."""

    def __init__(self, ui_config: Optional[UIConfig] = None, mapping_config: Optional[MappingConfig] = None):
        self.cfg = ui_config or UIConfig()
        self.map_cfg = mapping_config or MappingConfig()
        self._ripple_radius = 0
        self._ripple_pos: Optional[Tuple[int, int]] = None

    def render(
        self,
        frame: np.ndarray,
        fps: float,
        hand_result: Optional[HandLandmarksResult],
        gesture_event: Optional[GestureEvent],
        screen_pos: Optional[Tuple[int, int]] = None,
        filter_name: str = "OneEuro",
        zoom: float = 1.0,
    ) -> np.ndarray:
        """Render complete HUD overlay onto frame."""
        h, w, _ = frame.shape

        # 1. Draw Active Interaction Zone
        if self.cfg.show_interaction_zone:
            self._draw_interaction_zone(frame, w, h, gesture_event)

        # 2. Draw Hand Highlights and Visual Ripples
        if hand_result is not None:
            self._draw_fingertip_highlights(frame, hand_result, gesture_event)

        # 3. Draw Top Information HUD Banner
        if self.cfg.show_hud:
            self._draw_hud_banner(frame, w, h, fps, gesture_event, screen_pos, filter_name, zoom)

        return frame

    def _draw_interaction_zone(self, frame: np.ndarray, w: int, h: int, event: Optional[GestureEvent]) -> None:
        """Draw active boundary box with glowing styled corner brackets."""
        mx = self.map_cfg.margin_x
        my = self.map_cfg.margin_y
        x1, y1 = mx, my
        x2, y2 = w - mx, h - my

        # Box color changes dynamically based on active state
        box_color = (80, 80, 80)  # Subtle gray default
        if event and event.gesture_type != GestureType.NONE and event.gesture_type != GestureType.IDLE:
            box_color = self.cfg.color_box

        # Draw semi-transparent boundary rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw decorative corner brackets for modern AR aesthetic
        corner_len = 20
        bracket_color = self.cfg.color_box
        bracket_thick = 2
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), bracket_color, bracket_thick)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), bracket_color, bracket_thick)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), bracket_color, bracket_thick)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), bracket_color, bracket_thick)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), bracket_color, bracket_thick)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), bracket_color, bracket_thick)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), bracket_color, bracket_thick)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), bracket_color, bracket_thick)

        # Label active zone
        cv2.putText(
            frame, "ACTIVE INTERACTION ZONE", (x1 + 6, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, bracket_color, 1, cv2.LINE_AA
        )

    def _draw_fingertip_highlights(
        self,
        frame: np.ndarray,
        hand: HandLandmarksResult,
        event: Optional[GestureEvent],
    ) -> None:
        """Render glowing reticles and feedback ripples on fingertips."""
        idx_pt = hand.landmarks_pixel[INDEX_TIP]
        mid_pt = hand.landmarks_pixel[MIDDLE_TIP]
        rng_pt = hand.landmarks_pixel[RING_TIP]
        pky_pt = hand.landmarks_pixel[PINKY_TIP]
        thm_pt = hand.landmarks_pixel[THUMB_TIP]

        gesture = event.gesture_type if event else GestureType.NONE

        if gesture == GestureType.LEFT_CLICK:
            # Connect pinky and thumb with left-click indicator line and ripple
            cv2.line(frame, pky_pt, thm_pt, (255, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(frame, pky_pt, 12, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, thm_pt, 12, (255, 0, 255), 2, cv2.LINE_AA)
            self._ripple_pos = pky_pt
            self._ripple_radius = 5

        elif gesture == GestureType.RIGHT_CLICK:
            # Connect ring finger and thumb with right-click indicator line
            cv2.line(frame, rng_pt, thm_pt, (0, 165, 255), 3, cv2.LINE_AA)
            cv2.circle(frame, rng_pt, 12, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, thm_pt, 12, (0, 165, 255), 2, cv2.LINE_AA)

        elif gesture == GestureType.DRAGGING:
            # Connect all non-index fingers (middle, ring, pinky) to thumb
            for pt in (mid_pt, rng_pt, pky_pt):
                cv2.line(frame, pt, thm_pt, (255, 120, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, pt, 8, (255, 120, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, thm_pt, 12, (255, 120, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, idx_pt, 10, (255, 120, 0), -1)
            cv2.circle(frame, idx_pt, 16, (255, 255, 255), 2, cv2.LINE_AA)

        elif gesture == GestureType.SCROLL:
            # Up/down scroll indicators
            cv2.circle(frame, idx_pt, 8, (0, 255, 255), -1)
            cv2.circle(frame, mid_pt, 8, (0, 255, 255), -1)
            cv2.line(frame, idx_pt, mid_pt, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            # Standard navigation target reticle
            cv2.circle(frame, idx_pt, 8, (0, 255, 128), 2, cv2.LINE_AA)
            cv2.circle(frame, idx_pt, 3, (0, 255, 128), -1)

        # Animate expanding click ripple if active
        if self._ripple_pos and self._ripple_radius < 25:
            cv2.circle(frame, self._ripple_pos, self._ripple_radius, (255, 255, 255), 1, cv2.LINE_AA)
            self._ripple_radius += 4
        else:
            self._ripple_pos = None

    def _draw_hud_banner(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        fps: float,
        event: Optional[GestureEvent],
        screen_pos: Optional[Tuple[int, int]],
        filter_name: str,
        zoom: float = 1.0,
    ) -> np.ndarray:
        """Render semi-transparent modern glassmorphic HUD header."""
        header_h = 56
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, self.cfg.hud_alpha, frame, 1.0 - self.cfg.hud_alpha, 0, frame)

        # Bottom accent divider line
        cv2.line(frame, (0, header_h), (w, header_h), (60, 60, 75), 1)

        # 1. State / Gesture Badge
        gesture_text = "SEARCHING FOR HAND..."
        badge_color = (120, 120, 120)
        if event:
            gesture = event.gesture_type
            if gesture == GestureType.MOVE:
                gesture_text = "CURSOR NAVIGATION"
                badge_color = (0, 230, 100)
            elif gesture == GestureType.LEFT_CLICK:
                gesture_text = "LEFT CLICK [LOCKED]"
                badge_color = (255, 50, 255)
            elif gesture == GestureType.RIGHT_CLICK:
                gesture_text = "RIGHT CLICK [LOCKED]"
                badge_color = (0, 165, 255)
            elif gesture == GestureType.SCROLL:
                dir_str = "UP" if event.scroll_dy > 0 else ("DOWN" if event.scroll_dy < 0 else "")
                gesture_text = f"SCROLLING {dir_str}".strip()
                badge_color = (0, 230, 255)
            elif gesture == GestureType.DRAGGING:
                gesture_text = "DRAG & DROP (HOLD)"
                badge_color = (255, 120, 0)
            elif gesture == GestureType.IDLE:
                gesture_text = "HAND IDLE"
                badge_color = (180, 180, 180)

        # Draw Gesture Badge Pill
        pill_w = 190
        pill_h = 28
        pill_x = 12
        pill_y = 14
        cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), badge_color, -1)
        cv2.putText(
            frame, gesture_text, (pill_x + 8, pill_y + 19),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA
        )

        # 2. Zoom Indicator Badge
        zoom_x = pill_x + pill_w + 10
        zoom_w = 90
        zoom_y = 14
        zoom_color = (0, 180, 255) if zoom > 1.05 else (70, 70, 80)
        zoom_text_color = (0, 0, 0) if zoom > 1.05 else (200, 200, 200)
        cv2.rectangle(frame, (zoom_x, zoom_y), (zoom_x + zoom_w, zoom_y + pill_h), zoom_color, -1)
        cv2.putText(
            frame, f"ZOOM {zoom:.1f}x", (zoom_x + 8, zoom_y + 19),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, zoom_text_color, 1, cv2.LINE_AA
        )

        # 3. Performance Metrics (FPS & Estimated Latency)
        latency_ms = (1000.0 / fps) if fps > 0 else 33.3
        fps_str = f"FPS: {fps:4.1f} | Latency: ~{latency_ms:3.0f}ms"
        fps_color = (0, 255, 0) if fps >= 28.0 else (0, 215, 255)
        cv2.putText(
            frame, fps_str, (w - 240, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1, cv2.LINE_AA
        )

        # 4. Filter & Coordinate Display
        coord_str = f"Screen: ({screen_pos[0]}, {screen_pos[1]})" if screen_pos else "Screen: ---"
        filter_str = f"Filter: {filter_name} | {coord_str}"
        cv2.putText(
            frame, filter_str, (w - 240, 44),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 190), 1, cv2.LINE_AA
        )

        # 5. Keyboard Shortcuts Footer Bar
        cv2.putText(
            frame, "[Q/ESC] Quit  |  [H] HUD  |  [S] Filter  |  [Z] / [+/-] Zoom",
            (14, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1, cv2.LINE_AA
        )

        return frame
