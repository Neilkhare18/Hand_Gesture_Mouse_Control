"""Comprehensive automated unit test suite for Hand Gesture Mouse Control pipeline."""

import math
import unittest
import numpy as np

from src.config import AppConfig, MappingConfig, SmoothingConfig, GestureConfig
from src.smoothing import OneEuroFilter1D, OneEuroFilter2D, AdaptiveEMA2D, DeadbandFilter
from src.camera import CameraManager
from src.hand_tracker import (
    HandLandmarksResult,
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
    THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP,
    MIDDLE_MCP, PINKY_MCP,
)
from src.gesture_recognizer import GestureRecognizer, GestureType
from src.mouse_controller import MouseController, MockMouseDriver
from src.ui_overlay import UIOverlay


def create_synthetic_hand(
    index_up: bool = True,
    middle_up: bool = False,
    ring_up: bool = False,
    pinky_up: bool = False,
    thumb_up: bool = False,
    index_tip_x: float = 0.5,
    index_tip_y: float = 0.3,
    middle_tip_x: float = 0.55,
    middle_tip_y: float = 0.5,
    ring_tip_x: float = 0.55,
    ring_tip_y: float = 0.35,
    pinky_tip_x: float = 0.60,
    pinky_tip_y: float = 0.38,
    thumb_tip_x: float = 0.4,
    thumb_tip_y: float = 0.5,
) -> HandLandmarksResult:
    """Helper to construct 21 synthetic normalized landmarks matching hand topology."""
    # 21 points initialized around wrist (0.5, 0.8)
    norm_pts = [(0.5, 0.8, 0.0) for _ in range(21)]
    pixel_pts = [(int(x * 640), int(y * 480)) for x, y, _ in norm_pts]

    # Wrist
    norm_pts[WRIST] = (0.5, 0.8, 0.0)
    # Middle MCP (landmark 9) - reference scale = 0.25
    norm_pts[MIDDLE_MCP] = (0.5, 0.55, 0.0)
    # Pinky MCP (landmark 17)
    norm_pts[PINKY_MCP] = (0.6, 0.60, 0.0)

    # PIP joints (finger bases)
    norm_pts[INDEX_PIP] = (0.45, 0.50, 0.0)
    norm_pts[MIDDLE_PIP] = (0.50, 0.50, 0.0)
    norm_pts[RING_PIP] = (0.55, 0.52, 0.0)
    norm_pts[PINKY_PIP] = (0.60, 0.55, 0.0)
    norm_pts[THUMB_IP] = (0.42, 0.65, 0.0)

    # Fingertips (tip Y < pip Y means finger is extended UP in screen coordinates)
    norm_pts[INDEX_TIP] = (index_tip_x, index_tip_y if index_up else 0.65, 0.0)
    norm_pts[MIDDLE_TIP] = (middle_tip_x, middle_tip_y if middle_up else 0.65, 0.0)
    norm_pts[RING_TIP] = (ring_tip_x, ring_tip_y if ring_up else 0.67, 0.0)
    norm_pts[PINKY_TIP] = (pinky_tip_x, pinky_tip_y if pinky_up else 0.70, 0.0)
    norm_pts[THUMB_TIP] = (thumb_tip_x, thumb_tip_y, 0.0)

    # Update pixel coordinates
    for i, (x, y, _) in enumerate(norm_pts):
        pixel_pts[i] = (int(x * 640), int(y * 480))

    hand_scale = math.hypot(
        norm_pts[WRIST][0] - norm_pts[MIDDLE_MCP][0],
        norm_pts[WRIST][1] - norm_pts[MIDDLE_MCP][1],
    )

    return HandLandmarksResult(
        landmarks_norm=norm_pts,
        landmarks_pixel=pixel_pts,
        handedness="Right",
        hand_scale=hand_scale,
    )


class TestSmoothingFilters(unittest.TestCase):
    """Test mathematical behavior of motion filters."""

    def test_one_euro_filter_reduces_jitter(self):
        f = OneEuroFilter1D(min_cutoff=1.0, beta=0.01)
        # Feed high-frequency jitter around a constant value of 100.0
        t = 0.0
        outputs = []
        for i in range(30):
            noise = 2.0 if (i % 2 == 0) else -2.0
            val = 100.0 + noise
            out = f.filter(val, timestamp=t)
            outputs.append(out)
            t += 0.033  # 30 FPS

        # The filtered output jitter variance should be substantially smaller than raw jitter
        raw_deviations = [abs(x - 100.0) for x in [102.0, 98.0]]
        filtered_deviations = [abs(x - 100.0) for x in outputs[-10:]]
        self.assertLess(max(filtered_deviations), max(raw_deviations))

    def test_one_euro_2d_filter(self):
        f2d = OneEuroFilter2D()
        x, y = f2d.filter(100.0, 200.0, timestamp=0.0)
        self.assertEqual(x, 100.0)
        self.assertEqual(y, 200.0)

        # Move to (150, 250)
        x2, y2 = f2d.filter(150.0, 250.0, timestamp=0.033)
        self.assertTrue(100.0 < x2 < 150.0)
        self.assertTrue(200.0 < y2 < 250.0)

    def test_deadband_filter(self):
        db = DeadbandFilter(threshold=2.0)
        # First point
        x1, y1 = db.filter(50.0, 50.0)
        self.assertEqual((x1, y1), (50.0, 50.0))

        # Micro-tremor delta < 2.0
        x2, y2 = db.filter(50.8, 50.6)
        self.assertEqual((x2, y2), (50.0, 50.0))  # Suppressed!

        # Substantial movement > 2.0
        x3, y3 = db.filter(55.0, 55.0)
        self.assertEqual((x3, y3), (55.0, 55.0))  # Accepted!


class TestMouseControllerAndMapping(unittest.TestCase):
    """Test coordinate transformation and active interaction zone margins."""

    def setUp(self):
        self.mock_driver = MockMouseDriver(width=1920, height=1080)
        self.mapping_cfg = MappingConfig(margin_x=100, margin_y=80, acceleration_exponent=1.0)
        self.smoothing_cfg = SmoothingConfig(filter_type="none", deadband_threshold=0.0)
        self.ctrl = MouseController(
            mapping_config=self.mapping_cfg,
            smoothing_config=self.smoothing_cfg,
            driver=self.mock_driver,
            cam_width=640,
            cam_height=480,
        )

    def test_center_coordinate_mapping(self):
        # Center in webcam (0.5, 0.5) should map to center of screen (960, 540)
        sx, sy = self.ctrl.move(0.5, 0.5)
        self.assertAlmostEqual(sx, 960, delta=2)
        self.assertAlmostEqual(sy, 540, delta=2)
        self.assertEqual(self.mock_driver.cur_x, sx)
        self.assertEqual(self.mock_driver.cur_y, sy)

    def test_margin_clamping_to_corners(self):
        # Coordinates at or outside active margin boundary should clamp cleanly to screen limits
        # Left-Top margin point (margin_x/640, margin_y/480) = (100/640, 80/480)
        sx, sy = self.ctrl.move(100 / 640.0, 80 / 480.0)
        self.assertEqual(sx, 0)
        self.assertEqual(sy, 0)

        # Right-Bottom margin point ((640-100)/640, (480-80)/480)
        sx2, sy2 = self.ctrl.move(540 / 640.0, 400 / 480.0)
        self.assertEqual(sx2, 1919)
        self.assertEqual(sy2, 1079)

    def test_mouse_clicks_and_drag(self):
        self.ctrl.left_click()
        self.assertEqual(self.mock_driver.click_count, 1)

        self.ctrl.right_click()
        self.assertEqual(self.mock_driver.right_click_count, 1)

        self.ctrl.set_drag_state(True)
        self.assertTrue(self.mock_driver.is_down)

        self.ctrl.set_drag_state(False)
        self.assertFalse(self.mock_driver.is_down)

        self.ctrl.scroll(2)
        self.assertEqual(self.mock_driver.scroll_total, 2)

    def test_click_position_lock(self):
        # Move to center position
        sx, sy = self.ctrl.move(0.5, 0.5)
        # Trigger left click with freeze duration
        self.ctrl.left_click(freeze_sec=0.15)
        # Attempt to move while frozen
        locked_x, locked_y = self.ctrl.move(0.9, 0.9)
        # Cursor position must NOT move while click lock is active
        self.assertEqual((locked_x, locked_y), (sx, sy))

        # After freeze expires, movement resumes normally
        import time
        time.sleep(0.20)
        resumed_x, resumed_y = self.ctrl.move(0.9, 0.9)
        self.assertNotEqual((resumed_x, resumed_y), (sx, sy))


class TestGestureRecognizer(unittest.TestCase):
    """Test deterministic gesture classification with synthetic hand landmarks."""

    def setUp(self):
        self.config = GestureConfig()
        self.recognizer = GestureRecognizer(config=self.config)

    def test_navigation_gesture(self):
        # Open hand with index finger upright (relaxed palm)
        hand = create_synthetic_hand(index_up=True, middle_up=True, ring_up=True, pinky_up=True)
        # Classify twice to satisfy confirmation frames
        self.recognizer.classify(hand)
        event = self.recognizer.classify(hand)
        self.assertEqual(event.gesture_type, GestureType.MOVE)

    def test_left_click_gesture(self):
        # Little finger (pinky) + thumb pinched together -> Left Click
        hand = create_synthetic_hand(
            index_up=True,
            middle_up=True,
            ring_up=True,
            pinky_up=True,
            pinky_tip_x=0.46,
            pinky_tip_y=0.45,
            thumb_tip_x=0.45,  # Pinched to pinky
            thumb_tip_y=0.45,
        )
        event = self.recognizer.classify(hand)
        self.assertEqual(event.gesture_type, GestureType.LEFT_CLICK)

    def test_right_click_gesture(self):
        # Ring finger and thumb pinched together -> Right Click
        hand = create_synthetic_hand(
            index_up=True,
            middle_up=True,
            ring_up=True,
            pinky_up=True,
            ring_tip_x=0.46,
            ring_tip_y=0.45,
            thumb_tip_x=0.45,  # Pinched to ring
            thumb_tip_y=0.45,
        )
        event = self.recognizer.classify(hand)
        self.assertEqual(event.gesture_type, GestureType.RIGHT_CLICK)

    def test_scroll_gesture(self):
        # Index and middle raised together with horizontal separation
        hand1 = create_synthetic_hand(
            index_up=True,
            middle_up=True,
            ring_up=False,
            pinky_up=False,
            index_tip_x=0.45,
            index_tip_y=0.35,
            middle_tip_x=0.58,  # Separated
            middle_tip_y=0.35,
        )
        self.recognizer.classify(hand1)
        self.recognizer.classify(hand1)

        # Hand moved down by dy
        hand2 = create_synthetic_hand(
            index_up=True,
            middle_up=True,
            ring_up=False,
            pinky_up=False,
            index_tip_x=0.45,
            index_tip_y=0.42,  # Displaced down
            middle_tip_x=0.58,
            middle_tip_y=0.42,
        )
        event = self.recognizer.classify(hand2)
        self.assertEqual(event.gesture_type, GestureType.SCROLL)
        # Downward movement should produce non-zero scroll_dy
        self.assertNotEqual(event.scroll_dy, 0.0)

    def test_drag_and_drop_gesture(self):
        # Closed fist held over time
        hand = create_synthetic_hand(
            index_up=False,
            middle_up=False,
            ring_up=False,
            pinky_up=False,
            thumb_up=False,
        )
        import time
        # First classification sets start time
        self.recognizer.classify(hand)
        # Sleep for drag_hold_sec
        time.sleep(self.config.drag_hold_sec + 0.05)
        event = self.recognizer.classify(hand)
        self.assertEqual(event.gesture_type, GestureType.DRAGGING)
        self.assertTrue(event.is_dragging)

    def test_all_fingers_except_index_drag_gesture(self):
        # Index finger pointing upright; all other fingers (thumb, middle, ring, pinky) pinched together -> Drag & Drop
        hand = create_synthetic_hand(
            index_up=True,
            middle_up=True,
            ring_up=True,
            pinky_up=True,
            thumb_up=True,
            index_tip_x=0.45,
            index_tip_y=0.30,
            middle_tip_x=0.46,
            middle_tip_y=0.48,
            ring_tip_x=0.46,
            ring_tip_y=0.48,
            pinky_tip_x=0.46,
            pinky_tip_y=0.48,
            thumb_tip_x=0.45,
            thumb_tip_y=0.48,
        )
        import time
        self.recognizer.classify(hand)
        time.sleep(self.config.drag_hold_sec + 0.05)
        event = self.recognizer.classify(hand)
        self.assertEqual(event.gesture_type, GestureType.DRAGGING)
        self.assertTrue(event.is_dragging)

    def test_pointing_curled_drag_gesture(self):
        # Index upright, middle/ring/pinky curled into palm -> Drag & Drop
        hand = create_synthetic_hand(
            index_up=True,
            middle_up=False,
            ring_up=False,
            pinky_up=False,
            thumb_up=False,
            index_tip_x=0.45,
            index_tip_y=0.30,
        )
        import time
        self.recognizer.classify(hand)
        time.sleep(self.config.drag_hold_sec + 0.05)
        event = self.recognizer.classify(hand)
        self.assertEqual(event.gesture_type, GestureType.DRAGGING)
        self.assertTrue(event.is_dragging)

    def test_drag_release_hysteresis(self):
        # 1. Start drag
        hand_drag = create_synthetic_hand(
            index_up=True,
            middle_up=False,
            ring_up=False,
            pinky_up=False,
            thumb_up=False,
        )
        import time
        self.recognizer.classify(hand_drag)
        time.sleep(self.config.drag_hold_sec + 0.05)
        event = self.recognizer.classify(hand_drag)
        self.assertEqual(event.gesture_type, GestureType.DRAGGING)
        self.assertTrue(event.is_dragging)

        # 2. Transient 1-frame tracking noise (e.g. middle slightly wavers)
        hand_noisy = create_synthetic_hand(
            index_up=True,
            middle_up=True,
            ring_up=False,
            pinky_up=False,
        )
        event_noisy = self.recognizer.classify(hand_noisy)
        # Hysteresis keeps drag active!
        self.assertTrue(event_noisy.is_dragging)

        # 3. Deliberately open hand (all fingers extended) -> Drops immediately
        hand_open = create_synthetic_hand(
            index_up=True,
            middle_up=True,
            ring_up=True,
            pinky_up=True,
        )
        event_drop = self.recognizer.classify(hand_open)
        self.assertFalse(event_drop.is_dragging)

    def test_click_debounce_and_latch(self):
        # 1. Pinch pinky to thumb -> triggers LEFT_CLICK
        pinch_hand = create_synthetic_hand(
            index_up=True,
            middle_up=True,
            ring_up=True,
            pinky_up=True,
            pinky_tip_x=0.46,
            pinky_tip_y=0.45,
            thumb_tip_x=0.45,
            thumb_tip_y=0.45,
        )
        event1 = self.recognizer.classify(pinch_hand)
        self.assertEqual(event1.gesture_type, GestureType.LEFT_CLICK)

        # 2. Subsequent frame still pinched -> latched, no second click!
        event2 = self.recognizer.classify(pinch_hand)
        self.assertNotEqual(event2.gesture_type, GestureType.LEFT_CLICK)
        self.assertTrue(event2.lock_cursor)

        # 3. Release pinch (open hand)
        open_hand = create_synthetic_hand(
            index_up=True,
            pinky_tip_x=0.65,
            thumb_tip_x=0.35,
        )
        self.recognizer.classify(open_hand)

        # 4. Immediate re-pinch before debounce interval elapses -> blocked
        event3 = self.recognizer.classify(pinch_hand)
        self.assertNotEqual(event3.gesture_type, GestureType.LEFT_CLICK)

        # 5. After debounce interval elapses -> triggers next click
        import time
        time.sleep(self.config.click_debounce_sec + 0.05)
        event4 = self.recognizer.classify(pinch_hand)
        self.assertEqual(event4.gesture_type, GestureType.LEFT_CLICK)


class TestCameraAndUIOverlay(unittest.TestCase):
    """Test synthetic camera stream and HUD rendering."""

    def test_synthetic_camera_stream(self):
        cam = CameraManager(use_mock=True, frame_width=640, frame_height=480)
        self.assertTrue(cam.start())
        ret, frame = cam.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        cam.release()

    def test_camera_zoom_controls(self):
        cam = CameraManager(use_mock=True, zoom=1.0)
        self.assertEqual(cam.zoom, 1.0)

        # Test zoom in and clamping
        cam.zoom_in(0.5)
        self.assertEqual(cam.zoom, 1.5)
        cam.set_zoom(10.0)  # Should clamp to 4.0
        self.assertEqual(cam.zoom, 4.0)

        # Test zoom out and clamping
        cam.zoom_out(0.5)
        self.assertEqual(cam.zoom, 3.5)
        cam.set_zoom(0.2)  # Should clamp to 1.0
        self.assertEqual(cam.zoom, 1.0)

        # Test cycle presets
        z1 = cam.cycle_zoom()
        self.assertEqual(z1, 1.25)
        z2 = cam.cycle_zoom()
        self.assertEqual(z2, 1.5)

    def test_camera_zoom_frame_dimensions(self):
        cam = CameraManager(use_mock=True, frame_width=640, frame_height=480, zoom=2.0)
        self.assertTrue(cam.start())
        ret, frame = cam.read()
        self.assertTrue(ret)
        self.assertEqual(frame.shape, (480, 640, 3))
        cam.release()

    def test_ui_overlay_rendering(self):
        ui = UIOverlay()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand = create_synthetic_hand()
        event = GestureRecognizer().classify(hand)

        annotated = ui.render(
            frame=frame,
            fps=31.5,
            hand_result=hand,
            gesture_event=event,
            screen_pos=(800, 450),
            filter_name="OneEuro",
            zoom=2.0,
        )
        self.assertEqual(annotated.shape, (480, 640, 3))
        # Verify pixels were modified (HUD banner, corner lines, zoom badge, etc.)
        self.assertTrue(np.any(annotated > 0))


if __name__ == "__main__":
    unittest.main()
