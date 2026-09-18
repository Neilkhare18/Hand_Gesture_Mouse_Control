"""Main application entry point for Hand Gesture Mouse Control."""

import argparse
import sys
import time
import cv2

from src.config import AppConfig
from src.camera import CameraManager
from src.hand_tracker import HandTracker
from src.gesture_recognizer import GestureRecognizer, GestureType
from src.mouse_controller import MouseController, MockMouseDriver
from src.ui_overlay import UIOverlay


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hand Gesture Mouse Control - Touchless Computer Vision HCI Application"
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height (default: 480)")
    parser.add_argument("--fps", type=int, default=60, help="Camera target FPS (default: 60)")
    parser.add_argument("--zoom", type=float, default=1.0, help="Camera digital zoom factor (1.0 to 4.0, default: 1.0)")
    parser.add_argument(
        "--filter", choices=["one_euro", "ema", "none"], default="one_euro",
        help="Motion smoothing filter (default: one_euro)"
    )
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Detection confidence (default: 0.55)")
    parser.add_argument("--no-hud", action="store_true", help="Disable HUD AR overlay")
    parser.add_argument("--mock-camera", action="store_true", help="Use synthetic camera frames for testing")
    parser.add_argument("--mock-mouse", action="store_true", help="Use mock mouse driver (no OS cursor hijacking)")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (0 = run continuously)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize configuration
    config = AppConfig()
    config.camera.camera_index = args.camera
    config.camera.frame_width = args.width
    config.camera.frame_height = args.height
    config.camera.target_fps = args.fps
    config.camera.zoom = max(1.0, min(4.0, args.zoom))
    config.mediapipe.min_detection_confidence = args.min_confidence
    config.mediapipe.min_tracking_confidence = args.min_confidence
    config.smoothing.filter_type = args.filter
    if args.no_hud:
        config.ui.show_hud = False

    print("=" * 60)
    print("       TOUCHLESS HAND GESTURE MOUSE CONTROL SYSTEM")
    print("=" * 60)
    print(f" Camera Index     : {config.camera.camera_index}")
    print(f" Resolution       : {config.camera.frame_width}x{config.camera.frame_height} @ {config.camera.target_fps} FPS")
    print(f" Camera Zoom      : {config.camera.zoom:.1f}x")
    print(f" Smoothing Filter : {config.smoothing.filter_type.upper()}")
    print(f" Min Confidence   : {config.mediapipe.min_detection_confidence}")
    print(f" Synthetic Camera : {args.mock_camera}")
    print(f" Mock Mouse       : {args.mock_mouse}")
    print("-" * 60)
    print(" Gestures Supported:")
    print("  * Open Hand / Index Upright        -> Cursor Navigation")
    print("  * Little Finger + Thumb Pinch      -> Left Click")
    print("  * Ring Finger + Thumb Pinch        -> Right Click")
    print("  * All Fingers Except Index Pinch   -> Drag and Drop")
    print("  * Index + Middle Fingers Upright   -> Scroll Up / Down")
    print("-" * 60)
    print(" Hotkeys:")
    print("  * [Q] or [ESC] : Quit Application")
    print("  * [H]          : Toggle HUD Overlay")
    print("  * [S]          : Toggle Smoothing Filter (OneEuro <-> EMA)")
    print("  * [Z]          : Cycle Zoom Presets (1.0x -> 1.5x -> 2.0x -> 3.0x -> 1.0x)")
    print("  * [+] / [-]    : Zoom In / Out (+/- 0.1x)")
    print("  * [0]          : Reset Zoom to 1.0x")
    print("  * [R]          : Reset Tracking & Filters")
    print("=" * 60)

    # Initialize modules
    camera = CameraManager(
        camera_index=config.camera.camera_index,
        frame_width=config.camera.frame_width,
        frame_height=config.camera.frame_height,
        target_fps=config.camera.target_fps,
        flip_horizontal=config.camera.flip_horizontal,
        use_mock=args.mock_camera,
        zoom=config.camera.zoom,
    )

    if not camera.start():
        print(f"[Error] Failed to open camera device {config.camera.camera_index}.")
        print("Please check your webcam connection or launch with --mock-camera.")
        sys.exit(1)

    tracker = HandTracker(
        min_detection_confidence=config.mediapipe.min_detection_confidence,
        min_tracking_confidence=config.mediapipe.min_tracking_confidence,
        max_num_hands=config.mediapipe.max_num_hands,
    )

    recognizer = GestureRecognizer(config=config.gesture)

    driver = MockMouseDriver() if args.mock_mouse else None
    controller = MouseController(
        mapping_config=config.mapping,
        smoothing_config=config.smoothing,
        driver=driver,
        cam_width=camera.frame_width,
        cam_height=camera.frame_height,
    )

    ui = UIOverlay(ui_config=config.ui, mapping_config=config.mapping)

    window_name = "Hand Gesture Mouse Control - AI Vision HCI"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    screen_pos = None

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1

            # 1. Landmark detection
            hand_result = tracker.process_frame(frame)
            gesture_event = None

            if hand_result is not None:
                # Optional: draw MediaPipe skeleton
                if config.ui.show_landmarks:
                    tracker.draw_landmarks(frame, hand_result)

                # 2. Gesture classification
                gesture_event = recognizer.classify(hand_result)
                cur_norm_x, cur_norm_y = gesture_event.cursor_norm

                # 3. Action Dispatcher
                if gesture_event.lock_cursor:
                    controller.freeze(config.smoothing.click_freeze_sec)

                if gesture_event.gesture_type == GestureType.MOVE:
                    screen_pos = controller.move(cur_norm_x, cur_norm_y)
                    controller.set_drag_state(False)

                elif gesture_event.gesture_type == GestureType.LEFT_CLICK:
                    controller.left_click(freeze_sec=config.smoothing.click_freeze_sec)
                    screen_pos = controller.get_current_pos()

                elif gesture_event.gesture_type == GestureType.RIGHT_CLICK:
                    controller.right_click(freeze_sec=config.smoothing.click_freeze_sec)
                    screen_pos = controller.get_current_pos()

                elif gesture_event.gesture_type == GestureType.SCROLL:
                    controller.scroll(gesture_event.scroll_dy)

                elif gesture_event.gesture_type == GestureType.DRAGGING:
                    screen_pos = controller.move(cur_norm_x, cur_norm_y)
                    controller.set_drag_state(True)

                elif gesture_event.gesture_type == GestureType.IDLE:
                    controller.set_drag_state(False)
            else:
                # Reset state when no hand is present
                recognizer.reset()
                controller.reset()
                screen_pos = None

            # 4. Render AR HUD Overlay
            frame = ui.render(
                frame=frame,
                fps=camera.fps,
                hand_result=hand_result,
                gesture_event=gesture_event,
                screen_pos=screen_pos,
                filter_name=config.smoothing.filter_type.capitalize(),
                zoom=camera.zoom,
            )

            # 5. Display Frame
            cv2.imshow(window_name, frame)

            # 6. Hotkeys
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                print("[Info] Exiting gesture control loop...")
                break
            elif key == ord("h") or key == ord("H"):
                config.ui.show_hud = not config.ui.show_hud
            elif key == ord("s") or key == ord("S"):
                # Cycle smoothing filters
                if config.smoothing.filter_type == "one_euro":
                    config.smoothing.filter_type = "ema"
                elif config.smoothing.filter_type == "ema":
                    config.smoothing.filter_type = "none"
                else:
                    config.smoothing.filter_type = "one_euro"
                controller.smooth_cfg.filter_type = config.smoothing.filter_type
            elif key == ord("z") or key == ord("Z"):
                new_zoom = camera.cycle_zoom()
                print(f"[Info] Zoom level set to {new_zoom:.1f}x")
            elif key == ord("+") or key == ord("="):
                new_zoom = camera.zoom_in(0.1)
                print(f"[Info] Zoom level set to {new_zoom:.1f}x")
            elif key == ord("-") or key == ord("_"):
                new_zoom = camera.zoom_out(0.1)
                print(f"[Info] Zoom level set to {new_zoom:.1f}x")
            elif key == ord("0"):
                new_zoom = camera.set_zoom(1.0)
                print(f"[Info] Zoom reset to 1.0x")
            elif key == ord("r") or key == ord("R"):
                controller.reset()
                recognizer.reset()

            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"[Info] Reached max frames limit ({args.max_frames}).")
                break

    except KeyboardInterrupt:
        print("[Info] Interrupted by user.")
    finally:
        camera.release()
        tracker.close()
        controller.reset()
        cv2.destroyAllWindows()
        print("[Info] Shutdown complete. Resources released.")


if __name__ == "__main__":
    main()
