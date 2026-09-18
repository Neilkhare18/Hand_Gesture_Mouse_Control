import os
import sys
import time
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig
from src.hand_tracker import HandTracker
from src.gesture_recognizer import GestureRecognizer
from src.mouse_controller import MouseController, MockMouseDriver
from src.ui_overlay import UIOverlay


def run_benchmark(num_frames: int = 100):
    print("=" * 60)
    print(f" RUNNING LATENCY & FPS BENCHMARK ({num_frames} frames)")
    print("=" * 60)

    config = AppConfig()
    tracker = HandTracker(
        min_detection_confidence=config.mediapipe.min_detection_confidence,
        min_tracking_confidence=config.mediapipe.min_tracking_confidence,
    )
    recognizer = GestureRecognizer(config=config.gesture)
    mock_driver = MockMouseDriver()
    controller = MouseController(
        mapping_config=config.mapping,
        smoothing_config=config.smoothing,
        driver=mock_driver,
    )
    ui = UIOverlay(ui_config=config.ui, mapping_config=config.mapping)

    # Synthetic realistic frame (640x480)
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)

    latencies = []

    # Warm-up (10 frames)
    for _ in range(10):
        _ = tracker.process_frame(frame)

    print("Benchmarking active pipeline...")
    for i in range(num_frames):
        t_start = time.perf_counter()

        # 1. Tracker
        hand_result = tracker.process_frame(frame)

        # 2. Recognizer
        if hand_result:
            event = recognizer.classify(hand_result)
            # 3. Mouse controller
            controller.move(event.cursor_norm[0], event.cursor_norm[1])
        else:
            event = None

        # 4. UI Overlay
        _ = ui.render(
            frame=frame.copy(),
            fps=30.0,
            hand_result=hand_result,
            gesture_event=event,
            screen_pos=(960, 540),
            filter_name="OneEuro",
        )

        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000.0)

    tracker.close()

    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    max_fps = 1000.0 / mean_lat if mean_lat > 0 else 0

    print("-" * 60)
    print(f" Mean Pipeline Latency  : {mean_lat:6.2f} ms")
    print(f" Median Latency         : {median_lat:6.2f} ms")
    print(f" 95th Percentile Latency: {p95_lat:6.2f} ms")
    print(f" Max Theoretical FPS    : {max_fps:6.1f} FPS")
    print("-" * 60)

    if mean_lat < 30.0:
        print(" [PASS] Interaction latency is strictly below 30 ms target!")
    else:
        print(" [WARN] Interaction latency is above 30 ms target.")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark(100)
