# Contributing to Hand Gesture Mouse Control

Thank you for your interest in improving **Hand Gesture Mouse Control**! Contributions from the open-source community are very welcome.

---

## Code of Conduct

Please be respectful, collaborative, and considerate of others when interacting within this project's issues, discussions, and pull requests.

---

## How to Get Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/HandGestureMouseControl.git
   cd HandGestureMouseControl
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Windows CMD:
   .venv\Scripts\activate.bat
   ```
4. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

---

## Project Architecture

* **`main.py`**: Main application orchestration loop and CLI argument parsing.
* **`src/camera.py`**: Real-time webcam frame acquisition, DirectShow backend, MJPG codec, digital zoom.
* **`src/hand_tracker.py`**: MediaPipe Tasks HandLandmarker 21 3D landmarks extraction and topology rendering.
* **`src/gesture_recognizer.py`**: Deterministic gesture state machine, scale-invariant distances, hysteresis, and debounce.
* **`src/mouse_controller.py`**: Low-latency Windows Win32 API (`user32.dll`) mouse dispatch, coordinate mapping, and click-locking.
* **`src/smoothing.py`**: One-Euro adaptive filter, Exponential Moving Average, and deadband tremor suppressor.
* **`src/ui_overlay.py`**: Augmented Reality HUD banner, active boundaries, and fingertip feedback.
* **`tests/test_pipeline.py`**: Comprehensive automated unit test suite.

---

## Development Guidelines

1. **Keep Latency Low**: Any changes to landmark processing or mouse dispatch must not introduce frame delay. Keep per-frame compute under 5 ms.
2. **Deterministic Gestures**: Gesture thresholds should always be normalized by hand scale (`wrist` to `middle_mcp` distance) to remain invariant to camera distance.
3. **Run Unit Tests**: Ensure all automated unit tests pass before submitting a PR:
   ```bash
   python -m unittest tests/test_pipeline.py
   ```
4. **Benchmark Latency**:
   ```bash
   python tests/benchmark_latency.py
   ```

---

## Submitting Pull Requests

1. Create a feature branch (`git checkout -b feature/awesome-gesture`).
2. Commit your changes with clear messages (`git commit -m 'feat: add two-finger swipe gesture'`).
3. Push to your branch (`git push origin feature/awesome-gesture`).
4. Open a Pull Request against the `main` branch with a description of your changes and test results.
