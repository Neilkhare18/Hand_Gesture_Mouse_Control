"""Camera capture manager supporting physical webcams and headless synthetic frame testing."""

import time
from typing import Optional, Tuple
import cv2
import numpy as np


class CameraManager:
    """Manages OpenCV webcam video stream with mirroring, FPS tracking, and robust error recovery."""

    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 1920,
        frame_height: int = 1080,
        target_fps: int = 60,
        flip_horizontal: bool = True,
        use_mock: bool = False,
        zoom: float = 1.0,
    ):
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_fps = target_fps
        self.flip_horizontal = flip_horizontal
        self.use_mock = use_mock
        self._zoom = round(max(1.0, min(4.0, float(zoom))), 2)
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        
        # FPS estimation
        self._prev_time = 0.0
        self._fps_smooth = float(target_fps)
        self._frame_count = 0

    @property
    def zoom(self) -> float:
        """Current digital zoom factor (1.0x to 4.0x)."""
        return self._zoom

    @zoom.setter
    def zoom(self, value: float) -> None:
        self.set_zoom(value)

    def set_zoom(self, value: float) -> float:
        """Set digital zoom factor clamped between 1.0x and 4.0x."""
        self._zoom = round(max(1.0, min(4.0, float(value))), 2)
        return self._zoom

    def zoom_in(self, step: float = 0.1) -> float:
        """Increase zoom by step (clamped to 4.0x max)."""
        return self.set_zoom(self._zoom + step)

    def zoom_out(self, step: float = 0.1) -> float:
        """Decrease zoom by step (clamped to 1.0x min)."""
        return self.set_zoom(self._zoom - step)

    def cycle_zoom(self) -> float:
        """Cycle through common zoom presets: 1.0x -> 1.25x -> 1.5x -> 2.0x -> 3.0x -> 1.0x."""
        presets = [1.0, 1.25, 1.5, 2.0, 3.0]
        for p in presets:
            if p > self._zoom + 0.05:
                return self.set_zoom(p)
        return self.set_zoom(1.0)

    def _apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        """Crop center region and resize to original resolution for digital zoom."""
        if self._zoom <= 1.001:
            return frame
        h, w = frame.shape[:2]
        crop_w = max(10, int(w / self._zoom))
        crop_h = max(10, int(h / self._zoom))
        x1 = max(0, (w - crop_w) // 2)
        y1 = max(0, (h - crop_h) // 2)
        cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def start(self) -> bool:
        """Initialize and open the video capture stream."""
        if self.use_mock:
            self.is_opened = True
            return True

        # Try DirectShow on Windows first for fast startup
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            # If camera 0 failed, try index 1
            if self.camera_index == 0:
                self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(1)

        if self.cap and self.cap.isOpened():
            # Enable MJPEG fourcc codec for 60 FPS USB bandwidth throughput
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w > 0 and actual_h > 0:
                self.frame_width = actual_w
                self.frame_height = actual_h
            self.is_opened = True
            self._prev_time = time.perf_counter()
            return True

        self.is_opened = False
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the capture stream.
        
        Returns:
            Tuple of (success_flag, bgr_frame_array)
        """
        if self.use_mock:
            # Generate a clean dark synthetic frame for testing
            frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            zoom_tag = f" [ZOOM: {self._zoom:.1f}x]" if self._zoom > 1.0 else ""
            cv2.putText(
                frame, f"SYNTHETIC TEST STREAM{zoom_tag}", (50, self.frame_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
            frame = self._apply_zoom(frame)
            self._update_fps()
            return True, frame

        if not self.cap or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        frame = self._apply_zoom(frame)
        self._update_fps()
        return True, frame

    def _update_fps(self) -> None:
        """Compute exponentially smoothed FPS."""
        now = time.perf_counter()
        if self._prev_time > 0:
            dt = now - self._prev_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self._fps_smooth = 0.9 * self._fps_smooth + 0.1 * instant_fps
        self._prev_time = now
        self._frame_count += 1

    @property
    def fps(self) -> float:
        """Get the current running frame rate."""
        return max(1.0, self._fps_smooth)

    def release(self) -> None:
        """Safely release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
