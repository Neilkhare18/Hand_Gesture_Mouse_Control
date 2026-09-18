"""MediaPipe 21-landmark hand detection and feature extraction pipeline.

Supports modern MediaPipe Tasks HandLandmarker (MediaPipe 1.0+) with automatic model
caching, monotonic frame timestamping, and robust error recovery.
"""

import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Landmark constants according to MediaPipe Hand Landmark Topology
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (MIDDLE_MCP, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (RING_MCP, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (WRIST, PINKY_MCP),
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


@dataclass
class HandLandmarksResult:
    """Encapsulates extracted hand landmarks and derived geometric metrics."""
    landmarks_norm: List[Tuple[float, float, float]]
    landmarks_pixel: List[Tuple[int, int]]
    handedness: str
    hand_scale: float
    raw_landmarks: Optional[object] = None

    def distance_norm(self, idx1: int, idx2: int) -> float:
        """Compute scale-invariant Euclidean distance between two landmarks."""
        p1 = self.landmarks_norm[idx1]
        p2 = self.landmarks_norm[idx2]
        dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        return dist / (self.hand_scale + 1e-6)

    def distance_pixels(self, idx1: int, idx2: int) -> float:
        """Compute pixel distance between two landmarks."""
        p1 = self.landmarks_pixel[idx1]
        p2 = self.landmarks_pixel[idx2]
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def is_finger_extended(self, tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
        """Orientation-invariant check for finger extension using Euclidean distances and screen alignment."""
        p_tip = self.landmarks_norm[tip_idx]
        p_pip = self.landmarks_norm[pip_idx]
        p_mcp = self.landmarks_norm[mcp_idx]
        p_wrist = self.landmarks_norm[WRIST]

        d_tip_wrist = math.hypot(p_tip[0] - p_wrist[0], p_tip[1] - p_wrist[1])
        d_pip_wrist = math.hypot(p_pip[0] - p_wrist[0], p_pip[1] - p_wrist[1])
        d_tip_mcp = math.hypot(p_tip[0] - p_mcp[0], p_tip[1] - p_mcp[1])
        d_pip_mcp = math.hypot(p_pip[0] - p_mcp[0], p_pip[1] - p_mcp[1])

        # Extended if tip is higher than pip joint or clearly further from wrist & MCP
        vert_extended = p_tip[1] < (p_pip[1] + 0.02)
        dist_extended = (d_tip_wrist > d_pip_wrist * 1.02) and (d_tip_mcp > d_pip_mcp * 0.85)

        return vert_extended or dist_extended

    def is_finger_curled(self, tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
        """Orientation-invariant check for finger curled/folded towards palm."""
        return not self.is_finger_extended(tip_idx, pip_idx, mcp_idx)

    def get_finger_extended_states(self) -> Dict[str, bool]:
        """Determine whether each finger is extended (upright) or curled."""
        index_up = self.is_finger_extended(INDEX_TIP, INDEX_PIP, INDEX_MCP)
        middle_up = self.is_finger_extended(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        ring_up = self.is_finger_extended(RING_TIP, RING_PIP, RING_MCP)
        pinky_up = self.is_finger_extended(PINKY_TIP, PINKY_PIP, PINKY_MCP)

        # Thumb extension: check tip distance from pinky base (MCP 17) relative to IP joint
        dist_thumb_tip_pinky = math.hypot(
            self.landmarks_norm[THUMB_TIP][0] - self.landmarks_norm[PINKY_MCP][0],
            self.landmarks_norm[THUMB_TIP][1] - self.landmarks_norm[PINKY_MCP][1],
        )
        dist_thumb_ip_pinky = math.hypot(
            self.landmarks_norm[THUMB_IP][0] - self.landmarks_norm[PINKY_MCP][0],
            self.landmarks_norm[THUMB_IP][1] - self.landmarks_norm[PINKY_MCP][1],
        )
        thumb_up = dist_thumb_tip_pinky > (dist_thumb_ip_pinky * 1.05)

        return {
            "thumb": thumb_up,
            "index": index_up,
            "middle": middle_up,
            "ring": ring_up,
            "pinky": pinky_up,
        }


class HandTracker:
    """MediaPipe Hands wrapper managing detection and landmark processing."""

    def __init__(
        self,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
        max_num_hands: int = 1,
        model_path: Optional[str] = None,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_hands = max_num_hands
        self.model_path = model_path
        
        self.mode = "none"
        self.detector = None
        self.mp_module = None
        self._start_time = time.perf_counter()
        self._last_timestamp_ms = 0
        self._has_logged_first_detection = False
        
        self._init_mediapipe()

    def _ensure_model_file(self) -> str:
        """Download or locate hand_landmarker.task model file."""
        if self.model_path and os.path.exists(self.model_path):
            return self.model_path

        # Search portable locations for standalone builds and source runs
        candidates = []
        if getattr(sys, "frozen", False):
            if hasattr(sys, "_MEIPASS"):
                meipass = Path(sys._MEIPASS)
                candidates.append(meipass / "models" / "hand_landmarker.task")
                candidates.append(meipass / "_internal" / "models" / "hand_landmarker.task")
            exe_dir = Path(sys.executable).resolve().parent
            candidates.append(exe_dir / "models" / "hand_landmarker.task")
            candidates.append(exe_dir / "_internal" / "models" / "hand_landmarker.task")

        candidates.append(Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task")
        candidates.append(Path.cwd() / "models" / "hand_landmarker.task")

        for cand in candidates:
            if cand.exists():
                return str(cand)

        default_dir = Path(__file__).resolve().parent.parent / "models"
        default_dir.mkdir(parents=True, exist_ok=True)
        target_path = default_dir / "hand_landmarker.task"

        if not target_path.exists():
            print(f"[HandTracker] Downloading hand_landmarker model from {MODEL_URL}...")
            urllib.request.urlretrieve(MODEL_URL, str(target_path))
            print(f"[HandTracker] Model downloaded to {target_path} ({target_path.stat().st_size} bytes)")

        return str(target_path)

    def _init_mediapipe(self) -> None:
        """Initialize modern Tasks API or legacy solutions API."""
        try:
            import mediapipe as mp
            self.mp_module = mp

            # Try modern MediaPipe 1.0+ Tasks API first
            if hasattr(mp, "tasks"):
                try:
                    from mediapipe.tasks.python import vision
                    from mediapipe.tasks.python.core.base_options import BaseOptions
                except ImportError:
                    vision = mp.tasks.vision
                    BaseOptions = mp.tasks.BaseOptions

                task_file = self._ensure_model_file()
                options = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=task_file),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=self.max_num_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_hand_presence_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                self.mode = "tasks"
                print(f"[HandTracker] MediaPipe HandLandmarker active (Tasks API, model: {os.path.basename(task_file)})")
                return

            # Fallback to legacy solutions API
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.detector = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self.mode = "solutions"
                print("[HandTracker] MediaPipe Hands active (Solutions API)")
                return

            print("[HandTracker Warning] No compatible MediaPipe API found.")
            self.mode = "none"

        except Exception as e:
            print(f"[HandTracker Error] Failed to initialize MediaPipe: {e}")
            self.mode = "none"

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[HandLandmarksResult]:
        """Detect hand in BGR frame and return parsed HandLandmarksResult."""
        if self.detector is None or frame_bgr is None:
            return None

        h, w, _ = frame_bgr.shape

        try:
            if self.mode == "tasks":
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = self.mp_module.Image(
                    image_format=self.mp_module.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                # Ensure strictly monotonic timestamps for MediaPipe VIDEO running mode
                now_ms = int((time.perf_counter() - self._start_time) * 1000)
                if now_ms <= self._last_timestamp_ms:
                    now_ms = self._last_timestamp_ms + 1
                self._last_timestamp_ms = now_ms

                result = self.detector.detect_for_video(mp_image, now_ms)

                if not result.hand_landmarks or len(result.hand_landmarks) == 0:
                    return None

                hand_lms = result.hand_landmarks[0]
                handedness_label = "Right"
                if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
                    handedness_label = result.handedness[0][0].category_name

                landmarks_norm = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                landmarks_pixel = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

            elif self.mode == "solutions":
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                result = self.detector.process(frame_rgb)
                frame_rgb.flags.writeable = True

                if not result.multi_hand_landmarks:
                    return None

                hand_lms = result.multi_hand_landmarks[0]
                handedness_label = "Right"
                if result.multi_handedness:
                    handedness_label = results.multi_handedness[0].classification[0].label

                landmarks_norm = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
                landmarks_pixel = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
            else:
                return None

            # Compute reference hand scale: 2D distance between Wrist (0) and Middle MCP (9)
            wrist_pt = landmarks_norm[WRIST]
            middle_mcp_pt = landmarks_norm[MIDDLE_MCP]
            hand_scale = math.hypot(wrist_pt[0] - middle_mcp_pt[0], wrist_pt[1] - middle_mcp_pt[1])
            if hand_scale < 0.01:
                hand_scale = 0.2

            if not self._has_logged_first_detection:
                print(f"[HandTracker] Hand detected! Tracking {handedness_label} hand in frame.")
                self._has_logged_first_detection = True

            return HandLandmarksResult(
                landmarks_norm=landmarks_norm,
                landmarks_pixel=landmarks_pixel,
                handedness=handedness_label,
                hand_scale=hand_scale,
                raw_landmarks=hand_lms,
            )

        except Exception as e:
            # Prevent camera loop crash if single frame inference has an issue
            return None

    def draw_landmarks(self, frame_bgr: np.ndarray, result: HandLandmarksResult) -> None:
        """Render 21 landmarks and skeleton topology."""
        pts = result.landmarks_pixel
        # Draw skeleton connections
        for i1, i2 in HAND_CONNECTIONS:
            if i1 < len(pts) and i2 < len(pts):
                cv2.line(frame_bgr, pts[i1], pts[i2], (0, 215, 255), 2, cv2.LINE_AA)

        # Draw joints
        for i, (u, v) in enumerate(pts):
            if i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP):
                radius = 6
                color = (0, 255, 128) if i == INDEX_TIP else (255, 0, 255)
            else:
                radius = 3
                color = (0, 200, 255)
            cv2.circle(frame_bgr, (u, v), radius, color, -1, cv2.LINE_AA)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.detector is not None:
            self.detector.close()
            self.detector = None
