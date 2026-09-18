"""Ultra-low-latency desktop mouse controller using native Win32 API and PyAutoGUI fallback."""

import sys
import time
from typing import Optional, Tuple
from .config import MappingConfig, SmoothingConfig
from .smoothing import OneEuroFilter2D, AdaptiveEMA2D, DeadbandFilter


class BaseMouseDriver:
    """Base interface for mouse hardware automation."""
    def get_screen_size(self) -> Tuple[int, int]:
        return (1920, 1080)

    def move(self, x: int, y: int) -> None:
        pass

    def left_click(self) -> None:
        pass

    def right_click(self) -> None:
        pass

    def mouse_down(self) -> None:
        pass

    def mouse_up(self) -> None:
        pass

    def scroll(self, clicks: int) -> None:
        pass


class Win32MouseDriver(BaseMouseDriver):
    """Direct Windows User32 API driver with sub-millisecond dispatch time."""

    # Win32 Mouse Event Flags
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_WHEEL = 0x0800
    WHEEL_DELTA = 120

    def __init__(self):
        import ctypes
        self.user32 = ctypes.windll.user32
        # Enable Per-Monitor DPI awareness to ensure accurate pixel coordinates
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                self.user32.SetProcessDPIAware()
            except Exception:
                pass

    def get_screen_size(self) -> Tuple[int, int]:
        w = self.user32.GetSystemMetrics(0)  # SM_CXSCREEN
        h = self.user32.GetSystemMetrics(1)  # SM_CYSCREEN
        return (w, h)

    def move(self, x: int, y: int) -> None:
        self.user32.SetCursorPos(int(x), int(y))

    def left_click(self) -> None:
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.01)
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def right_click(self) -> None:
        self.user32.mouse_event(self.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        time.sleep(0.01)
        self.user32.mouse_event(self.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

    def mouse_down(self) -> None:
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    def mouse_up(self) -> None:
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def scroll(self, clicks: int) -> None:
        delta = int(clicks * self.WHEEL_DELTA)
        self.user32.mouse_event(self.MOUSEEVENTF_WHEEL, 0, 0, delta, 0)


class PyAutoGuiDriver(BaseMouseDriver):
    """Fallback cross-platform driver using PyAutoGUI."""

    def __init__(self):
        import pyautogui
        self.pyautogui = pyautogui
        self.pyautogui.PAUSE = 0.001
        self.pyautogui.FAILSAFE = False

    def get_screen_size(self) -> Tuple[int, int]:
        sz = self.pyautogui.size()
        return (sz.width, sz.height)

    def move(self, x: int, y: int) -> None:
        self.pyautogui.moveTo(x, y)

    def left_click(self) -> None:
        self.pyautogui.click()

    def right_click(self) -> None:
        self.pyautogui.rightClick()

    def mouse_down(self) -> None:
        self.pyautogui.mouseDown()

    def mouse_up(self) -> None:
        self.pyautogui.mouseUp()

    def scroll(self, clicks: int) -> None:
        self.pyautogui.scroll(int(clicks * 100))


class MockMouseDriver(BaseMouseDriver):
    """Mock mouse driver for headless automated test runs."""

    def __init__(self, width: int = 1920, height: int = 1080):
        self.w = width
        self.h = height
        self.cur_x = width // 2
        self.cur_y = height // 2
        self.is_down = False
        self.click_count = 0
        self.right_click_count = 0
        self.scroll_total = 0

    def get_screen_size(self) -> Tuple[int, int]:
        return (self.w, self.h)

    def move(self, x: int, y: int) -> None:
        self.cur_x = x
        self.cur_y = y

    def left_click(self) -> None:
        self.click_count += 1

    def right_click(self) -> None:
        self.right_click_count += 1

    def mouse_down(self) -> None:
        self.is_down = True

    def mouse_up(self) -> None:
        self.is_down = False

    def scroll(self, clicks: int) -> None:
        self.scroll_total += clicks


class MouseController:
    """Coordinates frame mapping, dynamic smoothing, and OS action dispatch."""

    def __init__(
        self,
        mapping_config: Optional[MappingConfig] = None,
        smoothing_config: Optional[SmoothingConfig] = None,
        driver: Optional[BaseMouseDriver] = None,
        cam_width: int = 640,
        cam_height: int = 480,
    ):
        self.map_cfg = mapping_config or MappingConfig()
        self.smooth_cfg = smoothing_config or SmoothingConfig()
        self.cam_w = cam_width
        self.cam_h = cam_height

        # Select fastest available driver
        if driver is not None:
            self.driver = driver
        elif sys.platform == "win32":
            try:
                self.driver = Win32MouseDriver()
            except Exception:
                self.driver = PyAutoGuiDriver()
        else:
            self.driver = PyAutoGuiDriver()

        self.screen_w, self.screen_h = self.driver.get_screen_size()

        # Initialize motion filters
        self.one_euro = OneEuroFilter2D(
            min_cutoff=self.smooth_cfg.min_cutoff,
            beta=self.smooth_cfg.beta,
            d_cutoff=self.smooth_cfg.d_cutoff,
        )
        self.adaptive_ema = AdaptiveEMA2D(base_alpha=self.smooth_cfg.ema_alpha)
        self.deadband = DeadbandFilter(threshold=self.smooth_cfg.deadband_threshold)

        self._is_dragging = False
        self._scroll_accumulator = 0.0
        self._last_scroll_time = 0.0

        # Click position lock / freeze state
        self._freeze_until_time = 0.0
        self._locked_pos: Optional[Tuple[int, int]] = None
        self._current_pos: Tuple[int, int] = (self.screen_w // 2, self.screen_h // 2)

        # Stage 1 landmark pre-filtering
        self._prev_norm_x: Optional[float] = None
        self._prev_norm_y: Optional[float] = None

    def freeze(self, duration_sec: float = 0.35) -> None:
        """Lock the cursor at its current position to prevent movement during clicks."""
        now = time.perf_counter()
        self._freeze_until_time = max(self._freeze_until_time, now + duration_sec)
        self._locked_pos = self._current_pos

    def is_locked(self) -> bool:
        """Check if cursor position is currently frozen."""
        return time.perf_counter() < self._freeze_until_time

    def get_current_pos(self) -> Tuple[int, int]:
        """Return current or locked screen cursor coordinates."""
        return self._locked_pos if self._locked_pos is not None else self._current_pos

    def transform_coordinates(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """Transform normalized webcam frame coordinate to screen pixel coordinate.
        
        Applies active interaction margin clamping, two-stage smoothing, and deadband.
        """
        # Convert to webcam pixel space
        px = norm_x * self.cam_w
        py = norm_y * self.cam_h

        # Normalize relative to active interaction bounding box
        mx = self.map_cfg.margin_x
        my = self.map_cfg.margin_y
        active_w = max(10, self.cam_w - 2 * mx)
        active_h = max(10, self.cam_h - 2 * my)

        # Clamped ratio [0.0, 1.0]
        rx = max(0.0, min(1.0, (px - mx) / active_w))
        ry = max(0.0, min(1.0, (py - my) / active_h))

        # Non-linear acceleration / sensitivity curve
        if self.map_cfg.acceleration_exponent != 1.0:
            cx = (rx - 0.5) * 2.0
            cy = (ry - 0.5) * 2.0
            sign_x = 1.0 if cx >= 0 else -1.0
            sign_y = 1.0 if cy >= 0 else -1.0
            cx = sign_x * (abs(cx) ** self.map_cfg.acceleration_exponent)
            cy = sign_y * (abs(cy) ** self.map_cfg.acceleration_exponent)
            rx = max(0.0, min(1.0, cx * 0.5 + 0.5))
            ry = max(0.0, min(1.0, cy * 0.5 + 0.5))

        target_x = rx * self.screen_w
        target_y = ry * self.screen_h

        # Stage 2: One-Euro velocity-adaptive smoothing
        if self.smooth_cfg.filter_type == "one_euro":
            fx, fy = self.one_euro.filter(target_x, target_y)
        elif self.smooth_cfg.filter_type == "ema":
            fx, fy = self.adaptive_ema.filter(target_x, target_y)
        else:
            fx, fy = target_x, target_y

        # Stage 3: Deadband tremor suppression
        fx, fy = self.deadband.filter(fx, fy)

        screen_x = int(max(0, min(self.screen_w - 1, fx)))
        screen_y = int(max(0, min(self.screen_h - 1, fy)))
        return screen_x, screen_y

    def move(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """Move desktop cursor to mapped position, honoring active position locks."""
        now = time.perf_counter()
        if now < self._freeze_until_time and self._locked_pos is not None:
            # Cursor is locked during click: DO NOT MOVE!
            return self._locked_pos

        # Released from lock
        self._locked_pos = None

        sx, sy = self.transform_coordinates(norm_x, norm_y)
        self.driver.move(sx, sy)
        self._current_pos = (sx, sy)
        return sx, sy

    def left_click(self, freeze_sec: float = 0.35) -> None:
        """Execute left-click while locking cursor position."""
        self.freeze(freeze_sec)
        self.driver.left_click()

    def right_click(self, freeze_sec: float = 0.35) -> None:
        """Execute right-click while locking cursor position."""
        self.freeze(freeze_sec)
        self.driver.right_click()

    def set_drag_state(self, is_dragging: bool) -> None:
        if is_dragging and not self._is_dragging:
            self.driver.mouse_down()
            self._is_dragging = True
        elif not is_dragging and self._is_dragging:
            self.driver.mouse_up()
            self._is_dragging = False

    def scroll(self, delta_clicks: float) -> None:
        """Accumulate fractional scroll deltas and dispatch throttled, clamped wheel events."""
        self._scroll_accumulator += delta_clicks
        now = time.perf_counter()
        if (now - self._last_scroll_time) >= 0.04:
            ticks = int(self._scroll_accumulator)
            if ticks != 0:
                clamped_ticks = max(-2, min(2, ticks))
                self.driver.scroll(clamped_ticks)
                self._scroll_accumulator -= ticks
                self._last_scroll_time = now

    def reset(self) -> None:
        """Reset filters, scroll accumulator, position locks, and release mouse buttons."""
        if self._is_dragging:
            self.driver.mouse_up()
            self._is_dragging = False
        self._scroll_accumulator = 0.0
        self._freeze_until_time = 0.0
        self._locked_pos = None
        self._prev_norm_x = None
        self._prev_norm_y = None
        self.one_euro.reset()
        self.adaptive_ema.reset()
        self.deadband.reset()
