# Changelog

All notable changes to **Touchless Hand Gesture Mouse Control** are documented in this file.

---

## [1.2.0] - 2026-09-18

### Added
- **Digital Camera Zoom (1.0x to 4.0x)**: Universal center-crop zoom engine with linear interpolation. Hand tracking, active boundaries, and HUD telemetry stay perfectly aligned.
- **In-App Zoom Shortcuts**: Key `Z` to cycle presets (`1.0x`, `1.25x`, `1.5x`, `2.0x`, `3.0x`), `+`/`-` to fine-tune (0.1x steps), `0` to reset.
- **CLI Argument `--zoom`**: Allows setting initial digital zoom factor directly from the command line.
- **Drag Release Hysteresis**: 5-frame grace buffer to prevent accidental drops during transient landmark noise while moving an object.
- **Orientation-Invariant Finger Curling**: 3D Euclidean joint checks relative to wrist and knuckles (MCP) so gestures work naturally at any hand angle or tilt.

### Changed
- **Drag & Drop Postures**: Added support for both pointing "finger-gun" posture (index extended, other fingers curled) and clustered thumb pinch postures.
- **Click Isolation**: Left click (pinky+thumb) and Right click (ring+thumb) now strictly require middle finger extension, completely eliminating accidental clicks during finger gathering.

---

## [1.1.0] - 2026-09-17

### Added
- **60 FPS Video Pipeline**: Enabled MJPEG FourCC hardware codec (`cv2.CAP_PROP_FOURCC = 'MJPG'`) under Windows DirectShow to bypass USB bandwidth bottlenecks.
- **Cross-Computer Portability**: Created self-healing portable launchers (`run.bat`, `run.ps1`) that detect Python installations dynamically without hardcoded paths.
- **Standalone Windows Distribution**: Pre-compiled PyInstaller bundle (`HandGestureMouse.exe`) containing full MediaPipe Tasks C-extensions and TFLite delegates.

---

## [1.0.0] - 2026-09-17

### Added
- **Initial Release**: Core Human-Computer Interaction application in Python.
- **21 3D Landmark Tracking**: MediaPipe Tasks HandLandmarker (`hand_landmarker.task`) with monotonic timestamping.
- **One-Euro Filter (`1€ Filter`)**: Adaptive velocity-dependent cutoff frequency for tremor elimination and zero-lag tracking.
- **Windows Win32 Native API**: Sub-millisecond cursor positioning and click injection using `ctypes.windll.user32`.
- **Augmented Reality HUD**: Glassmorphic top banner, active interaction bounding brackets, and visual ripple feedback.
- **Automated Test Suite**: Comprehensive unit tests covering filters, active zone transformations, debouncing, and synthetic cameras.
