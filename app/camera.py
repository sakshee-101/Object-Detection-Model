"""Webcam capture.

Module 1 of the project: open the webcam, hand frames to the detector and
release the device cleanly when the application exits.

Capture backends differ between operating systems. On Windows, Media
Foundation is tried first and DirectShow is kept as a fallback; on macOS,
AVFoundation is used; elsewhere OpenCV's own choice is used. A backend is only
accepted once it has actually delivered a frame, because some drivers report a
successful open and then stay silent.

The same class also accepts a video file instead of a camera index. That is not
a project feature, it is simply a convenient way to run the pipeline on a
machine that has no webcam attached.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any, Sequence

import cv2

__all__ = ["Camera", "CameraError", "backend_name", "candidate_backends"]

#: How many frames to ask for while checking that a backend really works. Some
#: drivers report a successful open and only start delivering frames a moment
#: later, so the first few reads are allowed to fail.
WARMUP_FRAMES = 10

#: Seconds to wait between those warm-up reads.
WARMUP_DELAY = 0.05


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a frame cannot be read."""


def _any_backend() -> int:
    """Return the constant that tells OpenCV to choose a backend itself."""
    return getattr(cv2, "CAP_ANY", 0)


def candidate_backends() -> list[tuple[str, int]]:
    """Return the capture backends to try for a live camera, best first.

    Windows is offered Media Foundation first, because it is the current
    backend there, with DirectShow kept as a fallback for older drivers. Every
    platform ends with ``CAP_ANY`` so that OpenCV can still choose for itself
    if the named backends are refused.
    """
    any_backend = _any_backend()

    if platform.system() == "Darwin":
        return [
            ("AVFoundation", getattr(cv2, "CAP_AVFOUNDATION", any_backend)),
            ("default", any_backend),
        ]
    if platform.system() == "Windows":
        return [
            ("Media Foundation", getattr(cv2, "CAP_MSMF", any_backend)),
            ("DirectShow", getattr(cv2, "CAP_DSHOW", any_backend)),
            ("default", any_backend),
        ]
    return [("V4L2", any_backend)]


def backend_name() -> str:
    """Return the preferred OpenCV capture backend for this operating system."""
    return candidate_backends()[0][0]


class Camera:
    """A webcam capture source.

    Args:
        index: Webcam device index. ``0`` is the built-in camera on most laptops.
        width: Requested frame width, or ``None`` to keep the driver default.
        height: Requested frame height, or ``None`` to keep the driver default.
        source: Optional path to a video file. When given, ``index`` is ignored.
    """

    def __init__(
        self,
        index: int = 0,
        width: int | None = None,
        height: int | None = None,
        source: str | Path | None = None,
    ) -> None:
        self.source: int | str = str(source) if source is not None else int(index)
        self.width = width
        self.height = height
        self._capture: Any | None = None
        self._frames_read = 0
        self._backend: str | None = None

    # -- state ------------------------------------------------------------- #

    @property
    def is_file_source(self) -> bool:
        """``True`` when frames come from a video file rather than a camera."""
        return isinstance(self.source, str)

    @property
    def is_open(self) -> bool:
        """``True`` while the underlying capture device is open."""
        return self._capture is not None and self._capture.isOpened()

    @property
    def frames_read(self) -> int:
        """Number of frames successfully read since :meth:`open`."""
        return self._frames_read

    @property
    def backend(self) -> str | None:
        """Name of the backend that actually delivered frames, if known.

        This is the backend that succeeded, which is not necessarily the
        preferred one for the platform: on Windows, Media Foundation is tried
        first and DirectShow is only used if the first attempt fails.
        """
        return self._backend

    @property
    def frame_width(self) -> int:
        """Frame width reported by the device, or ``0`` when unknown."""
        return self._device_property(cv2.CAP_PROP_FRAME_WIDTH)

    @property
    def frame_height(self) -> int:
        """Frame height reported by the device, or ``0`` when unknown."""
        return self._device_property(cv2.CAP_PROP_FRAME_HEIGHT)

    def _device_property(self, prop: int) -> int:
        """Read a numeric capture property, treating any failure as unknown."""
        if self._capture is None:
            return 0
        try:
            return int(self._capture.get(prop) or 0)
        except Exception:
            return 0

    def describe(self) -> str:
        """Return a short description of the capture source."""
        if self.is_file_source:
            return f"video file '{self.source}'"
        return f"camera index {self.source}"

    # -- lifecycle --------------------------------------------------------- #

    def open(self) -> "Camera":
        """Open the capture source.

        A live camera is opened with the first backend that both accepts the
        device and delivers a frame. See :func:`candidate_backends`.

        Returns:
            ``self``, so the call can be chained.

        Raises:
            CameraError: If the device or video file cannot be opened.
        """
        if self.is_open:
            return self

        if self.is_file_source:
            path = Path(str(self.source)).expanduser()
            if not path.exists():
                raise CameraError(f"Error: video file not found: {path}")
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise CameraError(self._error_message())
            self._backend = "video file"
        else:
            capture = self._open_camera()

        self._capture = capture
        self._frames_read = 0
        return self

    def _open_camera(self) -> Any:
        """Open the webcam with the first backend that really works.

        A backend is only accepted once it has delivered a frame. Some drivers
        report a successful open and then never produce anything, so opening
        the device is not enough on its own to prove that it works.

        Raises:
            CameraError: If no backend produced a usable frame.
        """
        attempts: list[str] = []
        any_backend = _any_backend()

        for name, flag in candidate_backends():
            if flag == any_backend:
                capture = cv2.VideoCapture(self.source)
            else:
                capture = cv2.VideoCapture(self.source, flag)

            if capture.isOpened():
                self._apply_size(capture)
                if self._wait_for_first_frame(capture):
                    self._backend = name
                    return capture
                attempts.append(f"{name}: opened but delivered no frames")
            else:
                attempts.append(f"{name}: could not open the device")

            capture.release()

        raise CameraError(self._error_message(attempts))

    def _apply_size(self, capture: Any) -> None:
        """Ask the driver for the requested capture size, if one was given."""
        if self.width and self.height:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

    def _wait_for_first_frame(self, capture: Any) -> bool:
        """Return ``True`` once the capture delivers a usable frame.

        A camera that has only just been switched on often needs a moment
        before the first frame arrives, so a short warm-up is normal rather
        than a failure.
        """
        for _ in range(WARMUP_FRAMES):
            ok, frame = capture.read()
            if ok and frame is not None:
                return True
            time.sleep(WARMUP_DELAY)
        return False

    def _error_message(self, attempts: Sequence[str] | None = None) -> str:
        """Build a helpful message explaining why the source would not open.

        Args:
            attempts: Optional notes about the backends that were tried.
        """
        if self.is_file_source:
            return f"Error: could not open the video file '{self.source}'."

        lines = [f"Error: could not open a webcam at camera index {self.source}."]

        if attempts:
            lines.append("Backends tried:")
            lines.extend(f"  - {attempt}" for attempt in attempts)

        lines.extend([
            "Possible reasons:",
            "  - No webcam is connected.",
            "  - Another application is already using the camera.",
            "  - The operating system has not granted camera permission to the",
            "    program you are running from.",
            "      Windows: Settings > Privacy & security > Camera",
            "      macOS  : System Settings > Privacy & Security > Camera",
            "  - The wrong index was used; try --camera 1.",
        ])
        return "\n".join(lines)

    def read(self) -> tuple[bool, Any | None]:
        """Read the next frame from the source.

        Returns:
            A ``(ok, frame)`` pair. ``ok`` is ``False`` when the stream has
            ended or the frame could not be decoded.

        Raises:
            CameraError: If the camera has not been opened.
        """
        if not self.is_open:
            raise CameraError("Camera is not open. Call open() first.")

        ok, frame = self._capture.read()
        if ok and frame is not None:
            self._frames_read += 1
            return True, frame
        return False, None

    def release(self) -> None:
        """Release the camera. Safe to call more than once."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._backend = None

    # -- context manager --------------------------------------------------- #

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"Camera(source={self.source!r}, open={self.is_open})"
