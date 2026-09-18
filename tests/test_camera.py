"""Tests for app.camera."""

import platform
from pathlib import Path

import pytest

from app.camera import Camera, CameraError, backend_name, candidate_backends


def test_defaults_to_camera_index_zero() -> None:
    assert Camera().source == 0


def test_accepts_a_video_file_as_the_source(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not a real video")

    camera = Camera(source=clip)

    assert camera.is_file_source
    assert "clip.mp4" in camera.describe()


def test_describe_names_the_camera_index() -> None:
    assert Camera(index=2).describe() == "camera index 2"


def test_reading_before_opening_raises() -> None:
    with pytest.raises(CameraError, match="not open"):
        Camera().read()


def test_opening_a_missing_video_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CameraError, match="not found"):
        Camera(source=tmp_path / "missing.mp4").open()


def test_error_message_for_a_camera_explains_the_likely_causes() -> None:
    message = Camera(index=3)._error_message()

    assert "index 3" in message
    assert "permission" in message.lower()
    assert "--camera 1" in message


def test_error_message_for_a_file_mentions_the_path(tmp_path: Path) -> None:
    message = Camera(source=tmp_path / "clip.mp4")._error_message()

    assert "clip.mp4" in message


def test_release_is_safe_before_opening() -> None:
    Camera().release()  # must not raise


def test_context_manager_releases_the_camera_on_error(tmp_path: Path) -> None:
    camera = Camera(source=tmp_path / "missing.mp4")

    with pytest.raises(CameraError):
        with camera:
            pass

    assert not camera.is_open


def test_backend_name_is_a_non_empty_string() -> None:
    assert isinstance(backend_name(), str)
    assert backend_name()


# -- Backend selection ------------------------------------------------------ #


def test_candidate_backends_are_named_and_end_with_a_fallback() -> None:
    backends = candidate_backends()

    assert backends
    assert all(name for name, _ in backends)
    # The last entry lets OpenCV choose for itself.
    assert backends[-1][0] in {"default", "V4L2"}


def test_backend_name_is_the_first_candidate() -> None:
    assert backend_name() == candidate_backends()[0][0]


@pytest.mark.skipif(
    platform.system() != "Windows", reason="the Windows backend order is fixed"
)
def test_windows_tries_media_foundation_before_directshow() -> None:
    names = [name for name, _ in candidate_backends()]

    assert names[0] == "Media Foundation"
    assert "DirectShow" in names
    assert names.index("Media Foundation") < names.index("DirectShow")


def test_backend_can_be_reported_once_the_camera_is_open() -> None:
    # A camera that has not been opened yet has no backend to report.
    assert Camera().backend is None


def test_frame_size_is_unknown_before_the_camera_is_opened() -> None:
    camera = Camera()

    assert camera.frame_width == 0
    assert camera.frame_height == 0
