"""Load arbitrary MP4 clips with the same frame interface as SoccerNet sequences."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoSequence:
    """Duck-compatible with :class:`SoccerNetSequence` for demo renderers."""

    name: str
    video_path: Path
    width: int
    height: int
    frame_rate: float
    length: int

    def frame_path(self, frame_idx: int) -> Path:
        raise NotImplementedError(
            f"{self.name}: use read_sequence_frame() for video sources"
        )


def load_video_sequence(path: str | Path) -> VideoSequence:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if length <= 0:
        raise RuntimeError(f"Video has no frames: {path}")
    return VideoSequence(
        name=path.stem,
        video_path=path.resolve(),
        width=width,
        height=height,
        frame_rate=fps,
        length=length,
    )


def read_sequence_frame(sequence, frame_idx: int) -> np.ndarray | None:
    """Read one 1-indexed BGR frame from a SoccerNet dir or an MP4 sequence."""
    video_path = getattr(sequence, "video_path", None)
    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx - 1, 0))
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    return cv2.imread(str(sequence.frame_path(frame_idx)))


class SequentialVideoReader:
    """Forward-only MP4 reader (1-indexed). Avoids unreliable per-frame OpenCV seek."""

    def __init__(self, video_path: str | Path) -> None:
        self._path = Path(video_path)
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {self._path}")
        self._next_idx = 1

    def read(self, frame_idx: int) -> np.ndarray | None:
        if frame_idx < self._next_idx:
            self._cap.release()
            self._cap = cv2.VideoCapture(str(self._path))
            if not self._cap.isOpened():
                return None
            self._next_idx = 1
        while self._next_idx < frame_idx:
            ok, _ = self._cap.read()
            if not ok:
                return None
            self._next_idx += 1
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._next_idx += 1
        return frame

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> SequentialVideoReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class H264StreamWriter:
    """Pipe BGR frames to ffmpeg libx264 (no OpenCV mp4v intermediate)."""

    def __init__(
        self,
        path: str | Path,
        *,
        width: int,
        height: int,
        fps: float,
        crf: int = 16,
        preset: str = "medium",
    ) -> None:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("Video export needs ffmpeg on PATH")
        self._path = Path(path)
        self._width = width
        self._height = height
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self._path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("H264StreamWriter is closed")
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            raise ValueError("Frame dimensions do not match writer")
        assert self._proc.stdin is not None
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        self._proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed to encode video: {self._path}")

    def __enter__(self) -> H264StreamWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def write_h264_video(
    path: str | Path,
    frames_bgr: list[np.ndarray],
    *,
    fps: float,
    crf: int = 16,
    preset: str = "slow",
) -> Path:
    """Encode BGR frames to h264 in one pass (no OpenCV mp4v intermediate)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Video export needs ffmpeg on PATH")
    if not frames_bgr:
        raise ValueError("No frames for video")
    path = Path(path)
    h, w = frames_bgr[0].shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames_bgr:
            if frame.shape[0] != h or frame.shape[1] != w:
                raise ValueError("All frames must share the same dimensions")
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to encode video: {path}")
    return path


def finalize_video_for_playback(path: Path, *, crf: int = 16) -> bool:
    """Re-encode OpenCV mp4v output to h264 yuv420p for broad playback (no green screen)."""
    if not shutil.which("ffmpeg"):
        return False
    path = Path(path)
    tmp = path.with_suffix(".h264.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(path)
        return True
    except subprocess.CalledProcessError:
        if tmp.exists():
            tmp.unlink()
        return False


def write_gif_from_mp4(
    gif_path: str | Path,
    mp4_path: str | Path,
    *,
    fps: float = 8.0,
    width: int | None = 1280,
) -> Path:
    """High-quality GIF via ffmpeg palettegen (much sharper than Pillow defaults)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("GIF export needs ffmpeg on PATH")
    gif_path = Path(gif_path)
    mp4_path = Path(mp4_path)
    scale = f"scale={width}:-1:flags=lanczos," if width else ""
    graph = (
        f"[0:v]{scale}fps={fps},split[s0][s1];"
        "[s0]palettegen=stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3[out]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp4_path),
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            str(gif_path),
        ],
        check=True,
    )
    return gif_path
