"""Video helpers: probe duration and sample frames with ffmpeg (no heavy CV deps)."""
from __future__ import annotations

import math
import shutil
import subprocess

from . import config


class VideoError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def probe_duration(path: str) -> float:
    if not ffmpeg_available():
        raise VideoError("ffmpeg/ffprobe not installed on the server; video analysis is unavailable.")
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        dur = float(out)
    except (subprocess.SubprocessError, ValueError) as exc:
        raise VideoError("Could not read the video. Is it a valid video file?") from exc
    if dur <= 0:
        raise VideoError("Video has zero duration.")
    if dur > config.MAX_VIDEO_SECONDS:
        raise VideoError(f"Video is {dur:.0f}s; the limit is {config.MAX_VIDEO_SECONDS}s.")
    return dur


def frame_times(duration: float, max_frames: int = 16) -> list[float]:
    """Dense sampling in the first 5s (the hook), sparse through the middle, dense on the last 2s."""
    pts = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    middle = 8
    pts += [5.0 + (duration - 5.0) * (i + 1) / (middle + 1) for i in range(middle)] if duration > 7 else []
    pts += [duration - 1.5, duration - 0.3]
    pts = sorted({round(min(max(t, 0.0), max(duration - 0.1, 0.0)), 1) for t in pts})
    if len(pts) > max_frames:                       # thin the middle, keep hook + ending
        keep_head = [t for t in pts if t <= 5.0]
        keep_tail = [t for t in pts if t >= duration - 2.0 and t > 5.0]
        mid = [t for t in pts if t not in keep_head and t not in keep_tail]
        room = max(0, max_frames - len(keep_head) - len(keep_tail))
        step = max(1, math.ceil(len(mid) / room)) if room else len(mid) + 1
        pts = sorted(keep_head + mid[::step][:room] + keep_tail)
    return pts


def extract_frames(path: str, times: list[float], width: int = 512) -> list[tuple[float, bytes]]:
    frames: list[tuple[float, bytes]] = []
    for t in times:
        try:
            res = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
                 "-vf", f"scale={width}:-2", "-q:v", "5", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
                capture_output=True, timeout=30, check=True,
            )
        except subprocess.SubprocessError:
            continue
        if res.stdout:
            frames.append((t, res.stdout))
    if not frames:
        raise VideoError("Could not extract any frames from the video.")
    return frames
