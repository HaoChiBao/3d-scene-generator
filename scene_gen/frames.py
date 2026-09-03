"""Extract reconstruction frames from images or video."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    *,
    target_fps: float = 1.0,
    max_frames: int = 24,
) -> list[Path]:
    """Sample frames from a video for multi-view reconstruction."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(int(round(src_fps / max(target_fps, 1e-3))), 1)

    paths: list[Path] = []
    index = 0
    saved = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if index % interval == 0:
            dest = out_dir / f"{saved:06d}.png"
            cv2.imwrite(str(dest), frame)
            paths.append(dest)
            saved += 1
        index += 1

    cap.release()
    if not paths:
        raise ValueError("No frames could be extracted from the video.")
    return paths


def extract_timed_frames(
    video_path: Path,
    out_dir: Path,
    *,
    target_fps: float = 1.0,
    max_frames: int = 24,
) -> list[tuple[Path, float]]:
    """Sample frames and record each frame's timestamp in seconds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if src_fps <= 1e-3:
        src_fps = 30.0
    interval = max(int(round(src_fps / max(target_fps, 1e-3))), 1)

    pairs: list[tuple[Path, float]] = []
    index = 0
    saved = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if index % interval == 0:
            dest = out_dir / f"{saved:06d}.png"
            cv2.imwrite(str(dest), frame)
            pairs.append((dest, float(index / src_fps)))
            saved += 1
        index += 1

    cap.release()
    if not pairs:
        raise ValueError("No frames could be extracted from the video.")
    return pairs


def prepare_timed_folder(
    media_path: Path,
    work_dir: Path,
    *,
    target_fps: float = 1.0,
    max_frames: int = 24,
) -> tuple[list[Path], list[float]]:
    """Normalize upload into frames plus timestamps in seconds."""
    images_dir = work_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    if media_path.is_dir():
        sources = sorted(
            p for p in media_path.iterdir() if p.is_file() and is_image(p)
        )
        if not sources:
            raise ValueError(f"No images found in {media_path}")
        sources = sources[:max_frames]
        paths: list[Path] = []
        for i, src in enumerate(sources):
            dest = images_dir / f"{i:06d}{src.suffix.lower()}"
            shutil.copy(src, dest)
            paths.append(dest)
        times = [float(i) for i in range(len(paths))]
        return paths, times

    if is_video(media_path):
        pairs = extract_timed_frames(
            media_path,
            images_dir,
            target_fps=target_fps,
            max_frames=max_frames,
        )
        return [p for p, _ in pairs], [t for _, t in pairs]

    if is_image(media_path):
        raise ValueError(
            "Video 4D needs a video or a folder of frames so time can change. "
            "Use Reconstruct or Orbit for a single still."
        )

    raise ValueError(
        f"Unsupported media type: {media_path.suffix}. "
        f"Use a video ({', '.join(sorted(VIDEO_SUFFIXES))}) "
        f"or a folder of images."
    )


def normalize_times(times: list[float]) -> list[float]:
    """Map timestamps to [0, 1]. A single frame stays at 0."""
    if not times:
        return []
    lo = min(times)
    hi = max(times)
    span = hi - lo
    if span <= 1e-9:
        return [0.0 for _ in times]
    return [(t - lo) / span for t in times]


def prepare_image_folder(
    media_path: Path,
    work_dir: Path,
    *,
    target_fps: float = 1.0,
    max_frames: int = 24,
) -> list[Path]:
    """
    Normalize upload into `work_dir/images/*.png|jpg`.

    Accepts a single image, a video, or a directory of images.
    """
    images_dir = work_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    if media_path.is_dir():
        sources = sorted(
            p for p in media_path.iterdir() if p.is_file() and is_image(p)
        )
        if not sources:
            raise ValueError(f"No images found in {media_path}")
        paths: list[Path] = []
        for i, src in enumerate(sources[:max_frames]):
            dest = images_dir / f"{i:06d}{src.suffix.lower()}"
            shutil.copy(src, dest)
            paths.append(dest)
        return paths

    if is_video(media_path):
        return extract_frames_from_video(
            media_path,
            images_dir,
            target_fps=target_fps,
            max_frames=max_frames,
        )

    if is_image(media_path):
        dest = images_dir / f"000000{media_path.suffix.lower()}"
        shutil.copy(media_path, dest)
        return [dest]

    raise ValueError(
        f"Unsupported media type: {media_path.suffix}. "
        f"Use an image ({', '.join(sorted(IMAGE_SUFFIXES))}) "
        f"or video ({', '.join(sorted(VIDEO_SUFFIXES))})."
    )
