"""Single-image orbit: generate novel views with Gemini or OpenAI."""

from __future__ import annotations

import json
import math
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image


ProgressFn = Callable[[str, dict[str, Any]], None]

PROVIDERS = {
    "gemini": {
        "label": "Gemini",
        "default_model": "gemini-2.5-flash-image",
        "models": [
            "gemini-2.5-flash-image",
            "gemini-2.5-flash-image-preview",
        ],
    },
    "openai": {
        "label": "OpenAI",
        "default_model": "gpt-image-1",
        "models": ["gpt-image-1"],
    },
}


def iter_angles(start_deg: float, end_deg: float, increment_deg: float) -> list[float]:
    """Angles from start (inclusive) to end (exclusive). 0→360 step 10 → 0,10,…,350."""
    if increment_deg <= 0:
        raise ValueError("increment_deg must be > 0")
    if end_deg <= start_deg:
        raise ValueError("end_deg must be greater than start_deg")
    span = end_deg - start_deg
    if span / increment_deg > 72:
        raise ValueError("Too many views (max 72). Increase increment or shrink the range.")

    angles: list[float] = []
    a = float(start_deg)
    while a < end_deg - 1e-6:
        angles.append(round(a % 360 if a != 360 else 0.0, 4))
        a += increment_deg
    if not angles:
        raise ValueError("No angles in range")
    return angles


def camera_xyz(
    azimuth_deg: float,
    elevation_deg: float,
    radius: float,
) -> tuple[float, float, float]:
    """Y-up world: azimuth 0 sits on +Z looking at the origin."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = radius * math.sin(az) * math.cos(el)
    y = radius * math.sin(el)
    z = radius * math.cos(az) * math.cos(el)
    return (x, y, z)


CONTEXT_MODES = ("nearest", "previous", "original")
ORDER_MODES = ("bidirectional", "sequential")


def angular_distance(a: float, b: float) -> float:
    return min((a - b) % 360.0, (b - a) % 360.0)


def generation_order(angles: list[float], order_mode: str = "bidirectional") -> list[float]:
    """Bidirectional walks ±increment from the first angle so the back view is fewer hops."""
    if order_mode not in ORDER_MODES:
        raise ValueError(f"Unknown order_mode: {order_mode}")
    if order_mode == "sequential" or not angles:
        return list(angles)
    origin = angles[0]

    def sort_key(angle: float) -> tuple[float, int]:
        dist = angular_distance(angle, origin)
        signed = ((angle - origin + 180.0) % 360.0) - 180.0
        return (dist, 0 if signed >= 0 else 1)

    return sorted(angles, key=sort_key)


def select_context(
    target_angle: float,
    generated: list[dict[str, Any]],
    *,
    mode: str,
    max_neighbors: int = 2,
) -> list[dict[str, Any]]:
    """Always keep the original. Optionally add previous or spatially nearest generated views."""
    if mode not in CONTEXT_MODES:
        raise ValueError(f"Unknown context_mode: {mode}")
    if not generated:
        return []
    original = generated[0]
    refs = [original]
    others = [f for f in generated[1:] if f.get("image") is not None]
    if mode == "original" or not others:
        return refs
    if mode == "previous":
        refs.append(others[-1])
        return refs
    nearest = sorted(others, key=lambda f: angular_distance(target_angle, float(f["angle"])))
    refs.extend(nearest[: max(0, int(max_neighbors))])
    return refs


def describe_azimuth(azimuth_deg: float) -> str:
    az = azimuth_deg % 360
    if az < 1 or az > 359:
        return "the original viewpoint (front)"
    if az < 45:
        return f"{az:.0f}° to the right of the original view"
    if az < 135:
        return f"the right side ({az:.0f}° clockwise)"
    if az < 225:
        return f"behind the subject ({az:.0f}°)"
    if az < 315:
        return f"the left side ({az:.0f}° clockwise)"
    return f"{az:.0f}° to the left of the original view"


def build_prompt(
    azimuth_deg: float,
    elevation_deg: float,
    extra: str = "",
    context_angles: list[float] | None = None,
) -> str:
    where = describe_azimuth(azimuth_deg)
    extra = (extra or "").strip()
    tail = f"\nAdditional direction: {extra}" if extra else ""
    refs = context_angles or [0.0]
    ref_line = ", ".join(f"{a:.0f}°" for a in refs)
    return (
        "You are given one or more reference photographs of the SAME real scene.\n"
        f"The first image is the identity lock (original capture). "
        f"Any later images are already-generated nearby views at [{ref_line}].\n"
        f"Synthesize the scene from a new camera pose: orbit {azimuth_deg:.0f}° "
        f"clockwise — {where}. Elevation {elevation_deg:.0f}°, same distance.\n"
        "Match both identity (from the original) and local continuity (from nearby views). "
        "Do not invent a different room, lighting, or objects. "
        "Only the viewpoint should change. No text, borders, frames, or watermarks. "
        "Match the original aspect ratio and photographic style."
        f"{tail}"
    )


def resolve_api_key(provider: str, override: str | None = None) -> str:
    if override and override.strip():
        return override.strip()
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
    else:
        raise ValueError(f"Unknown provider: {provider}")
    if not key:
        raise RuntimeError(
            f"No API key for {provider}. Paste one in the form, or set "
            f"{'GEMINI_API_KEY' if provider == 'gemini' else 'OPENAI_API_KEY'}."
        )
    return key


def generate_view(
    images: Image.Image | list[Image.Image],
    *,
    provider: str,
    model: str,
    prompt: str,
    api_key: str,
) -> bytes:
    refs = images if isinstance(images, list) else [images]
    refs = [im.convert("RGB") for im in refs]
    if not refs:
        raise ValueError("Need at least one reference image")
    if provider == "gemini":
        return _generate_gemini(refs, model=model, prompt=prompt, api_key=api_key)
    if provider == "openai":
        return _generate_openai(refs, model=model, prompt=prompt, api_key=api_key)
    raise ValueError(f"Unknown provider: {provider}")


def _generate_gemini(
    images: list[Image.Image], *, model: str, prompt: str, api_key: str
) -> bytes:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[prompt, *images],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    for cand in response.candidates or []:
        parts = getattr(cand.content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return bytes(inline.data)
    raise RuntimeError("Gemini returned no image")


def _generate_openai(
    images: list[Image.Image], *, model: str, prompt: str, api_key: str
) -> bytes:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    buffers = []
    for i, image in enumerate(images):
        buf = BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        buf.name = f"ref_{i}.png"
        buffers.append(buf)
    result = client.images.edit(
        model=model,
        image=buffers if len(buffers) > 1 else buffers[0],
        prompt=prompt,
        size=_openai_size(images[0]),
    )
    item = result.data[0]
    if getattr(item, "b64_json", None):
        import base64

        return base64.b64decode(item.b64_json)
    raise RuntimeError("OpenAI returned no image bytes")


def _openai_size(image: Image.Image) -> str:
    w, h = image.size
    ratio = w / max(h, 1)
    if ratio > 1.2:
        return "1536x1024"
    if ratio < 0.85:
        return "1024x1536"
    return "1024x1024"


def reconstruct_orbit(
    media_path: Path,
    output_dir: Path,
    *,
    increment_deg: float = 10.0,
    start_deg: float = 0.0,
    end_deg: float = 360.0,
    elevation_deg: float = 12.0,
    radius: float = 2.4,
    fov_deg: float = 45.0,
    provider: str = "gemini",
    model: str | None = None,
    extra_prompt: str = "",
    api_key: str | None = None,
    context_mode: str = "nearest",
    order_mode: str = "bidirectional",
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    context_mode = (context_mode or "nearest").strip().lower()
    order_mode = (order_mode or "bidirectional").strip().lower()
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"Unknown context_mode: {context_mode}")
    if order_mode not in ORDER_MODES:
        raise ValueError(f"Unknown order_mode: {order_mode}")
    model = (model or PROVIDERS[provider]["default_model"]).strip()
    angles = iter_angles(start_deg, end_deg, increment_deg)
    work_order = generation_order(angles, order_mode)
    key = resolve_api_key(provider, api_key)

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    source = Image.open(media_path).convert("RGB")
    source_path = output_dir / "source.jpg"
    source.save(source_path, quality=92)

    generated: list[dict[str, Any]] = []

    def public_frames() -> list[dict[str, Any]]:
        return [_public_frame(f) for f in generated]

    def emit(message: str) -> None:
        meta = {
            "model": f"{provider}/{model}",
            "kind": "orbit",
            "frames": public_frames(),
            "params": _params(
                increment_deg,
                start_deg,
                end_deg,
                elevation_deg,
                radius,
                fov_deg,
                provider,
                model,
                extra_prompt,
                angles,
                context_mode,
                order_mode,
            ),
        }
        if on_progress:
            on_progress(message, meta)

    for step, angle in enumerate(work_order):
        frame_name = f"{int(round(angle)):03.0f}deg.jpg"
        frame_path = frames_dir / frame_name
        use_original = abs(angle) < 1e-3 or (not generated and abs(angle - angles[0]) < 1e-3)
        context_angles: list[float] = []
        if use_original and not generated:
            source.save(frame_path, quality=92)
            image = source
            source_kind = "original"
        else:
            refs = select_context(angle, generated, mode=context_mode)
            if not refs:
                refs = [{"angle": 0.0, "image": source, "source": "original"}]
            context_angles = [float(r["angle"]) for r in refs]
            emit(
                f"Generating {angle:.0f}° ({step + 1}/{len(work_order)}) "
                f"· context {', '.join(f'{a:.0f}°' for a in context_angles)}"
            )
            prompt = build_prompt(
                angle, elevation_deg, extra_prompt, context_angles=context_angles
            )
            raw = generate_view(
                [r["image"] for r in refs],
                provider=provider,
                model=model,
                prompt=prompt,
                api_key=key,
            )
            image = Image.open(BytesIO(raw)).convert("RGB")
            image.save(frame_path, quality=92)
            source_kind = "generated"

        generated.append(
            {
                "index": len(generated),
                "angle": angle,
                "file": f"frames/{frame_name}",
                "source": source_kind,
                "xyz": camera_xyz(angle, elevation_deg, radius),
                "context_angles": context_angles,
                "image": image,
            }
        )
        emit(f"Ready {angle:.0f}° ({step + 1}/{len(work_order)})")

    generated.sort(key=lambda f: float(f["angle"]))
    for i, frame in enumerate(generated):
        frame["index"] = i

    frames = public_frames()
    meta = {
        "model": f"{provider}/{model}",
        "kind": "orbit",
        "has_gaussians": False,
        "n_views": len(frames),
        "frames": frames,
        "params": _params(
            increment_deg,
            start_deg,
            end_deg,
            elevation_deg,
            radius,
            fov_deg,
            provider,
            model,
            extra_prompt,
            angles,
            context_mode,
            order_mode,
        ),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _public_frame(frame: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in frame.items() if k != "image"}


def _params(
    increment_deg: float,
    start_deg: float,
    end_deg: float,
    elevation_deg: float,
    radius: float,
    fov_deg: float,
    provider: str,
    model: str,
    extra_prompt: str,
    angles: Iterable[float],
    context_mode: str = "nearest",
    order_mode: str = "bidirectional",
) -> dict[str, Any]:
    angle_list = list(angles)
    return {
        "increment_deg": increment_deg,
        "start_deg": start_deg,
        "end_deg": end_deg,
        "elevation_deg": elevation_deg,
        "radius": radius,
        "fov_deg": fov_deg,
        "provider": provider,
        "model": model,
        "extra_prompt": extra_prompt,
        "context_mode": context_mode,
        "order_mode": order_mode,
        "angles": angle_list,
        "n_views": len(angle_list),
    }
