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
        "default_model": "gemini-3.1-flash-image",
        "models": [
            "gemini-3.1-flash-image",
            "gemini-3-pro-image",
            "gemini-2.5-flash-image",
            "gemini-2.5-flash-image-preview",
            "gemini-3.1-flash-lite-image",
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-image-preview",
        ],
    },
    "openai": {
        "label": "OpenAI",
        "default_model": "gpt-image-2",
        "models": [
            "gpt-image-2",
            "gpt-image-1.5",
            "gpt-image-1",
            "gpt-image-1-mini",
        ],
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


CONTEXT_MODES = ("adaptive", "nearest", "previous", "original")
ORDER_MODES = ("cardinal", "bidirectional", "sequential")
DEFAULT_ORIGINAL_LOCK_DEG = 60.0
CARDINAL_OFFSETS = (90.0, 270.0, 180.0)

SCENE_BRIEF_PROMPT = """Look at this real photograph. Write a compact orbit brief
(160-220 words) with these headings, in this order:
People: each person separately. Face, hair, skin, age, clothes, pose, gaze, and
where they stand (left/center/right, near/far). If none, say none.
Small objects: named items and exact placement (on the table, third shelf, in a
hand, on the floor). Include text on labels, screens, books, and signs.
Placement: left-to-right and near-to-far anchors that must stay in world space.
Layout: indoor/outdoor, room type, openings, likely shape.
Right (90°): what a quarter-turn right should reveal. Be concrete.
Back (180°): what is behind the subject. Do not describe the front again.
Left (270°): what a quarter-turn left should reveal.
Constants: materials, lighting, white balance, floor, time of day, weather.
No cameras, no markdown fences, no bullet characters."""

PRESERVE_LOCK = (
    "Preserve (do not restyle, replace, or invent):\n"
    "- People: exact likeness, face, hair, skin, age, clothes, body shape, "
    "expression, and where each person stands in the room.\n"
    "- Small objects: same items, count, color, wear, printed text, and "
    "world-space placement. Do not drop or add clutter.\n"
    "- Furniture and architecture: same pieces in the same places; only the "
    "camera moves around them.\n"
    "- Light: same time of day, color temperature, shadow direction, and grain.\n"
    "- Photorealistic photograph of this real place. No illustration, no extra "
    "people, no text overlay, no borders, no watermarks."
)


def angular_distance(a: float, b: float) -> float:
    return min((a - b) % 360.0, (b - a) % 360.0)


def generation_order(angles: list[float], order_mode: str = "cardinal") -> list[float]:
    """Cardinal-first plants 90/180/270 anchors, then fills. Bidirectional walks ±step."""
    if order_mode not in ORDER_MODES:
        raise ValueError(f"Unknown order_mode: {order_mode}")
    if order_mode == "sequential" or not angles:
        return list(angles)
    origin = angles[0]
    if order_mode == "bidirectional":
        def sort_key(angle: float) -> tuple[float, int]:
            dist = angular_distance(angle, origin)
            signed = ((angle - origin + 180.0) % 360.0) - 180.0
            return (dist, 0 if signed >= 0 else 1)

        return sorted(angles, key=sort_key)

    order: list[float] = [origin]
    remaining = [a for a in angles if a != origin]
    for offset in CARDINAL_OFFSETS:
        if not remaining:
            break
        target = (origin + offset) % 360.0
        pick = min(remaining, key=lambda a: (angular_distance(a, target), a))
        order.append(pick)
        remaining.remove(pick)
    while remaining:
        pick = min(
            remaining,
            key=lambda a: min(angular_distance(a, planted) for planted in order),
        )
        order.append(pick)
        remaining.remove(pick)
    return order


def select_context(
    target_angle: float,
    generated: list[dict[str, Any]],
    *,
    mode: str,
    max_neighbors: int = 2,
    origin_angle: float = 0.0,
    original_lock_deg: float = DEFAULT_ORIGINAL_LOCK_DEG,
) -> list[dict[str, Any]]:
    """Pick reference views. Adaptive keeps the original only near the front."""
    if mode not in CONTEXT_MODES:
        raise ValueError(f"Unknown context_mode: {mode}")
    if not generated:
        return []

    originals = [f for f in generated if f.get("source") == "original"]
    original = originals[0] if originals else generated[0]
    origin_angle = float(original.get("angle", origin_angle))
    others = [
        f
        for f in generated
        if f.get("image") is not None and f is not original
    ]

    lock = max(0.0, float(original_lock_deg))
    near_front = angular_distance(target_angle, origin_angle) <= lock + 1e-6
    include_original = False
    if mode == "original":
        include_original = True
    elif mode == "nearest":
        include_original = True
    elif mode in {"adaptive", "previous"}:
        include_original = near_front or not others

    refs: list[dict[str, Any]] = []
    if include_original:
        refs.append(original)
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


def facing_name(azimuth_deg: float) -> str:
    az = azimuth_deg % 360
    if az < 1 or az > 359:
        return "front"
    if az < 45:
        return "front-right"
    if az < 135:
        return "right"
    if az < 225:
        return "back"
    if az < 315:
        return "left"
    return "front-left"


def _label_refs(
    context_angles: list[float] | None,
    context_refs: list[dict[str, Any]] | None,
) -> str:
    if context_refs:
        lines = []
        for i, ref in enumerate(context_refs, start=1):
            angle = float(ref.get("angle", 0.0))
            source = ref.get("source") or "generated"
            if source == "original":
                role = "original front still. Identity lock for people, objects, and materials"
            else:
                role = f"already-generated view at {angle:.0f}°. Local continuity only"
            lines.append(f"Image {i}: {role}.")
        return "\n".join(lines)
    if context_angles:
        labeled = ", ".join(f"{a:.0f}°" for a in context_angles)
        return f"Reference views, in order: [{labeled}]. Other poses, not the target."
    return "No extra references."


def build_prompt(
    azimuth_deg: float,
    elevation_deg: float,
    extra: str = "",
    context_angles: list[float] | None = None,
    context_refs: list[dict[str, Any]] | None = None,
    scene_brief: str = "",
    origin_angle: float = 0.0,
    includes_original: bool = False,
) -> str:
    where = describe_azimuth(azimuth_deg)
    side = facing_name(azimuth_deg)
    extra = (extra or "").strip()
    tail = f"\nAdditional direction: {extra}" if extra else ""
    turn = angular_distance(azimuth_deg, origin_angle)
    brief = (scene_brief or "").strip()
    brief_block = f"\nScene notes (use for unseen sides; never override a visible person or object):\n{brief}\n" if brief else ""
    ref_block = _label_refs(context_angles, context_refs)

    if turn >= 135:
        change = (
            f"Change only the camera. Photorealistic view from {azimuth_deg:.0f}° "
            f"({where}, the {side}). This is the opposite side ({turn:.0f}° from the original). "
            "Show what is BEHIND the subject. Do not reuse the front facade or a mild skew "
            "of the original framing. People and objects keep their world positions; "
            "we now see their other sides."
        )
    elif turn >= 60:
        change = (
            f"Change only the camera. Photorealistic view from {azimuth_deg:.0f}° "
            f"({where}, the {side}). Large turn ({turn:.0f}°). "
            "Objects that faced the camera should appear in profile or from behind. "
            "Do not keep the original framing."
        )
    else:
        change = (
            f"Change only the camera. Photorealistic orbit to {azimuth_deg:.0f}° ({where}). "
            "Same place, same people, same objects. Only the viewpoint moves."
        )

    original_note = ""
    if includes_original and turn >= 60:
        original_note = (
            "If an original front still is included, use it to lock faces, clothes, "
            "small objects, and materials, not as the target composition.\n"
        )

    return (
        "Photorealistic novel-view photograph of the SAME real scene.\n"
        f"{ref_block}\n"
        f"{original_note}"
        f"{brief_block}"
        f"Change:\n{change} Elevation {elevation_deg:.0f}°, same distance.\n"
        f"{PRESERVE_LOCK}\n"
        "Match the original aspect ratio. Keep everything else the same."
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


def write_scene_brief(
    image: Image.Image,
    *,
    provider: str,
    api_key: str,
) -> str:
    """Ask a text/vision model what the front/sides/back of this place should contain."""
    if provider == "gemini":
        return _brief_gemini(image, api_key=api_key)
    if provider == "openai":
        return _brief_openai(image, api_key=api_key)
    raise ValueError(f"Unknown provider: {provider}")


def _brief_gemini(image: Image.Image, *, api_key: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[SCENE_BRIEF_PROMPT, image.convert("RGB")],
    )
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned no scene brief")
    return text


def _brief_openai(image: Image.Image, *, api_key: str) -> str:
    import base64

    from openai import OpenAI

    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=88)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    client = OpenAI(api_key=api_key)
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SCENE_BRIEF_PROMPT},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        max_tokens=500,
    )
    text = (result.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned no scene brief")
    return text


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
    config = types.GenerateContentConfig(response_modalities=["IMAGE"])
    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt, *images],
            config=config,
        )
    except Exception:
        response = client.models.generate_content(
            model=model,
            contents=[prompt, *images],
        )
    parts = list(getattr(response, "parts", None) or [])
    for cand in response.candidates or []:
        parts.extend(getattr(cand.content, "parts", None) or [])
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
        buf.name = f"image_{i + 1}.png"
        buffers.append(buf)
    kwargs: dict[str, Any] = {
        "model": model,
        "image": buffers if len(buffers) > 1 else buffers[0],
        "prompt": prompt,
        "size": _openai_size(images[0]),
        "quality": "high",
    }
    if not str(model).startswith("gpt-image-2"):
        kwargs["input_fidelity"] = "high"
    result = client.images.edit(**kwargs)
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
    context_mode: str = "adaptive",
    order_mode: str = "cardinal",
    original_lock_deg: float = DEFAULT_ORIGINAL_LOCK_DEG,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    context_mode = (context_mode or "adaptive").strip().lower()
    order_mode = (order_mode or "cardinal").strip().lower()
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"Unknown context_mode: {context_mode}")
    if order_mode not in ORDER_MODES:
        raise ValueError(f"Unknown order_mode: {order_mode}")
    original_lock_deg = max(0.0, min(float(original_lock_deg), 180.0))
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
    origin_angle = angles[0]
    scene_brief = ""

    def public_frames() -> list[dict[str, Any]]:
        return [_public_frame(f) for f in generated]

    def emit(message: str) -> None:
        meta = {
            "model": f"{provider}/{model}",
            "kind": "orbit",
            "scene_brief": scene_brief,
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
                original_lock_deg,
            ),
        }
        if on_progress:
            on_progress(message, meta)

    emit("Reading the still — noting front, sides, and back…")
    try:
        scene_brief = write_scene_brief(source, provider=provider, api_key=key)
        emit("Scene brief ready — generating cardinal views next")
    except Exception as exc:  # noqa: BLE001
        scene_brief = ""
        emit(f"Scene brief skipped ({exc}) — continuing from the still")

    for step, angle in enumerate(work_order):
        frame_name = f"{int(round(angle)):03.0f}deg.jpg"
        frame_path = frames_dir / frame_name
        use_original = abs(angle - origin_angle) < 1e-3 or (
            not generated and abs(angle - origin_angle) < 1e-3
        )
        context_angles: list[float] = []
        if use_original and not generated:
            source.save(frame_path, quality=92)
            image = source
            source_kind = "original"
        else:
            refs = select_context(
                angle,
                generated,
                mode=context_mode,
                origin_angle=origin_angle,
                original_lock_deg=original_lock_deg,
            )
            if not refs:
                refs = [{"angle": origin_angle, "image": source, "source": "original"}]
            context_angles = [float(r["angle"]) for r in refs]
            includes_original = any(r.get("source") == "original" for r in refs)
            emit(
                f"Generating {angle:.0f}° ({step + 1}/{len(work_order)}) "
                f"· {facing_name(angle)} · context {', '.join(f'{a:.0f}°' for a in context_angles)}"
            )
            prompt = build_prompt(
                angle,
                elevation_deg,
                extra_prompt,
                context_angles=context_angles,
                context_refs=refs,
                scene_brief=scene_brief,
                origin_angle=origin_angle,
                includes_original=includes_original,
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
        "scene_brief": scene_brief,
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
            original_lock_deg,
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
    context_mode: str = "adaptive",
    order_mode: str = "cardinal",
    original_lock_deg: float = DEFAULT_ORIGINAL_LOCK_DEG,
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
        "original_lock_deg": original_lock_deg,
        "angles": angle_list,
        "n_views": len(angle_list),
    }
