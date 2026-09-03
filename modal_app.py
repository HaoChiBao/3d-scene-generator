"""
Modal deployment — whole-space reconstruction via recon3d (VGGT + gsplat).

Deploy:
  modal setup
  modal deploy modal_app.py
"""

import json
import uuid
from pathlib import Path

import modal

APP_NAME = "3d-scene-generator"
ARTIFACT_ROOT = Path("/artifacts")
HF_CACHE = Path("/root/.cache/huggingface")
# Keep torch hub weights on the same volume (Modal forbids mounting one volume twice).
TORCH_CACHE = HF_CACHE / "torch"

# Match CUDA toolkit to the torch wheel (cu128) so gsplat can compile.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "build-essential",
        "ninja-build",
        "wget",
        "ca-certificates",
    )
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "Pillow",
        "tqdm",
        "huggingface_hub",
        "hf_transfer",
        "safetensors",
        "einops",
        "plyfile",
        "scipy==1.14.1",
        "scikit-learn==1.5.2",
        "tyro",
        "click",
        "fastapi",
        "python-multipart",
        "uvicorn",
        "packaging",
        "ninja",
        "viser",
    )
    .run_commands(
        "pip install gsplat --no-build-isolation",
        gpu="A100",
    )
    .run_commands(
        "pip install 'git+https://github.com/jashshah999/recon3d.git'",
        "pip install 'git+https://github.com/facebookresearch/vggt.git'",
        # MoGe is optional (metric scale). Fail soft if the API moves.
        "pip install 'git+https://github.com/microsoft/MoGe.git' || true",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": str(HF_CACHE),
            "TORCH_HOME": str(TORCH_CACHE),
            "TORCH_CUDA_ARCH_LIST": "8.0;9.0",
            "CUDA_HOME": "/usr/local/cuda",
            # Avoid HuggingFace xet leaving open log files that break volume commits.
            "HF_HUB_DISABLE_XET": "1",
        }
    )
    .add_local_python_source("scene_gen")
    .add_local_dir("web", remote_path="/app/web")
)

# CPU image for the site + orbit (Gemini / OpenAI). Keep A100 off this path.
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "python-multipart",
        "uvicorn",
        "Pillow",
        "google-genai",
        "openai",
        "httpx",
    )
    .add_local_python_source("scene_gen")
    .add_local_dir("web", remote_path="/app/web")
)

app = modal.App(APP_NAME)
artifact_vol = modal.Volume.from_name("3d-scene-artifacts", create_if_missing=True)
hf_vol = modal.Volume.from_name("3d-scene-hf-cache", create_if_missing=True)
jobs = modal.Dict.from_name("3d-scene-jobs", create_if_missing=True)
# Loaded at deploy from local .env — never sent to the browser.
llm_secret = modal.Secret.from_dotenv(__file__)


def _job_dir(job_id: str) -> Path:
    return ARTIFACT_ROOT / job_id


@app.cls(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    scaledown_window=10 * 60,
    memory=65536,
    volumes={
        str(ARTIFACT_ROOT): artifact_vol,
        str(HF_CACHE): hf_vol,
    },
)
class SceneReconstructor:
    @modal.enter()
    def setup(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for recon3d / VGGT")
        self.device = torch.device("cuda")
        # Touch CUDA once so cold start failures show up early.
        _ = torch.zeros(1, device=self.device)
        print("SceneReconstructor ready (recon3d / VGGT + gsplat)")

    @modal.method()
    def reconstruct(
        self,
        job_id: str,
        filename: str,
        media_bytes: bytes,
        *,
        target_fps: float = 2.0,
        max_frames: int = 48,
        train_steps: int = 7000,
        resize_long_edge: int = 960,
    ) -> dict:
        from scene_gen.recon3d_runner import reconstruct_with_recon3d

        prev = dict(jobs[job_id]) if job_id in jobs else {"id": job_id}

        def set_status(message: str) -> None:
            jobs[job_id] = {
                **prev,
                "status": "running",
                "message": message,
                "model": "recon3d",
            }

        set_status("Starting GPU worker…")
        out = _job_dir(job_id)
        out.mkdir(parents=True, exist_ok=True)
        media_path = out / filename
        media_path.write_bytes(media_bytes)

        try:
            hf_vol.reload()
            set_status("Loading VGGT / running recon3d (first run downloads weights)…")
            meta = reconstruct_with_recon3d(
                media_path,
                out,
                max_frames=int(max_frames),
                target_fps=float(target_fps),
                resize_long_edge=int(resize_long_edge),
                train_steps=int(train_steps),
                metric_align=True,
                use_factor_graph=False,
                on_progress=set_status,
            )
            try:
                hf_vol.commit()
            except Exception as commit_exc:  # noqa: BLE001
                print(f"HF cache commit skipped: {commit_exc}", flush=True)
            artifact_vol.commit()
            kind = "splat" if meta.get("has_gaussians") else "points"
            result = {
                **prev,
                "status": "succeeded",
                "message": (
                    f"Reconstructed Gaussian scene with recon3d "
                    f"({meta.get('n_gaussians') or '?'} Gaussians)"
                ),
                "meta": meta,
                "ply_url": f"/api/jobs/{job_id}/gaussians.ply"
                if meta.get("has_gaussians")
                else f"/api/jobs/{job_id}/points.ply",
                "viewer": kind,
                "model": "recon3d",
            }
            jobs[job_id] = result
            return result
        except Exception as exc:  # noqa: BLE001
            jobs[job_id] = {
                **prev,
                "status": "failed",
                "message": str(exc),
                "model": "recon3d",
            }
            artifact_vol.commit()
            raise


@app.function(
    image=web_image,
    timeout=45 * 60,
    memory=4096,
    secrets=[llm_secret],
    volumes={str(ARTIFACT_ROOT): artifact_vol},
)
def run_orbit(
    job_id: str,
    filename: str,
    media_bytes: bytes,
    params: dict,
) -> dict:
    from scene_gen.orbit_views import reconstruct_orbit

    prev = dict(jobs[job_id]) if job_id in jobs else {"id": job_id, "kind": "orbit"}

    def set_status(message: str, meta: dict | None = None) -> None:
        jobs[job_id] = {
            **prev,
            "status": "running",
            "message": message,
            "kind": "orbit",
            "model": params.get("model") or params.get("provider"),
            "meta": meta or prev.get("meta"),
        }
        try:
            artifact_vol.commit()
        except Exception as commit_exc:  # noqa: BLE001
            print(f"orbit artifact commit skipped: {commit_exc}", flush=True)

    set_status("Starting orbit generation…")
    out = _job_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    media_path = out / filename
    media_path.write_bytes(media_bytes)

    try:
        meta = reconstruct_orbit(
            media_path,
            out,
            increment_deg=float(params["increment_deg"]),
            start_deg=float(params["start_deg"]),
            end_deg=float(params["end_deg"]),
            elevation_deg=float(params["elevation_deg"]),
            radius=float(params["radius"]),
            fov_deg=float(params["fov_deg"]),
            provider=str(params["provider"]),
            model=params.get("model"),
            extra_prompt=str(params.get("extra_prompt") or ""),
            api_key=params.get("api_key"),
            context_mode=str(params.get("context_mode") or "nearest"),
            order_mode=str(params.get("order_mode") or "bidirectional"),
            on_progress=set_status,
        )
        artifact_vol.commit()
        result = {
            **prev,
            "status": "succeeded",
            "message": f"Generated {meta.get('n_views')} orbit views",
            "kind": "orbit",
            "model": meta.get("model"),
            "meta": meta,
        }
        jobs[job_id] = result
        return result
    except Exception as exc:  # noqa: BLE001
        jobs[job_id] = {
            **prev,
            "status": "failed",
            "message": str(exc),
            "kind": "orbit",
        }
        artifact_vol.commit()
        raise


@app.function(
    image=web_image,
    timeout=60 * 60,
    secrets=[llm_secret],
    volumes={str(ARTIFACT_ROOT): artifact_vol},
)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles

    web = FastAPI(title="3D Scene Generator", version="0.3.0")
    web.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web.get("/api/health")
    def health():
        import os

        providers = {
            "gemini": bool(
                os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            ),
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
        }
        return {
            "ok": True,
            "app": APP_NAME,
            "docs": "METHODS.md",
            "providers": providers,
            "methods": [
                {
                    "id": "reconstruct",
                    "path": "/",
                    "model": "recon3d (VGGT + gsplat)",
                },
                {
                    "id": "orbit",
                    "path": "/orbit",
                    "model": "gemini-2.5-flash-image / gpt-image-1",
                },
            ],
        }

    @web.post("/api/jobs")
    async def create_job(request: Request):
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(400, "Missing file")

        filename = getattr(upload, "filename", None) or "upload.bin"
        data = await upload.read()
        if not data:
            raise HTTPException(400, "Empty upload")
        if len(data) > 200 * 1024 * 1024:
            raise HTTPException(400, "File too large (max 200MB)")

        target_fps = float(form.get("target_fps") or 2.0)
        max_frames = int(form.get("max_frames") or 48)
        train_steps = int(form.get("train_steps") or 7000)
        resize_long_edge = int(form.get("resize") or form.get("target_size") or 960)

        # Keep jobs within A100-friendly bounds for the web UI.
        max_frames = max(3, min(max_frames, 80))
        train_steps = max(1000, min(train_steps, 15000))
        target_fps = max(0.2, min(target_fps, 5.0))

        job_id = uuid.uuid4().hex[:12]
        await jobs.put.aio(
            job_id,
            {
                "id": job_id,
                "status": "queued",
                "message": (
                    "Queued — waiting for GPU. First run downloads VGGT weights; "
                    "full reconstruct usually takes several minutes."
                ),
                "filename": filename,
                "model": "recon3d",
            },
        )

        await SceneReconstructor().reconstruct.spawn.aio(
            job_id,
            filename,
            data,
            target_fps=target_fps,
            max_frames=max_frames,
            train_steps=train_steps,
            resize_long_edge=resize_long_edge,
        )
        return {
            "id": job_id,
            "status": "queued",
            "message": "Queued — waiting for GPU",
        }

    @web.post("/api/orbit/jobs")
    async def create_orbit_job(request: Request):
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(400, "Missing file")

        filename = getattr(upload, "filename", None) or "source.jpg"
        data = await upload.read()
        if not data:
            raise HTTPException(400, "Empty upload")
        if len(data) > 25 * 1024 * 1024:
            raise HTTPException(400, "File too large (max 25MB)")

        increment_deg = float(form.get("increment_deg") or 10)
        start_deg = float(form.get("start_deg") or 0)
        end_deg = float(form.get("end_deg") or 360)
        elevation_deg = float(form.get("elevation_deg") or 12)
        radius = float(form.get("radius") or 2.4)
        fov_deg = float(form.get("fov_deg") or 45)
        provider = str(form.get("provider") or "gemini").lower()
        model = str(form.get("model") or "").strip() or None
        extra_prompt = str(form.get("extra_prompt") or "")
        api_key = str(form.get("api_key") or "").strip() or None
        context_mode = str(form.get("context_mode") or "nearest").strip().lower()
        order_mode = str(form.get("order_mode") or "bidirectional").strip().lower()
        if context_mode not in ("nearest", "previous", "original"):
            raise HTTPException(400, "context_mode must be nearest, previous, or original")
        if order_mode not in ("bidirectional", "sequential"):
            raise HTTPException(400, "order_mode must be bidirectional or sequential")

        increment_deg = max(5, min(increment_deg, 90))
        start_deg = max(0, min(start_deg, 350))
        end_deg = max(start_deg + increment_deg, min(end_deg, 360))
        elevation_deg = max(0, min(elevation_deg, 75))
        radius = max(0.6, min(radius, 8))
        fov_deg = max(20, min(fov_deg, 90))
        if provider not in ("gemini", "openai"):
            raise HTTPException(400, "provider must be gemini or openai")

        job_id = uuid.uuid4().hex[:12]
        await jobs.put.aio(
            job_id,
            {
                "id": job_id,
                "status": "queued",
                "kind": "orbit",
                "message": "Queued — generating novel views",
                "filename": filename,
                "model": model or provider,
            },
        )
        await run_orbit.spawn.aio(
            job_id,
            filename,
            data,
            {
                "increment_deg": increment_deg,
                "start_deg": start_deg,
                "end_deg": end_deg,
                "elevation_deg": elevation_deg,
                "radius": radius,
                "fov_deg": fov_deg,
                "provider": provider,
                "model": model,
                "extra_prompt": extra_prompt,
                "api_key": api_key,
                "context_mode": context_mode,
                "order_mode": order_mode,
            },
        )
        return {"id": job_id, "status": "queued", "kind": "orbit"}

    @web.get("/api/orbit/jobs/{job_id}")
    async def get_orbit_job(job_id: str):
        data = await jobs.get.aio(job_id)
        if data is None:
            raise HTTPException(404, "Job not found")
        return data

    @web.get("/api/orbit/jobs/{job_id}/frames/{index}")
    def get_orbit_frame(job_id: str, index: int):
        artifact_vol.reload()
        job = _job_dir(job_id)
        frames = []
        meta_path = job / "meta.json"
        if meta_path.exists():
            frames = json.loads(meta_path.read_text(encoding="utf-8")).get("frames") or []
        else:
            live = jobs.get(job_id) or {}
            frames = (live.get("meta") or {}).get("frames") or []
        if 0 <= index < len(frames):
            path = job / frames[index]["file"]
            if path.exists():
                return FileResponse(path, media_type="image/jpeg")
        matches = sorted((job / "frames").glob("*deg.jpg"))
        if 0 <= index < len(matches):
            return FileResponse(matches[index], media_type="image/jpeg")
        raise HTTPException(404, "Frame not ready")

    @web.get("/api/jobs/{job_id}")
    async def get_job(job_id: str):
        data = await jobs.get.aio(job_id)
        if data is None:
            raise HTTPException(404, "Job not found")
        return data

    def _artifact(job_id: str, name: str) -> FileResponse:
        artifact_vol.reload()
        path = _job_dir(job_id) / name
        if not path.exists():
            raise HTTPException(404, f"{name} not ready")
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=f"{job_id}_{name}",
        )

    @web.get("/api/jobs/{job_id}/gaussians.ply")
    def get_gaussians(job_id: str):
        artifact_vol.reload()
        job = _job_dir(job_id)
        for name in ("gaussians.ply", "scene.ply"):
            path = job / name
            if path.exists():
                return FileResponse(
                    path,
                    media_type="application/octet-stream",
                    filename=f"{job_id}_{name}",
                )
        raise HTTPException(404, "gaussians.ply not ready")

    @web.get("/api/jobs/{job_id}/points.ply")
    def get_points(job_id: str):
        return _artifact(job_id, "points.ply")

    @web.get("/api/jobs/{job_id}/scene.ply")
    def get_scene(job_id: str):
        artifact_vol.reload()
        job = _job_dir(job_id)
        for name in ("gaussians.ply", "scene.ply", "points.ply"):
            path = job / name
            if path.exists():
                return FileResponse(
                    path,
                    media_type="application/octet-stream",
                    filename=f"{job_id}_{name}",
                )
        raise HTTPException(404, "Scene not ready")

    @web.get("/api/jobs/{job_id}/meta.json")
    def get_meta(job_id: str):
        artifact_vol.reload()
        path = _job_dir(job_id) / "meta.json"
        if not path.exists():
            raise HTTPException(404, "Meta not ready")
        return json.loads(path.read_text(encoding="utf-8"))

    static_dir = Path("/app/web")
    if static_dir.exists():
        web.mount("/assets", StaticFiles(directory=static_dir), name="assets")

        @web.get("/")
        def index():
            return HTMLResponse((static_dir / "index.html").read_text(encoding="utf-8"))

        @web.get("/orbit")
        def orbit():
            return HTMLResponse((static_dir / "orbit.html").read_text(encoding="utf-8"))

    return web
