"""
Modal deployment — high-quality whole-space reconstruction via WorldMirror 2.0.

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
HYWORLD_REPO = Path("/opt/HY-World-2.0")

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
    .run_commands(
        f"git clone --depth 1 https://github.com/Tencent-Hunyuan/HY-World-2.0.git {HYWORLD_REPO}"
    )
    .pip_install(
        "diffusers==0.36.0",
        "transformers==5.2.0",
        "accelerate",
        "peft==0.18.1",
        "safetensors",
        "omegaconf",
        "einops",
        "kornia",
        "easydict",
        "scipy==1.14.1",
        "timm==1.0.11",
        "Pillow",
        "imageio[ffmpeg]",
        "decord",
        "imagesize",
        "opencv-python==4.10.0.84",
        "matplotlib==3.10.3",
        "scikit-image==0.25.2",
        "ftfy",
        "regex",
        "trimesh",
        "plyfile",
        "loguru==0.7.3",
        "tqdm",
        "tyro==1.0.8",
        "numpy==1.26.4",
        "huggingface_hub",
        "hf_transfer",
        "fastapi",
        "python-multipart",
        "uvicorn",
        "packaging",
        "ninja",
        "onnxruntime-gpu",
    )
    .run_commands(
        "pip install gsplat --no-build-isolation",
        gpu="A100",
    )
    .pip_install("wheel")
    .run_commands(
        "pip install flash-attn==2.7.4.post1 --no-build-isolation || true",
        gpu="A100",
    )
    .add_local_file(
        "scripts/patch_worldmirror_flash_attn.py",
        remote_path="/tmp/patch_worldmirror_flash_attn.py",
        copy=True,
    )
    .run_commands(
        f"python /tmp/patch_worldmirror_flash_attn.py {HYWORLD_REPO}"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": str(HF_CACHE),
            "PYTHONPATH": str(HYWORLD_REPO),
            "TORCH_CUDA_ARCH_LIST": "8.0;9.0",
            "CUDA_HOME": "/usr/local/cuda",
        }
    )
    .add_local_python_source("scene_gen")
    .add_local_dir("web", remote_path="/app/web")
)

app = modal.App(APP_NAME)
artifact_vol = modal.Volume.from_name("3d-scene-artifacts", create_if_missing=True)
hf_vol = modal.Volume.from_name("3d-scene-hf-cache", create_if_missing=True)
jobs = modal.Dict.from_name("3d-scene-jobs", create_if_missing=True)


def _job_dir(job_id: str) -> Path:
    return ARTIFACT_ROOT / job_id


@app.cls(
    image=image,
    gpu="A100",
    timeout=45 * 60,
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
            raise RuntimeError("CUDA required for WorldMirror inference")
        self.device = torch.device("cuda")
        self.pipeline = None

    def _ensure_pipeline(self, job_id: str, prev: dict) -> None:
        if self.pipeline is not None:
            return

        from scene_gen.worldmirror import load_worldmirror_pipeline

        jobs[job_id] = {
            **prev,
            "status": "running",
            "message": "Loading WorldMirror 2.0 weights (first run can take several minutes)…",
        }
        hf_vol.reload()
        self.pipeline = load_worldmirror_pipeline(enable_bf16=True)
        hf_vol.commit()
        print("WorldMirror 2.0 loaded")

    @modal.method()
    def reconstruct(
        self,
        job_id: str,
        filename: str,
        media_bytes: bytes,
        *,
        target_fps: float = 1.0,
        max_frames: int = 32,
        target_size: int = 952,
    ) -> dict:
        from scene_gen.worldmirror import reconstruct_with_worldmirror

        prev = dict(jobs[job_id]) if job_id in jobs else {"id": job_id}
        jobs[job_id] = {
            **prev,
            "status": "running",
            "message": "Starting GPU worker…",
        }

        out = _job_dir(job_id)
        out.mkdir(parents=True, exist_ok=True)
        media_path = out / filename
        media_path.write_bytes(media_bytes)

        try:
            self._ensure_pipeline(job_id, prev)
            jobs[job_id] = {
                **prev,
                "status": "running",
                "message": "Running WorldMirror 2.0 reconstruction…",
            }
            meta = reconstruct_with_worldmirror(
                media_path,
                out,
                self.pipeline,
                target_size=int(target_size),
                fps=max(int(round(float(target_fps))), 1),
                video_max_frames=int(max_frames),
            )
            artifact_vol.commit()
            kind = "gaussians" if meta.get("has_gaussians") else "points"
            result = {
                **prev,
                "status": "succeeded",
                "message": f"Reconstructed {kind} scene with WorldMirror 2.0",
                "meta": meta,
                "ply_url": f"/api/jobs/{job_id}/gaussians.ply"
                if meta.get("has_gaussians")
                else f"/api/jobs/{job_id}/points.ply",
                "viewer": "splat" if meta.get("has_gaussians") else "points",
            }
            jobs[job_id] = result
            return result
        except Exception as exc:  # noqa: BLE001
            jobs[job_id] = {
                **prev,
                "status": "failed",
                "message": str(exc),
            }
            artifact_vol.commit()
            raise


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={str(ARTIFACT_ROOT): artifact_vol},
)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles

    web = FastAPI(title="3D Scene Generator", version="0.2.0")
    web.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web.get("/api/health")
    def health():
        return {
            "ok": True,
            "model": "tencent/HY-World-2.0 (WorldMirror 2.0)",
            "app": APP_NAME,
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

        target_fps = float(form.get("target_fps") or 1.0)
        max_frames = int(form.get("max_frames") or 32)
        target_size = int(form.get("target_size") or 952)

        job_id = uuid.uuid4().hex[:12]
        await jobs.put.aio(
            job_id,
            {
                "id": job_id,
                "status": "queued",
                "message": (
                    "Queued — waiting for GPU. First WorldMirror start can take "
                    "several minutes while weights load."
                ),
                "filename": filename,
                "model": "worldmirror-2.0",
            },
        )

        await SceneReconstructor().reconstruct.spawn.aio(
            job_id,
            filename,
            data,
            target_fps=target_fps,
            max_frames=max_frames,
            target_size=target_size,
        )
        return {
            "id": job_id,
            "status": "queued",
            "message": "Queued — waiting for GPU",
        }

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
        return _artifact(job_id, "gaussians.ply")

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

    return web
