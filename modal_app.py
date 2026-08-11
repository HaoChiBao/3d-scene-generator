"""
Modal deployment for whole-space 3D scene generation (VGGT-1B).

Deploy:
  modal setup          # once
  modal deploy modal_app.py

Serve locally against Modal:
  modal serve modal_app.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import modal

APP_NAME = "3d-scene-generator"
ARTIFACT_ROOT = Path("/artifacts")
HF_CACHE = Path("/root/.cache/huggingface")
VGGT_REPO = Path("/opt/vggt")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy<2",
        "Pillow",
        "huggingface_hub",
        "einops",
        "safetensors",
        "opencv-python-headless",
        "trimesh",
        "fastapi",
        "python-multipart",
        "hf_transfer",
        "uvicorn",
    )
    .run_commands(
        f"git clone --depth 1 https://github.com/facebookresearch/vggt.git {VGGT_REPO}"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": str(HF_CACHE),
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
    timeout=30 * 60,
    scaledown_window=120,
    volumes={
        str(ARTIFACT_ROOT): artifact_vol,
        str(HF_CACHE): hf_vol,
    },
)
class SceneReconstructor:
    @modal.enter()
    def load_model(self):
        import torch
        from scene_gen.reconstruct import load_vggt_model

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for VGGT MVP inference")
        self.device = torch.device("cuda")
        self.model = load_vggt_model(self.device, vggt_repo=VGGT_REPO)
        print("VGGT-1B loaded")

    @modal.method()
    def reconstruct(
        self,
        job_id: str,
        filename: str,
        media_bytes: bytes,
        *,
        target_fps: float = 1.0,
        max_frames: int = 24,
        conf_thres: float = 50.0,
    ) -> dict:
        from scene_gen.reconstruct import reconstruct_media

        prev = dict(jobs[job_id]) if job_id in jobs else {"id": job_id}
        jobs[job_id] = {
            **prev,
            "status": "running",
            "message": "Running VGGT reconstruction",
        }

        out = _job_dir(job_id)
        out.mkdir(parents=True, exist_ok=True)
        media_path = out / filename
        media_path.write_bytes(media_bytes)

        try:
            meta = reconstruct_media(
                media_path,
                out,
                self.model,
                device=self.device,
                target_fps=target_fps,
                max_frames=max_frames,
                conf_thres=conf_thres,
            )
            artifact_vol.commit()
            result = {
                **prev,
                "status": "succeeded",
                "message": (
                    f"Reconstructed {meta['num_frames']} frames → "
                    f"{meta['num_points']} points"
                ),
                "meta": meta,
                "ply_url": f"/api/jobs/{job_id}/scene.ply",
            }
            jobs[job_id] = result
            return result
        except Exception as exc:  # noqa: BLE001 - surface to job status
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
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles

    web = FastAPI(title="3D Scene Generator", version="0.1.0")
    web.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web.get("/api/health")
    def health():
        return {"ok": True, "model": "facebook/VGGT-1B", "app": APP_NAME}

    @web.post("/api/jobs")
    async def create_job(
        file: UploadFile = File(...),
        target_fps: float = Form(1.0),
        max_frames: int = Form(24),
        conf_thres: float = Form(50.0),
    ):
        if not file.filename:
            raise HTTPException(400, "Missing filename")

        data = await file.read()
        if not data:
            raise HTTPException(400, "Empty upload")

        # Soft limit ~150MB
        if len(data) > 150 * 1024 * 1024:
            raise HTTPException(400, "File too large (max 150MB)")

        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "message": "Queued for GPU reconstruction",
            "filename": file.filename,
        }

        SceneReconstructor().reconstruct.spawn(
            job_id,
            file.filename,
            data,
            target_fps=float(target_fps),
            max_frames=int(max_frames),
            conf_thres=float(conf_thres),
        )
        return {"id": job_id, "status": "queued"}

    @web.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "Job not found")
        return jobs[job_id]

    @web.get("/api/jobs/{job_id}/scene.ply")
    def get_ply(job_id: str):
        artifact_vol.reload()
        path = _job_dir(job_id) / "scene.ply"
        if not path.exists():
            raise HTTPException(404, "PLY not ready")
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=f"{job_id}_scene.ply",
        )

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
