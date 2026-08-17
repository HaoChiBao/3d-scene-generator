# 3D Scene Generator

Upload a **video or photos of a place** → reconstruct the **entire space** as a **3D Gaussian splat** → explore it in the browser.

Whole-space product (4dv.ai-style navigation), **not** a single-object mesh generator.

- Architecture: [PIPELINE.md](./PIPELINE.md)  
- Model research (papers, download, why this stack): [MODELS.md](./MODELS.md)

## Model

**recon3d** — [VGGT](https://github.com/facebookresearch/vggt) camera poses + per-scene [gsplat](https://github.com/nerfstudio-project/gsplat) training ([recon3d](https://github.com/jashshah999/recon3d), MIT).

Feed-forward-only WorldMirror was replaced because novel-view quality was not good enough for demos. Per-scene Gaussian optimization is what makes navigable scenes look right.

## Live

- App: https://jamesyang663--3d-scene-generator-api.modal.run  
- Dashboard: https://modal.com/apps/jamesyang663/main/deployed/3d-scene-generator  

## Quick start (Modal)

```bash
pip install modal
modal setup
modal deploy modal_app.py
```

## API

| Method | Path | Notes |
|--------|------|--------|
| `GET` | `/api/health` | Liveness + model name |
| `POST` | `/api/jobs` | multipart: `file`, optional `max_frames`, `target_fps`, `train_steps`, `resize` |
| `GET` | `/api/jobs/{id}` | Job status |
| `GET` | `/api/jobs/{id}/gaussians.ply` | Primary 3DGS asset |
| `GET` | `/api/jobs/{id}/meta.json` | Timings / flags |

## Tips

- Prefer a **slow walkthrough** with camera motion (parallax)  
- Expect **several minutes** per scene (pose + ~7k gsplat steps)  
- First GPU cold start downloads VGGT weights  

## Repo layout

```
MODELS.md            # research dive + papers + install
PIPELINE.md          # architecture decisions
modal_app.py         # Modal GPU worker + FastAPI + UI
scene_gen/           # recon3d integration
web/                 # upload UI + Gaussian splat viewer
```
