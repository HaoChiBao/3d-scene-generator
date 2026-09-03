# 3D Scene Generator

Upload a **video or photos of a place** → reconstruct the **entire space** as a **3D Gaussian splat** → explore it in the browser.

Whole-space product (4dv.ai-style navigation), **not** a single-object mesh generator.

- Methods (reconstruct, orbit, video 4D): [METHODS.md](./METHODS.md)  
- Architecture: [PIPELINE.md](./PIPELINE.md)  
- Reconstruct model research: [MODELS.md](./MODELS.md)

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
| `GET` | `/` | Reconstruct (video → 3DGS) |
| `GET` | `/orbit` | Orbit views (still → image-model novel views) |
| `GET` | `/4d` | Video 4D (video → time-sliced 4D-GS) |
| `GET` | `/api/health` | Liveness + method list |
| `POST` | `/api/jobs` | Reconstruct job |
| `GET` | `/api/jobs/{id}` | Reconstruct status |
| `GET` | `/api/jobs/{id}/gaussians.ply` | 3DGS asset |
| `POST` | `/api/orbit/jobs` | Orbit job (`increment_deg`, `elevation_deg`, provider, …) |
| `GET` | `/api/orbit/jobs/{id}` | Orbit status + frames so far |
| `GET` | `/api/orbit/jobs/{id}/frames/{i}` | Generated still |
| `POST` | `/api/4d/jobs` | Video 4D job (`n_times`, frames, steps, …) |
| `GET` | `/api/4d/jobs/{id}` | Video 4D status + timeline |
| `GET` | `/api/4d/jobs/{id}/times/{i}` | Gaussian PLY at time *i* |

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
