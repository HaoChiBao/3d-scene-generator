# 3D Scene Generator

Upload a **video or photos of a place** → reconstruct the **entire space** as a **3D Gaussian splat** → explore it in the browser.

This is a whole-space product (4dv.ai-style navigation), **not** a single-object mesh generator.

See [PIPELINE.md](./PIPELINE.md) for architecture and model rationale.

## Model (high quality)

**WorldMirror 2.0** from [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) (`tencent/HY-World-2.0`)

- Multi-view / video → cameras, depth, **3DGS**
- Hosted on Modal (A100-80GB)
- Best results from a slow walkthrough / orbit video

Previous VGGT point-cloud MVP was replaced because quality was too low for demos.

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
| `POST` | `/api/jobs` | multipart: `file`, optional `max_frames`, `target_fps`, `target_size` |
| `GET` | `/api/jobs/{id}` | Job status |
| `GET` | `/api/jobs/{id}/gaussians.ply` | Primary 3DGS asset |
| `GET` | `/api/jobs/{id}/points.ply` | Fallback point cloud |
| `GET` | `/api/jobs/{id}/meta.json` | Timings / flags |

## Tips

- Prefer a **slow 5–20s video** of the space  
- First GPU cold start downloads large weights — wait a few minutes  
- Single images work but novel views will be weaker  

## Repo layout

```
PIPELINE.md          # model + architecture decisions
modal_app.py         # Modal GPU worker + FastAPI + UI
scene_gen/           # WorldMirror integration
web/                 # upload UI + Gaussian splat viewer
```
