# 3D Scene Generator

Upload a **video or photos of a place** → reconstruct the **entire space** → explore it with a free camera in the browser.

This is a whole-space MVP (4dv.ai-style navigation), **not** a single-object mesh generator.

See [PIPELINE.md](./PIPELINE.md) for the full architecture and model roadmap.

## Model

**VGGT-1B** ([facebookresearch/vggt](https://github.com/facebookresearch/vggt)) hosted on **Modal** (A100).

- Video → sampled frames → multi-view geometry  
- Output: colored `scene.ply` + `meta.json`  
- UI: Three.js point-cloud viewer  

## Quick start (Modal)

```bash
# 1) Auth (once)
pip install modal
modal setup

# 2) From this repo
modal deploy modal_app.py
# or for iterative dev:
modal serve modal_app.py
```

Open the printed Modal URL. Upload a short walkthrough video of a room (slow pan works best).

### API

| Method | Path | Notes |
|--------|------|--------|
| `GET` | `/api/health` | Liveness |
| `POST` | `/api/jobs` | multipart: `file`, optional `max_frames`, `target_fps`, `conf_thres` |
| `GET` | `/api/jobs/{id}` | Job status |
| `GET` | `/api/jobs/{id}/scene.ply` | Point cloud |
| `GET` | `/api/jobs/{id}/meta.json` | Timings / counts |

## Local CLI (optional)

Needs a local CUDA GPU and a VGGT checkout at `/opt/vggt` (or pass `--vggt-repo`):

```bash
python scripts/run_local.py path/to/video.mp4 --out outputs/demo
```

## Repo layout

```
PIPELINE.md          # saved product/model pipeline
modal_app.py         # Modal GPU worker + FastAPI + static UI
scene_gen/           # frame extract, VGGT run, PLY export
web/                 # upload UI + Three.js viewer
scripts/run_local.py # offline helper
```

## Tips for good results

- Prefer a **slow video** that orbits / walks the space (5–20s)  
- Avoid heavy motion blur and jump cuts  
- Start with `max_frames=12–16` for faster jobs  
- Single images work, but coverage of the space will be limited  

## Next

1. VGGT poses → InstantSplat / gsplat for denser novel views  
2. Streamable splat format in the viewer  
3. Optional 4D path (FreeTimeGS lineage) once static spaces are solid  
