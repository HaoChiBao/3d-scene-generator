# 3D Scene Generator — Pipeline

Whole-space reconstruction (4dv.ai-style navigable space), **not** single-object meshes.

## Product goal

| | |
|---|---|
| **Input** | Short video (primary) or one/more photos of a place |
| **Output** | Navigable 3D reconstruction of the **entire space** |
| **v1 representation** | Colored point cloud (PLY) from VGGT |
| **v1 interaction** | Free camera in a web viewer |
| **Later** | Gaussian splat export, then optional 4D (time) |

## Chosen model (MVP)

**[VGGT](https://github.com/facebookresearch/vggt)** (`facebook/VGGT-1B`)

- Feed-forward multi-view geometry: cameras, depth, world points
- Works from **1 image**, sparse views, or **video frames**
- Fast enough for an interactive MVP on A100-class GPUs
- Exports into formats that can feed InstantSplat / gsplat later

### Why not TRELLIS / Hunyuan3D?

Those target **object assets** (one mesh). This product needs **full-space** geometry.

### Roadmap after MVP

1. **VGGT → InstantSplat / gsplat** — denser novel-view quality  
2. **WorldMirror / HY-World** — stronger single-image worlds  
3. **FreeTimeGS / EasyVolcap** — true 4D volumetric video (4dv parity)

## Architecture

```
Browser (upload + Three.js viewer)
        │
        ▼
Modal FastAPI  ── jobs Dict + artifacts Volume
        │
        ▼
Modal GPU class (VGGT-1B on A100)
        │
        ├─ extract frames (video @ ~1 fps, capped)
        ├─ load_and_preprocess_images
        ├─ VGGT forward → depth / cameras / world points
        └─ filter by confidence → scene.ply (+ meta.json)
```

## Job flow

1. `POST /api/jobs` — multipart `file` (image or video)  
2. Worker extracts frames → runs VGGT → writes `/artifacts/{job_id}/scene.ply`  
3. `GET /api/jobs/{id}` — `queued | running | succeeded | failed`  
4. Frontend loads PLY and enables free orbit / dolly  

## Hosting

- **Modal**: model weights (HF cache volume), GPU inference, API, static UI  
- **GitHub**: this repo (app + pipeline docs)

## Explicit non-goals for v1

- Single-object GLB generators (TRELLIS, Hunyuan3D, InstantMesh)  
- Multi-camera studio capture / live 4D streaming  
- Photogrammetry-grade metric accuracy  
