# 3D Scene Generator — Pipeline

Whole-space reconstruction (4dv.ai-style navigable space), **not** single-object meshes.

## Product goal

| | |
|---|---|
| **Input** | Short video (best) or multiple photos of a place |
| **Output** | Navigable **3D Gaussian Splat** of the entire space |
| **Interaction** | Free camera in a web splat viewer |

## Model choice (high quality)

### Current: **WorldMirror 2.0** (`tencent/HY-World-2.0`)

Part of [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0).

- Feed-forward **multi-view / video → 3DGS**
- Outputs Gaussians + cameras + depth + point cloud
- Open SOTA for whole-space reconstruction (not object meshing)
- Best with a slow orbit / walkthrough video (many views)

### Why not VGGT (previous MVP)?

VGGT is a strong geometry backbone, but the MVP exported a **sparse colored point cloud**. That looks thin and “bad” for product demos. WorldMirror predicts **Gaussian splat attributes** meant for photoreal novel views.

### Alternatives considered

| Model | Role | Notes |
|-------|------|--------|
| **WorldMirror 2.0** | ✅ Primary | Best open reconstruct-from-video/photos → 3DGS |
| AnySplat | Contender | Feed-forward unconstrained GS; good alternative |
| InstantSplat | Contender | Sparse-view GS with short optimization |
| Full HY-World gen (Pano+Stereo) | Later | Single-image *generative* worlds; much heavier (80B+17B) |
| FreeTimeGS / EasyVolcap | Later | True 4D volumetric video (4dv parity) |
| TRELLIS / Hunyuan3D | ❌ | Object assets, wrong product |

## Architecture

```
Browser (upload + Gaussian splat viewer)
        │
        ▼
Modal FastAPI  ── jobs Dict + artifacts Volume
        │
        ▼
Modal GPU class (WorldMirror 2.0 on A100/H100)
        │
        ├─ accept video or image folder
        ├─ WorldMirrorPipeline → gaussians.ply (+ points.ply)
        └─ store under /artifacts/{job_id}/
```

## Quality tips

- Prefer **5–20s video** slowly panning the space  
- Avoid motion blur / jump cuts  
- Single images work, but coverage and novel-view quality drop hard  
- Raise `target_size` (default 952) for sharper reconstruction at higher VRAM/time cost  

## Later upgrades

1. Optional **quality mode**: WorldMirror → short 3DGS refine  
2. **Single-image generative** path via HY-Pano + WorldStereo (full HY-World)  
3. **4D** path for temporal volumetric video  
