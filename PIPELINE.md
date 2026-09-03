# 3D Scene Generator — Pipeline

Whole-space reconstruction (4dv.ai-style navigable space), **not** single-object meshes.

Full model comparison, papers, and install notes: **[MODELS.md](./MODELS.md)**.

## Product goal

| | |
|---|---|
| **Input** | Short video (best) or multiple photos of a place |
| **Output** | Navigable **3D Gaussian Splat** of the entire space |
| **Interaction** | Free camera in a web splat viewer |

## Model choice

### Current: **recon3d** (VGGT → gsplat)

Open pipeline: [jashshah999/recon3d](https://github.com/jashshah999/recon3d) (MIT).

1. Sample frames from video  
2. **VGGT-1B** estimates cameras + dense points (CVPR 2025 Best Paper)  
3. Optional **MoGe-2** metric scale  
4. Per-scene **gsplat** training → standard `gaussians.ply`  

This is the quality path used by real products: **optimize Gaussians on your scene**, don’t only run a feed-forward net.

### Why not WorldMirror 2.0?

Tried in production. Feed-forward 3DGS is fast but novel views looked wrong for casual room video. See MODELS.md.

### Alternatives

| Model | Role | Notes |
|-------|------|--------|
| **recon3d** | ✅ Primary | Best open “video → good splat” stack |
| AnySplat | Fast feed-forward | Good research alt; weaker novel views |
| LongSplat | Long casual video | Heavier; future quality mode |
| Full HY-World gen | Generative single image | Later / heavy |
| TRELLIS / Hunyuan3D | ❌ | Object assets, wrong product |

## Methods

See **[METHODS.md](./METHODS.md)**. Reconstruct (`/`) is this pipeline. Orbit views (`/orbit`) is a separate image-model experiment.

## Architecture

```
Browser
  /           reconstruct UI + splat viewer
  /orbit      still + 3D frustum diagram + generated views
        │
        ▼
Modal FastAPI (CPU image)
        │
        ├─ POST /api/jobs        → A100 SceneReconstructor (recon3d)
        └─ POST /api/orbit/jobs  → CPU run_orbit (Gemini / OpenAI)
           context: original + nearest (default) | + previous | original only
           order: bidirectional (default) | sequential
```

## Quality tips

- Prefer **5–30s video** walking / orbiting the space (parallax matters)  
- Default ~**48 frames** @ **2 fps**, **7000** train steps  
- Avoid motion blur and jump cuts  
- Need **≥3 frames**; single stills are not enough for this pipeline  

## Later upgrades

1. Enable GTSAM factor-graph for 80–300 frame walks  
2. Optional LongSplat / post-opt quality mode  
3. True **4D** path for temporal volumetric video  
