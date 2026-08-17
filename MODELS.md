# Model research — whole-space video → navigable 3D

Goal: phone video / multi-view photos of a **place** → navigable **3D Gaussian splat** (4dv.ai-style free camera). Not single-object meshes.

## Verdict

**Primary: [recon3d](https://github.com/jashshah999/recon3d)** — VGGT poses + per-scene **gsplat** training.

Feed-forward-only models (WorldMirror, AnySplat) are fast demos but fail the product bar for casual room video: soft geometry, broken novel views, fragile on single images. High-quality products (Luma, Polycam, classic 3DGS) all **optimize Gaussians per scene** after getting cameras. recon3d is the best open, one-command version of that stack without COLMAP.

| | Feed-forward (WorldMirror / AnySplat) | recon3d (VGGT → gsplat) |
|---|---|---|
| Latency | Seconds–1 min | ~3–15 min on A100 |
| Novel-view quality | Often soft / wrong | Matches real captures when video has parallax |
| Single image | Weak / broken | Needs ≥3 frames with motion |
| Deploy complexity | Heavy custom deps | pip + VGGT + gsplat |

## Contenders compared

| Model | Paper / venue | Open? | Download | Fit for us |
|-------|---------------|-------|----------|------------|
| **recon3d** | Software pipeline (2026); builds on VGGT + 3DGS | MIT | `pip install` + VGGT from GitHub; weights via HF | **Winner** — video → standard `scene.ply` |
| **VGGT-1B** | [Wang et al., CVPR 2025 Best Paper](https://arxiv.org/abs/2503.11651) | Code MIT; checkpoint Meta Research / CC-BY-NC | [`facebook/VGGT-1B`](https://huggingface.co/facebook/VGGT-1B) | Pose + dense points; not photoreal alone |
| **gsplat / 3DGS** | [Kerbl et al., SIGGRAPH 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/); [gsplat](https://github.com/nerfstudio-project/gsplat) | Apache-2.0 (gsplat) | `pip install gsplat` | Training / rasterization engine |
| **WorldMirror 2.0** | HY-World 2.0 (Tencent) | Open weights | [`tencent/HY-World-2.0`](https://huggingface.co/tencent/HY-World-2.0) | Tried in production — quality not good enough |
| **AnySplat** | [Jiang et al., SIGGRAPH Asia 2025 / TOG](https://arxiv.org/abs/2505.23716) | Yes | [`lhjiang/anysplat`](https://huggingface.co/lhjiang/anysplat); [GitHub](https://github.com/InternRobotics/AnySplat) | Strong feed-forward GS; still no per-scene fit |
| **LongSplat** | [ICCV 2025](https://github.com/NVlabs/LongSplat) | NVlabs | Repo + convert to 3DGS PLY | Excellent for long casual video; heavy install |
| **StreamSplat** | [ICLR 2026](https://streamsplat3d.github.io/) | Research | [DSL-Lab/StreamSplat](https://github.com/DSL-Lab/StreamSplat) | Online / dynamic; more research than product |
| **DepthSplat** | CVPR 2025 | MIT | [autonomousvision/depthsplat](https://github.com/autonomousvision/depthsplat) | Needs multi-view depth; good backend option |
| TRELLIS / Hunyuan3D | — | Open | HF | **Wrong product** (objects, not spaces) |

## How recon3d works

```
Video / images
    → frame extract (blur filter, resize)
    → VGGT poses + dense points  (chunked + optional GTSAM for long clips)
    → optional MoGe-2 metric scale
    → gsplat train (~7k steps)
    → scene.ply / scene.splat
```

### Download / install

```bash
git clone https://github.com/jashshah999/recon3d.git && cd recon3d
pip install -e ".[all]"
pip install git+https://github.com/facebookresearch/vggt.git
pip install git+https://github.com/microsoft/MoGe.git   # optional metric
```

Weights auto-download on first run:

| Weights | Hugging Face | Role |
|---------|--------------|------|
| VGGT-1B | `facebook/VGGT-1B` | Cameras + geometry |
| MoGe-2 ViT-L | `Ruicheng/moge-2-vitl` | Metric alignment (optional) |

### Use

```bash
recon3d run my_video.mp4
recon3d run ./photos/ --max-frames 80 --steps 7000
```

Python:

```python
from recon3d.pipeline import reconstruct, PipelineConfig
from recon3d.gaussian_train import TrainConfig

ply = reconstruct(
    "video.mp4",
    "out/",
    PipelineConfig(
        max_frames=48,
        target_fps=2.0,
        launch_viewer=False,
        train_config=TrainConfig(max_steps=7000),
    ),
)
```

### Papers to read

1. **3D Gaussian Splatting** — Kerbl et al., SIGGRAPH 2023  
2. **VGGT** — Wang et al., CVPR 2025 Best Paper ([arXiv:2503.11651](https://arxiv.org/abs/2503.11651))  
3. **AnySplat** — Jiang et al., TOG 2025 ([arXiv:2505.23716](https://arxiv.org/abs/2505.23716)) — best pure feed-forward alternative  
4. **LongSplat** — ICCV 2025 — if we need longer walks later  
5. **MoGe-2** — Microsoft metric depth (optional scale)

## Capture tips (product)

- Slow walk / orbit with **parallax** (move through the space, don’t only pan)
- Prefer **5–30s video**, ~1–2 fps sampled, **24–80 frames**
- Avoid motion blur, jump cuts, and heavy dynamics (people walking)
- Single photos: upload a **folder / multi-image** set with overlapping views; ≥3 required

## Why we left WorldMirror

WorldMirror 2.0 was a reasonable SOTA feed-forward pick and did produce Gaussians, but end-to-end demos looked wrong for navigable rooms. recon3d trades minutes of GPU for Gaussians that actually match the input views — the bar for this product.
