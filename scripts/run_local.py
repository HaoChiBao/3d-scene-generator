#!/usr/bin/env python3
"""Local reconstruction helper (requires CUDA + VGGT checkout)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local VGGT scene reconstruction")
    parser.add_argument("media", type=Path, help="Image, video, or image folder")
    parser.add_argument("--out", type=Path, default=Path("outputs/local_job"))
    parser.add_argument("--vggt-repo", type=Path, default=Path("/opt/vggt"))
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--conf-thres", type=float, default=50.0)
    args = parser.parse_args()

    import torch
    from scene_gen.reconstruct import load_vggt_model, reconstruct_media

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model = load_vggt_model(device, vggt_repo=args.vggt_repo)
    meta = reconstruct_media(
        args.media,
        args.out,
        model,
        device=device,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        conf_thres=args.conf_thres,
    )
    print(meta)


if __name__ == "__main__":
    main()
