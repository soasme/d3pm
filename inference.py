#!/usr/bin/env python3
"""Predict the next walk-cycle frame(s) from a sprite GIF using a trained D3PM model."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from d3pm_runner import D3PM
from d3pm_runner_spritesheet import (
    DIRECTION_MAP,
    SpriteX0Model,
    generate_frame,
    generate_frames,
    save_as_gif,
)


def _infer_direction(path: str):
    stem = Path(path).stem
    for code, idx in DIRECTION_MAP.items():
        if f"_{code}" in stem:
            return idx, code
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input frame1 GIF")
    parser.add_argument("output", help="Output GIF path for predicted frame(s)")
    parser.add_argument(
        "--ckpt",
        default="ckpt/spritesheet_step1000.pt",
        help="Model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--direction",
        choices=list(DIRECTION_MAP),
        default=None,
        help="Walk direction (auto-inferred from filename when omitted)",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=1,
        help="Number of frames to predict (1 = just frame2, >1 = autoregressive rollout)",
    )
    parser.add_argument(
        "--palette-size",
        type=int,
        default=246,
        help="Palette classes used during training",
    )
    parser.add_argument(
        "--n-T",
        type=int,
        default=1000,
        help="Diffusion timesteps (must match training)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    N = args.palette_size

    if args.direction is not None:
        direction = DIRECTION_MAP[args.direction]
        dir_name = args.direction
    else:
        direction, dir_name = _infer_direction(args.input)
        if direction is None:
            parser.error(
                f"Cannot infer direction from '{Path(args.input).name}'. "
                "Pass --direction {fr,bk,lf,rt} explicitly."
            )

    print(f"Input:     {args.input}")
    print(f"Direction: {dir_name} ({direction})")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device:    {device}")

    x0_model = SpriteX0Model(n_channel=2, N=N)
    x0_model.load_state_dict(
        torch.load(args.ckpt, map_location=device, weights_only=True)
    )
    d3pm = D3PM(x0_model, n_T=args.n_T, num_classes=N, hybrid_loss_coeff=0.0).to(device)
    d3pm.eval()

    src = Image.open(args.input)
    palette = src.getpalette()
    frame1 = torch.tensor(np.array(src), dtype=torch.long)
    H, W = frame1.shape
    print(f"Frame size: {W}x{H}")

    out = Path(args.output)
    with torch.no_grad():
        if args.n_frames == 1:
            print("Generating frame 2 …")
            frame2 = generate_frame(d3pm, {0: frame1}, direction=direction, device=device)
            save_as_gif(frame2, palette, str(out))
            print(f"Saved: {out}")
        else:
            print(f"Generating {args.n_frames} frames autoregressively …")
            frames = generate_frames(
                d3pm, frame1, direction=direction, device=device,
                n_frames=args.n_frames + 1,
            )
            for i, f in enumerate(frames[1:], start=2):
                out_path = out.parent / f"{out.stem}_frame{i}{out.suffix}"
                save_as_gif(f, palette, str(out_path))
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
