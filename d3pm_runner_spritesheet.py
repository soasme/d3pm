# d3pm_runner_spritesheet.py
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

from d3pm_runner import D3PM, DummyX0Model

DIRECTION_MAP = {"fr": 0, "bk": 1, "lf": 2, "rt": 3}
_PAIR_RE = re.compile(r"^([a-z0-9]+)_(fr|bk|lf|rt)1\.gif$")


class SpritesheetDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.pairs = []  # list of (Path frame1, Path frame2, int direction)

        for f in sorted(self.data_dir.iterdir()):
            m = _PAIR_RE.match(f.name)
            if not m:
                continue
            char, direction = m.group(1), m.group(2)
            frame2 = self.data_dir / f"{char}_{direction}2.gif"
            if frame2.exists():
                self.pairs.append((f, frame2, DIRECTION_MAP[direction]))

        if not self.pairs:
            raise ValueError(
                f"No valid GIF pairs found in {self.data_dir!r}. "
                "Files must follow the pattern <char>_<dir>1.gif / <char>_<dir>2.gif "
                "where <dir> is one of: fr, bk, lf, rt."
            )

        max_idx = 0
        for path1, path2, _ in self.pairs:
            for path in (path1, path2):
                arr = np.array(Image.open(path))
                max_idx = max(max_idx, int(arr.max()))
        self.palette_size = max_idx + 1

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, direction = self.pairs[idx]
        arr1 = np.array(Image.open(path1))
        arr2 = np.array(Image.open(path2))
        x = torch.tensor(np.stack([arr1, arr2], axis=0), dtype=torch.long)
        return x, direction, str(path1)


class SpriteX0Model(DummyX0Model):
    """DummyX0Model with 4-class direction conditioning instead of 10-class MNIST."""

    def __init__(self, n_channel: int, N: int):
        super().__init__(n_channel, N)
        self.cond_embedding_1 = nn.Embedding(4, 16)
        self.cond_embedding_2 = nn.Embedding(4, 32)
        self.cond_embedding_3 = nn.Embedding(4, 64)
        self.cond_embedding_4 = nn.Embedding(4, 512)
        self.cond_embedding_5 = nn.Embedding(4, 512)
        self.cond_embedding_6 = nn.Embedding(4, 64)


def generate_frame(
    d3pm,
    anchor_frames,
    direction,
    device,
    total_frames=2,
    predict_frame=1,
):
    """
    Generate one frame via reverse diffusion, clamping all known anchor frames
    after every p_sample step.

    Args:
        d3pm: trained D3PM instance
        anchor_frames: dict {frame_idx: LongTensor [H, W]} of known frames
        direction: int, 0=fr 1=bk 2=lf 3=rt
        device: torch device string or object
        total_frames: total frames in the clip (default 2)
        predict_frame: index of the frame to generate (default 1)

    Returns:
        LongTensor [H, W] — generated frame palette indices
    """
    if not anchor_frames:
        raise ValueError("anchor_frames must contain at least one known frame")
    if not (0 <= predict_frame < total_frames):
        raise ValueError(f"predict_frame={predict_frame} out of range [0, {total_frames})")
    if predict_frame in anchor_frames:
        raise ValueError(f"predict_frame={predict_frame} is already an anchor — nothing to generate")

    N = d3pm.num_classses
    H, W = next(iter(anchor_frames.values())).shape
    x = torch.randint(0, N, (1, total_frames, H, W), device=device)
    for idx, frame in anchor_frames.items():
        x[:, idx] = frame.to(device)

    cond = torch.tensor([direction], device=device)

    for t in reversed(range(1, d3pm.n_T)):
        t_tensor = torch.tensor([t], device=device)
        noise = torch.rand((*x.shape, N), device=device)
        x = d3pm.p_sample(x, t_tensor, cond, noise)
        for idx, frame in anchor_frames.items():
            x[:, idx] = frame.to(device)

    return x[0, predict_frame]


def generate_frames(d3pm, frame1, direction, device, n_frames):
    """
    Autoregressively generate n_frames total (including frame1) by chaining
    generate_frame calls. Each generated frame becomes the anchor for the next.

    Args:
        d3pm: trained D3PM instance
        frame1: LongTensor [H, W] — the known first frame
        direction: int, 0=fr 1=bk 2=lf 3=rt
        device: torch device string or object
        n_frames: total number of frames to return (including frame1)

    Returns:
        list of n_frames LongTensors, each [H, W]

    Note:
        Each call uses only the immediately preceding frame as anchor (position 0).
        This matches the 2-frame training setup. For longer sequences the model
        accumulates drift since earlier frames are not re-anchored.
    """
    frames = [frame1]
    for _ in range(n_frames - 1):
        next_frame = generate_frame(d3pm, {0: frames[-1]}, direction, device)
        frames.append(next_frame)
    return frames


def save_as_gif(pixel_indices, palette, path):
    """
    Save a palette-indexed image as a GIF file.

    Args:
        pixel_indices: LongTensor or uint8 ndarray [H, W] of palette indices
        palette: flat list of 768 ints (256 RGB entries) from Image.getpalette()
        path: output file path string
    """
    if isinstance(pixel_indices, torch.Tensor):
        pixel_indices = pixel_indices.cpu().numpy()
    pixel_indices = pixel_indices.astype(np.uint8)
    img = Image.fromarray(pixel_indices, mode="P")
    img.putpalette(palette)
    img.save(str(path))
