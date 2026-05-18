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
