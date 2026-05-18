import io
import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader

from d3pm_runner_spritesheet import SpritesheetDataset


def make_gif_bytes(h=32, w=32, max_index=7):
    """Create a minimal single-frame GIF with palette indices 0..max_index."""
    arr = np.zeros((h, w), dtype=np.uint8)
    # Fill first row with all indices 0..max_index so GIF preserves them all
    for i in range(max_index + 1):
        arr[0, i % w] = i
    img = Image.fromarray(arr, mode="P")
    img.putpalette(bytes(range(256)) * 3)  # 256 RGB entries, all values 0-255
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def data_dir(tmp_path):
    gif = make_gif_bytes(max_index=7)  # palette_size should be 8

    # Valid pairs
    (tmp_path / "amg1_fr1.gif").write_bytes(gif)
    (tmp_path / "amg1_fr2.gif").write_bytes(gif)
    (tmp_path / "amg1_bk1.gif").write_bytes(gif)
    (tmp_path / "amg1_bk2.gif").write_bytes(gif)

    # Should be skipped: no direction suffix
    (tmp_path / "house1.gif").write_bytes(gif)
    # Should be skipped: wrong extension
    (tmp_path / "back2.jpg").write_bytes(b"not-a-gif")
    # Should be skipped: frame2 is missing
    (tmp_path / "amg2_lf1.gif").write_bytes(gif)

    return tmp_path


def test_dataset_discovers_pairs(data_dir):
    ds = SpritesheetDataset(str(data_dir))
    assert len(ds) == 2  # amg1_fr + amg1_bk


def test_dataset_skips_non_matching(data_dir):
    ds = SpritesheetDataset(str(data_dir))
    names = [Path(str(p1)).name for p1, _, _ in ds.pairs]  # (path1, path2, direction) tuples
    assert not any("house" in n for n in names)
    assert not any("lf" in n for n in names)


def test_dataset_getitem_shape(data_dir):
    ds = SpritesheetDataset(str(data_dir))
    x, direction, path_frame1 = ds[0]
    assert x.shape == (2, 32, 32)
    assert x.dtype == torch.long
    assert direction in (0, 1, 2, 3)
    assert path_frame1.endswith(".gif")


def test_dataset_palette_size(data_dir):
    ds = SpritesheetDataset(str(data_dir))
    assert ds.palette_size == 8  # max_index=7 → palette_size = 7+1 = 8


def test_dataset_direction_encoding(data_dir):
    ds = SpritesheetDataset(str(data_dir))
    directions = {ds[i][1] for i in range(len(ds))}
    assert directions == {0, 1}  # fr=0, bk=1


def test_dataset_dataloader_batches(data_dir):
    ds = SpritesheetDataset(str(data_dir))
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    x_batch, cond_batch, paths = next(iter(loader))
    assert x_batch.shape == (2, 2, 32, 32)
    assert cond_batch.shape == (2,)
    assert len(paths) == 2
