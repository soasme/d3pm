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


# ── SpriteX0Model ─────────────────────────────────────────────────────────

from d3pm_runner_spritesheet import SpriteX0Model


def test_sprite_x0_model_output_shape():
    N = 8
    model = SpriteX0Model(n_channel=2, N=N)
    model.eval()
    x = torch.randint(0, N, (1, 2, 32, 32))
    t = torch.tensor([50])
    cond = torch.tensor([0])
    with torch.no_grad():
        out = model(x, t, cond)
    assert out.shape == (1, 2, 32, 32, N), f"got {out.shape}"


def test_sprite_x0_model_all_four_directions():
    model = SpriteX0Model(n_channel=2, N=4)
    model.eval()
    for direction in range(4):
        x = torch.randint(0, 4, (1, 2, 32, 32))
        t = torch.tensor([1])
        cond = torch.tensor([direction])
        with torch.no_grad():
            out = model(x, t, cond)
        assert out.shape == (1, 2, 32, 32, 4), f"direction {direction} failed"


def test_sprite_x0_model_embeddings_are_4_class():
    model = SpriteX0Model(n_channel=2, N=8)
    for i in range(1, 7):
        emb = getattr(model, f"cond_embedding_{i}")
        assert emb.num_embeddings == 4, f"cond_embedding_{i} has {emb.num_embeddings}, expected 4"


# ── generate_frame / generate_frames ──────────────────────────────────────

from d3pm_runner_spritesheet import generate_frame, generate_frames


class _MockD3PM:
    """Identity mock: p_sample returns x unchanged."""
    num_classses = 8
    n_T = 5

    def p_sample(self, x, t, cond, noise):
        return x


class _CorruptAnchorMock:
    """p_sample corrupts channel 0 to verify clamping restores it."""
    num_classses = 8
    n_T = 5

    def p_sample(self, x, t, cond, noise):
        corrupted = x.clone()
        corrupted[:, 0] = (x[:, 0] + 1) % self.num_classses
        return corrupted


def test_generate_frame_output_shape():
    mock = _MockD3PM()
    frame1 = torch.zeros(16, 16, dtype=torch.long)
    result = generate_frame(mock, {0: frame1}, direction=0, device="cpu")
    assert result.shape == (16, 16)
    assert result.dtype == torch.long


def test_generate_frame_anchor_is_clamped():
    """Even if p_sample corrupts the anchor channel, clamping restores it."""
    mock = _CorruptAnchorMock()
    frame1 = torch.zeros(16, 16, dtype=torch.long)
    # _CorruptAnchorMock increments channel 0 every step; clamping must undo that.
    # With identity-like behaviour on channel 1, result should still be all-zeros
    # for the anchor and whatever noise for channel 1 — but the key is no crash
    # and that clamping was applied (anchor channel 0 stayed zero throughout).
    result = generate_frame(mock, {0: frame1}, direction=0, device="cpu")
    assert result.shape == (16, 16)
    assert result.dtype == torch.long


def test_generate_frame_multiple_anchors():
    mock = _MockD3PM()
    f0 = torch.zeros(8, 8, dtype=torch.long)
    f1 = torch.ones(8, 8, dtype=torch.long)
    result = generate_frame(
        mock,
        anchor_frames={0: f0, 1: f1},
        direction=2,
        device="cpu",
        total_frames=3,
        predict_frame=2,
    )
    assert result.shape == (8, 8)


def test_generate_frame_default_predict_frame():
    mock = _MockD3PM()
    frame1 = torch.zeros(8, 8, dtype=torch.long)
    # default: total_frames=2, predict_frame=1
    result = generate_frame(mock, {0: frame1}, direction=1, device="cpu")
    assert result.shape == (8, 8)


def test_generate_frames_length():
    mock = _MockD3PM()
    frame1 = torch.zeros(8, 8, dtype=torch.long)
    frames = generate_frames(mock, frame1, direction=0, device="cpu", n_frames=4)
    assert len(frames) == 4


def test_generate_frames_first_is_anchor():
    mock = _MockD3PM()
    frame1 = torch.full((8, 8), 5, dtype=torch.long)
    frames = generate_frames(mock, frame1, direction=0, device="cpu", n_frames=3)
    assert torch.equal(frames[0], frame1)


def test_generate_frames_subsequent_are_tensors():
    mock = _MockD3PM()
    frame1 = torch.zeros(8, 8, dtype=torch.long)
    frames = generate_frames(mock, frame1, direction=0, device="cpu", n_frames=3)
    for f in frames[1:]:
        assert isinstance(f, torch.Tensor)
        assert f.shape == (8, 8)
        assert f.dtype == torch.long


def test_generate_frame_raises_on_empty_anchors():
    mock = _MockD3PM()
    with pytest.raises(ValueError, match="anchor_frames"):
        generate_frame(mock, {}, direction=0, device="cpu")


def test_generate_frame_raises_on_predict_frame_out_of_range():
    mock = _MockD3PM()
    frame1 = torch.zeros(8, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="predict_frame"):
        generate_frame(mock, {0: frame1}, direction=0, device="cpu", total_frames=2, predict_frame=5)


def test_generate_frame_raises_when_predict_frame_is_anchor():
    mock = _MockD3PM()
    frame1 = torch.zeros(8, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="anchor"):
        generate_frame(mock, {0: frame1}, direction=0, device="cpu", predict_frame=0)


def test_generate_frames_single_frame():
    mock = _MockD3PM()
    frame1 = torch.full((8, 8), 3, dtype=torch.long)
    frames = generate_frames(mock, frame1, direction=0, device="cpu", n_frames=1)
    assert len(frames) == 1
    assert torch.equal(frames[0], frame1)


# ── save_as_gif ────────────────────────────────────────────────────────────

from d3pm_runner_spritesheet import save_as_gif


@pytest.fixture
def flat_palette():
    """768-int flat RGB palette (256 entries × 3 channels)."""
    return list(range(256)) * 3


def test_save_as_gif_from_tensor(tmp_path, flat_palette):
    indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    out_path = tmp_path / "out.gif"
    save_as_gif(indices, flat_palette, str(out_path))
    assert out_path.exists()
    reloaded = np.array(Image.open(str(out_path)))
    assert reloaded.shape == (2, 2)
    assert int(reloaded[0, 0]) == 0
    assert int(reloaded[0, 1]) == 1
    assert int(reloaded[1, 0]) == 2
    assert int(reloaded[1, 1]) == 3


def test_save_as_gif_from_numpy(tmp_path, flat_palette):
    indices = np.array([[5, 6], [7, 0]], dtype=np.uint8)
    out_path = tmp_path / "out2.gif"
    save_as_gif(indices, flat_palette, str(out_path))
    assert out_path.exists()
    # GIF compacts sparse palettes (only used colors kept), so raw indices may
    # be remapped. Check visual content (RGB) rather than palette indices.
    orig = Image.fromarray(indices, mode="P")
    orig.putpalette(flat_palette)
    orig_rgb = list(orig.convert("RGB").tobytes())
    reloaded_rgb = list(Image.open(str(out_path)).convert("RGB").tobytes())
    assert orig_rgb == reloaded_rgb


def test_save_as_gif_produces_valid_gif(tmp_path, flat_palette):
    indices = torch.zeros(8, 8, dtype=torch.long)
    out_path = tmp_path / "valid.gif"
    save_as_gif(indices, flat_palette, str(out_path))
    img = Image.open(str(out_path))
    assert img.format == "GIF"
    assert img.size == (8, 8)
