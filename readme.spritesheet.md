# D3PM Spritesheet — Walk Cycle Frame Prediction

Train a discrete diffusion model to predict pixel-art walk-cycle frames.
Given a character's frame1 in any direction, the model generates frame2.

## How it works

Each sprite in `data/` follows the naming convention `{char}_{dir}{frame}.gif`:

- `{dir}` is one of `fr` (front), `bk` (back), `lf` (left), `rt` (right)
- `{frame}` is `1` (input) or `2` (target)

The model treats each `(frame1, frame2)` pair as a 2-channel `[2, H, W]` palette-index tensor. During training, both frames are diffused together. During inference, frame1 is clamped to its known values after every reverse-diffusion step, so only frame2 is generated.

## Setup

```bash
uv venv
uv pip install torch torchvision numpy Pillow tqdm pytest
```

## Training

```bash
uv run python3 d3pm_runner_spritesheet.py
```

Checkpoints are saved to `ckpt/spritesheet_step{N}.pt` every 1000 steps.
Training visualizations (frame1 | frame2_predicted) are saved to `contents/sprite_step{N}.png` every 300 steps.

## Inference — predict frame2 from frame1

```python
import torch
from PIL import Image
import numpy as np
from d3pm_runner import D3PM
from d3pm_runner_spritesheet import SpriteX0Model, generate_frame, save_as_gif

# Load model
N = 246  # match palette_size used during training (printed at startup)
model = SpriteX0Model(n_channel=2, N=N)
model.load_state_dict(torch.load("ckpt/spritesheet_final.pt", map_location="cpu"))
d3pm = D3PM(model, n_T=1000, num_classes=N, hybrid_loss_coeff=0.0)
d3pm.eval()

# Load frame1
frame1_img = Image.open("data/amg1_fr1.gif")
palette = frame1_img.getpalette()           # flat list of 768 ints
frame1 = torch.tensor(np.array(frame1_img), dtype=torch.long)  # [H, W]

# Generate frame2
with torch.no_grad():
    frame2 = generate_frame(d3pm, {0: frame1}, direction=0, device="cpu")

# Save as GIF with original palette
save_as_gif(frame2, palette, "frame2_predicted.gif")
```

Direction codes: `fr=0, bk=1, lf=2, rt=3`

## Inference — generate a longer walk cycle autoregressively

Each call to `generate_frames` chains `generate_frame` calls, using each
generated frame as the anchor for the next. Note that drift accumulates over
long sequences because earlier frames are not re-anchored.

```python
from d3pm_runner_spritesheet import generate_frames

with torch.no_grad():
    frames = generate_frames(d3pm, frame1, direction=0, device="cpu", n_frames=8)

for i, f in enumerate(frames):
    save_as_gif(f, palette, f"walk_{i:02d}.gif")
```

## Data format

Files in `data/` must match `{char}_{dir}1.gif` / `{char}_{dir}2.gif` pairs.
Files that do not match this pattern (e.g. `house1.gif`, `back2.jpg`) are silently skipped.
Unpaired files (frame1 present but frame2 missing) are also skipped.
The `palette_size` (number of discrete classes) is inferred automatically from the maximum palette index found across all GIF files.

Sprite spatial dimensions must be divisible by 16 for the 4-level U-Net downsampling.

## Model

`SpriteX0Model` is a 2-channel U-Net with transformer bottleneck, subclassing `DummyX0Model` from `d3pm_runner.py`. It replaces the 10-class MNIST conditioning embeddings with 4-class direction embeddings (one per direction). The model takes walk-direction labels (0–3) as conditioning at each scale of the U-Net.

## Tests

```bash
uv run python3 -m pytest tests/test_spritesheet.py -v
```
