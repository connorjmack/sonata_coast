# AGENTS.md

## Mission
Use the pre-trained Sonata Point Transformer V3 encoder to classify coastal cliff LiDAR
point clouds (point-level semantic segmentation, or optional scene-level classification).

## Repo map (start here)
- `README.md`: install + quick-start + model loading + feature upsampling recipe.
- `demo/2_sem_seg.py`: linear head example for point-level semantic segmentation.
- `sonata/transform.py`: default preprocessing pipeline (`default()`).
- `sonata/data.py`: `collate_fn` for batched point clouds.
- `sonata/model.py`: `sonata.model.load`, `PointTransformerV3` backbone.

## Data expectations
Input is a dict of numpy arrays (single cloud) or tensors (batched).
Required keys for default pipeline:
- `coord`: (N, 3) XYZ in meters.
- `color`: (N, 3) RGB in [0, 255]. If LiDAR has intensity only, repeat it into 3 channels.
- `normal`: (N, 3) unit normals. If absent, estimate normals or use zeros and adjust
  `feat_keys` in the Collect step.
- Optional: `segment`: (N,) integer labels for training/eval.

For batching, add `batch` (N,) or use `sonata.data.collate_fn`.

## Preprocessing pipeline
`sonata.transform.default()` does:
1. CenterShift (XY center, Z min).
2. GridSample with `grid_size=0.02` (tune to LiDAR density; larger saves memory).
3. NormalizeColor, ToTensor.
4. Collect keys and features: `feat = concat(coord, color, normal)`.

If your features differ, clone the default config and update the `Collect` `feat_keys`.
Reference: `sonata/transform.py:1205`.

## Inference recipe (point-level classification)
Example wiring for coastal classes (rock/soil/veg/water/etc). Adapt the class count.

```python
import torch
import sonata
import torch.nn as nn

num_classes = 6
model = sonata.model.load("sonata", repo_id="facebook/sonata").cuda()
transform = sonata.transform.default()
head = None  # initialize after you know feat dim (or load a trained head)

point = transform(point)  # point is a dict with coord/color/normal
for key, val in point.items():
    if isinstance(val, torch.Tensor):
        point[key] = val.cuda(non_blocking=True)

with torch.inference_mode():
    point = model(point)
    # upsample hierarchical features (see README and demo/2_sem_seg.py)
    for _ in range(2):
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point:
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent

    feat = point.feat  # downsampled grid points
    if head is None:
        head = nn.Linear(feat.shape[-1], num_classes).cuda()
    logits = head(feat)
    pred_grid = logits.argmax(dim=-1)
    pred_full = pred_grid[point.inverse]  # map back to original points
```

Notes:
- If FlashAttention is missing, load with `enable_flash=False` and reduce `enc_patch_size`
  (see `README.md` and `demo/2_sem_seg.py`).
- `point.inverse` is created by GridSample to map back to the original points.

## Training / fine-tuning guidance
- Start with a frozen encoder + train only a linear head (like `demo/2_sem_seg.py`).
- Use `sonata.data.collate_fn` for batching; ensure `segment` is aligned with `coord`.
- Once stable, optionally fine-tune the encoder with a smaller LR.

## Coastal cliff specifics
- Define a consistent label schema (rock, talus, soil, vegetation, water, man-made, etc).
- Normalize units (meters) and coordinate frame before CenterShift.
- Tune `grid_size` to the LiDAR point spacing to avoid over-downsampling.

## Common gotchas
- Missing `color`/`normal` will break `Collect` unless you adjust `feat_keys`.
- Large scenes need larger `grid_size` or smaller `enc_patch_size` to fit GPU memory.
- The model is encoder-only; always add your own classification head.
