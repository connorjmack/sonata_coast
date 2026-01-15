# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run Sonata on LAS/LAZ/PLY, then export PCA visualization (and optional features).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import torch

import sonata

try:
    import laspy
except ImportError:
    laspy = None

try:
    import flash_attn
except ImportError:
    flash_attn = None


def get_pca_color(feat: torch.Tensor, brightness: float = 1.2, center: bool = True):
    u, s, v = torch.pca_lowrank(feat, center=center, q=6, niter=5)
    projection = feat @ v
    projection = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    return color.clamp(0.0, 1.0)


def estimate_normals(coord: np.ndarray, radius: float, max_nn: int) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )
    pcd.normalize_normals()
    return np.asarray(pcd.normals, dtype=np.float32)


def load_las(path: Path):
    if laspy is None:
        raise RuntimeError("laspy is required for LAS/LAZ. Install with `pip install laspy lazrs`.")
    las = laspy.read(path)
    coord = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)

    dims = set(las.point_format.dimension_names)
    normals = None
    if {"normal_x", "normal_y", "normal_z"}.issubset(dims):
        normals = np.vstack(
            (las["normal_x"], las["normal_y"], las["normal_z"])
        ).T.astype(np.float32)

    color = None
    if {"red", "green", "blue"}.issubset(dims):
        color = np.vstack((las.red, las.green, las.blue)).T.astype(np.float32)
        if color.max() > 255.0:
            color *= 255.0 / max(color.max(), 1.0)

    intensity = None
    if "intensity" in dims:
        intensity = np.asarray(las.intensity, dtype=np.float32)

    return coord, color, normals, intensity


def load_ply(path: Path):
    pcd = o3d.io.read_point_cloud(str(path))
    coord = np.asarray(pcd.points, dtype=np.float32)
    color = None
    if pcd.has_colors():
        color = np.asarray(pcd.colors, dtype=np.float32)
        if color.max() <= 1.0:
            color *= 255.0
        elif color.max() > 255.0:
            color *= 255.0 / max(color.max(), 1.0)
    normals = None
    if pcd.has_normals():
        normals = np.asarray(pcd.normals, dtype=np.float32)
    return coord, color, normals, None


def ensure_color_and_normals(
    coord: np.ndarray,
    color: np.ndarray | None,
    normals: np.ndarray | None,
    intensity: np.ndarray | None,
    normal_radius: float,
    normal_max_nn: int,
):
    if color is None:
        if intensity is not None and intensity.max() > 0:
            scaled = intensity / intensity.max() * 255.0
            color = np.repeat(scaled[:, None], 3, axis=1).astype(np.float32)
        else:
            color = np.zeros((coord.shape[0], 3), dtype=np.float32)
    if normals is None:
        normals = estimate_normals(coord, normal_radius, normal_max_nn)
    return color, normals


def build_transform(grid_size: float):
    config = [
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True,
        ),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "color", "inverse"),
            feat_keys=("coord", "color", "normal"),
        ),
    ]
    return sonata.transform.Compose(config)


def upscale_features(point):
    for _ in range(2):
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    return point


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Sonata on LAS/LAZ/PLY and export PCA visualization."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "data/training/Encinitas/20190228_00716_00777_NoWaves_SouthCarlsbad_beach_cliff_ground_cropped.las"
        ),
        help="Directory or file containing LAS/LAZ/PLY data.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--recursive", action="store_true", help="Search recursively.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N files.")
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument("--normal-radius", type=float, default=0.5)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Randomly subsample points before inference if > 0.",
    )
    parser.add_argument("--seed", type=int, default=53124)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="sonata")
    parser.add_argument("--repo-id", type=str, default="facebook/sonata")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--disable-flash", action="store_true")
    parser.add_argument("--enc-patch-size", type=int, default=1024)
    parser.add_argument(
        "--dump-features",
        action="store_true",
        help="Save features + mappings for linear probe training.",
    )
    parser.add_argument(
        "--dump-mode",
        choices=("grid", "full"),
        default="grid",
        help="Save grid features with inverse mapping, or full per-point features.",
    )
    parser.add_argument(
        "--feat-dtype",
        choices=("float16", "float32"),
        default="float32",
        help="Feature dtype for saved arrays.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sonata.utils.set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.data_dir.exists():
        raise RuntimeError(f"Data path not found: {args.data_dir}")

    suffixes = {".las", ".laz", ".ply"}
    if args.data_dir.is_file():
        files = [args.data_dir]
    else:
        globber = args.data_dir.rglob("*") if args.recursive else args.data_dir.glob("*")
        files = sorted([p for p in globber if p.suffix.lower() in suffixes])
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No LAS/LAZ/PLY files found in {args.data_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print("CUDA not available, falling back to CPU.", file=sys.stderr)

    if args.ckpt:
        model = sonata.model.load(args.ckpt).to(device)
    else:
        if flash_attn is not None and not args.disable_flash:
            model = sonata.model.load(args.model, repo_id=args.repo_id).to(device)
        else:
            custom_config = dict(
                enc_patch_size=[args.enc_patch_size for _ in range(5)],
                enable_flash=False,
            )
            model = sonata.model.load(
                args.model, repo_id=args.repo_id, custom_config=custom_config
            ).to(device)
    model.eval()
    transform = build_transform(args.grid_size)

    rng = np.random.default_rng(args.seed)
    for path in files:
        if path.suffix.lower() in {".las", ".laz"}:
            coord, color, normals, intensity = load_las(path)
        else:
            coord, color, normals, intensity = load_ply(path)

        if coord.size == 0:
            print(f"Skipping empty point cloud: {path}", file=sys.stderr)
            continue

        subsample_idx = None
        if args.max_points and coord.shape[0] > args.max_points:
            subsample_idx = rng.choice(coord.shape[0], args.max_points, replace=False)
            coord = coord[subsample_idx]
            if color is not None:
                color = color[subsample_idx]
            if normals is not None:
                normals = normals[subsample_idx]
            if intensity is not None:
                intensity = intensity[subsample_idx]

        color, normals = ensure_color_and_normals(
            coord, color, normals, intensity, args.normal_radius, args.normal_max_nn
        )

        point = {
            "coord": coord,
            "color": color,
            "normal": normals,
        }
        original_coord = point["coord"].copy()
        point = transform(point)

        with torch.inference_mode():
            for key, val in point.items():
                if isinstance(val, torch.Tensor):
                    point[key] = val.to(device, non_blocking=True)
            point = model(point)
            point = upscale_features(point)

        feat = point.feat
        pca_color = get_pca_color(feat, brightness=1.2, center=True)
        pca_full = pca_color[point.inverse].cpu().numpy()

        pca_ply = args.out_dir / f"{path.stem}_pca.ply"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(original_coord)
        pcd.colors = o3d.utility.Vector3dVector(pca_full)
        o3d.io.write_point_cloud(str(pca_ply), pcd)
        print(f"Wrote: {pca_ply}")

        if args.dump_features:
            feat_dtype = np.float16 if args.feat_dtype == "float16" else np.float32
            dump_path = args.out_dir / f"{path.stem}_feat_{args.dump_mode}.npz"
            if args.dump_mode == "full":
                feat_dump = to_numpy(feat[point.inverse]).astype(feat_dtype, copy=False)
                coord_dump = original_coord.astype(np.float32, copy=False)
                np.savez_compressed(
                    dump_path,
                    feat=feat_dump,
                    coord=coord_dump,
                    subsample_idx=subsample_idx,
                )
            else:
                feat_dump = to_numpy(feat).astype(feat_dtype, copy=False)
                coord_dump = to_numpy(point.coord).astype(np.float32, copy=False)
                grid_coord = to_numpy(point.grid_coord).astype(np.int32, copy=False)
                inverse = to_numpy(point.inverse).astype(np.int64, copy=False)
                np.savez_compressed(
                    dump_path,
                    feat=feat_dump,
                    coord=coord_dump,
                    grid_coord=grid_coord,
                    inverse=inverse,
                    coord_full=original_coord.astype(np.float32, copy=False),
                    subsample_idx=subsample_idx,
                )
            print(f"Wrote: {dump_path}")


if __name__ == "__main__":
    main()
