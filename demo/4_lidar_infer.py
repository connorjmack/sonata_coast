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
Run Sonata on LAS/LAZ/PLY, then export PCA and k-means cluster visualizations.
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


def assign_labels(feat: torch.Tensor, centers: torch.Tensor, chunk_size: int):
    n = feat.shape[0]
    labels = torch.empty(n, device=feat.device, dtype=torch.long)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = feat[start:end]
        dist = torch.cdist(chunk, centers)
        labels[start:end] = dist.argmin(dim=1)
    return labels


def kmeans_torch(
    feat: torch.Tensor,
    num_clusters: int,
    max_iter: int = 20,
    chunk_size: int = 100000,
    seed: int = 0,
):
    device = feat.device
    n = feat.shape[0]
    if n < num_clusters:
        raise RuntimeError("Number of clusters exceeds number of points.")
    gen = torch.Generator(device=device).manual_seed(seed)
    centers = feat[torch.randperm(n, generator=gen, device=device)[:num_clusters]].clone()

    for _ in range(max_iter):
        counts = torch.zeros(num_clusters, device=device, dtype=feat.dtype)
        sums = torch.zeros(num_clusters, feat.shape[1], device=device, dtype=feat.dtype)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = feat[start:end]
            dist = torch.cdist(chunk, centers)
            labels = dist.argmin(dim=1)
            sums.index_add_(0, labels, chunk)
            counts.index_add_(
                0, labels, torch.ones_like(labels, dtype=feat.dtype, device=device)
            )
        mask = counts > 0
        new_centers = centers.clone()
        new_centers[mask] = sums[mask] / counts[mask].unsqueeze(1)
        shift = torch.norm(new_centers - centers)
        centers = new_centers
        if shift < 1e-4:
            break
    labels = assign_labels(feat, centers, chunk_size)
    return labels, centers


def label_colors(labels: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 255, size=(labels.max() + 1, 3), dtype=np.uint8)
    return palette[labels].astype(np.float32) / 255.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Sonata on LAS/LAZ/PLY and export PCA + cluster PLYs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Volumes/group/LiDAR/LidarProcessing/ptv3/data/training"),
        help="Directory containing LAS/LAZ/PLY files.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
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
    parser.add_argument("--num-clusters", type=int, default=12)
    parser.add_argument("--cluster-subsample", type=int, default=200000)
    parser.add_argument("--cluster-iter", type=int, default=20)
    parser.add_argument("--cluster-chunk", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=53124)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="sonata")
    parser.add_argument("--repo-id", type=str, default="facebook/sonata")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--disable-flash", action="store_true")
    parser.add_argument("--enc-patch-size", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    sonata.utils.set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.data_dir.exists():
        raise RuntimeError(f"Data dir not found: {args.data_dir}")

    suffixes = {".las", ".laz", ".ply"}
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

    for path in files:
        if path.suffix.lower() in {".las", ".laz"}:
            coord, color, normals, intensity = load_las(path)
        else:
            coord, color, normals, intensity = load_ply(path)

        if coord.size == 0:
            print(f"Skipping empty point cloud: {path}", file=sys.stderr)
            continue

        if args.max_points and coord.shape[0] > args.max_points:
            idx = np.random.choice(coord.shape[0], args.max_points, replace=False)
            coord = coord[idx]
            if color is not None:
                color = color[idx]
            if normals is not None:
                normals = normals[idx]
            if intensity is not None:
                intensity = intensity[idx]

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

        feat_for_kmeans = feat
        if args.cluster_subsample and feat.shape[0] > args.cluster_subsample:
            idx = torch.randperm(
                feat.shape[0], device=feat.device
            )[: args.cluster_subsample]
            feat_for_kmeans = feat[idx]

        _, centers = kmeans_torch(
            feat_for_kmeans,
            num_clusters=args.num_clusters,
            max_iter=args.cluster_iter,
            chunk_size=args.cluster_chunk,
            seed=args.seed,
        )
        labels_full = assign_labels(feat, centers, args.cluster_chunk)
        labels_full = labels_full[point.inverse].cpu().numpy()
        cluster_color = label_colors(labels_full, seed=args.seed)

        pca_ply = args.out_dir / f"{path.stem}_pca.ply"
        cluster_ply = args.out_dir / f"{path.stem}_cluster_k{args.num_clusters}.ply"
        cluster_npy = args.out_dir / f"{path.stem}_cluster_k{args.num_clusters}.npy"

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(original_coord)
        pcd.colors = o3d.utility.Vector3dVector(pca_full)
        o3d.io.write_point_cloud(str(pca_ply), pcd)

        pcd.colors = o3d.utility.Vector3dVector(cluster_color)
        o3d.io.write_point_cloud(str(cluster_ply), pcd)
        np.save(cluster_npy, labels_full)

        print(f"Wrote: {pca_ply}")
        print(f"Wrote: {cluster_ply}")
        print(f"Wrote: {cluster_npy}")


if __name__ == "__main__":
    main()
