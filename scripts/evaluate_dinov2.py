#!/usr/bin/env python3
"""Evaluate DINOv2 nearest-neighbor correspondence on SOCO."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from soco import Match, SOCOPairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("SOCO_ROOT", "data/SOCOv1")),
        help="Extracted SOCOv1 directory (default: SOCO_ROOT or data/SOCOv1)",
    )
    parser.add_argument("--pair-type", choices=("intra", "cross", "both"), default="both")
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Pair group/category to evaluate; repeat for multiple categories",
    )
    parser.add_argument(
        "--max-pairs-per-category",
        type=int,
        default=1,
        help="Deterministic per-category limit; 0 evaluates every pair (default: 1)",
    )
    parser.add_argument("--image-size", type=int, default=800, help="Requested square input size")
    parser.add_argument("--threshold", type=float, default=0.1, help="BBox-normalized PCK threshold")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. auto, cuda, cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dinov2_vitb14_soco.json"),
        help="Aggregate JSON output path",
    )
    return parser.parse_args()


def load_dinov2(device: str):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("DINOv2 dependencies are missing. Run: uv sync --extra dinov2") from exc

    resolved = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    model = model.to(resolved).eval().requires_grad_(False)
    patch_size = model.patch_size[0] if isinstance(model.patch_size, tuple) else model.patch_size
    return model, resolved, int(patch_size)


def prepare_image(image, size: int):
    from PIL import Image
    from torchvision import transforms

    side = max(image.size)
    square = Image.new("RGB", (side, side), color="white")
    square.paste(image, (0, 0))
    transform = transforms.Compose(
        [
            transforms.Resize(
                (size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transform(square)


def _dense_features(model, images, patch_size: int):
    """Return raw final-block patch tokens, matching OmniProbe's DINOv2 wrapper."""
    import torch.nn.functional as functional

    grid_height = images.shape[-2] // patch_size
    grid_width = images.shape[-1] // patch_size
    tokens = model.prepare_tokens_with_masks(images, None)
    for block in model.blocks:
        tokens = block(tokens)
    tokens = tokens[:, -(grid_height * grid_width) :]
    dense = tokens.transpose(1, 2).reshape(images.shape[0], -1, grid_height, grid_width)
    return functional.normalize(dense, p=2, dim=1)


def _eval_points(keypoints, image_size, effective_size: int, device):
    """Apply OmniProbe's two integer truncations before normalizing coordinates."""
    import torch

    points = torch.tensor(
        [[int(point.position[0]), int(point.position[1])] for point in keypoints],
        dtype=torch.int32,
    )
    points = (points * effective_size / max(image_size)).to(torch.int32)
    return points.to(device=device, dtype=torch.float32).div_(effective_size)


def predict_pair(model, sample, effective_size: int, patch_size: int, device):
    import torch
    import torch.nn.functional as functional

    images = torch.stack(
        (prepare_image(sample.source.image, effective_size), prepare_image(sample.target.image, effective_size))
    ).to(device)
    with torch.inference_mode():
        dense = _dense_features(model, images, patch_size)

    source_points = _eval_points(sample.source.keypoints, sample.source.image_size, effective_size, device)
    query_grid = (source_points * 2.0 - 1.0)[None, None]
    query_features = functional.grid_sample(
        dense[0:1], query_grid, mode="bilinear", align_corners=True
    )[0, :, 0].transpose(0, 1)
    similarities = torch.einsum("kc,chw->khw", query_features, dense[1])
    flat_indices = similarities.flatten(start_dim=1).argmax(dim=1)
    grid_height, grid_width = dense.shape[-2:]
    rows = torch.div(flat_indices, grid_width, rounding_mode="floor")
    columns = flat_indices % grid_width
    return torch.stack((columns / grid_width, rows / grid_height), dim=1)


def _pck(predictions, targets, matches: tuple[Match, ...], radius: float) -> tuple[float | None, int]:
    import torch

    # A concept can span several annotation groups; merge valid targets per source.
    valid_targets: dict[int, set[int]] = {}
    for match in matches:
        for source_index in match.source_indices:
            valid_targets.setdefault(source_index, set()).update(match.target_indices)
    if not valid_targets:
        return None, 0

    correct = []
    for source_index, target_indices in valid_targets.items():
        errors = torch.linalg.vector_norm(
            predictions[source_index] - targets[sorted(target_indices)], dim=1
        )
        correct.append(errors.min() < radius)
    return float(torch.stack(correct).float().mean().mul(100).cpu()), len(correct)


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["category"], []).append(row)

    categories = {}
    for category, category_rows in sorted(grouped.items()):
        categories[category] = {
            "pairs": len(category_rows),
            "semantic_pck": _mean([row["semantic_pck"] for row in category_rows]),
            "concept_pck": _mean([row["concept_pck"] for row in category_rows]),
            "semantic_keypoints": sum(row["semantic_keypoints"] for row in category_rows),
            "concept_keypoints": sum(row["concept_keypoints"] for row in category_rows),
        }
    return {
        "overall": {
            "pairs": len(rows),
            "categories": len(categories),
            "semantic_pck": _mean([row["semantic_pck"] for row in categories.values()]),
            "concept_pck": _mean([row["concept_pck"] for row in categories.values()]),
        },
        "per_category": categories,
    }


def evaluate_pair_type(args, pair_type: str, model, device, effective_size: int, patch_size: int):
    from tqdm import tqdm

    dataset = SOCOPairs(
        args.data_root,
        pair_type=pair_type,
        categories=args.categories,
        max_pairs_per_category=args.max_pairs_per_category,
    )
    if not dataset:
        raise RuntimeError(f"No {pair_type} pair annotations were selected")

    rows = []
    for sample in tqdm(dataset, desc=f"SOCO {pair_type}", unit="pair"):
        predictions = predict_pair(model, sample, effective_size, patch_size, device)
        targets = _eval_points(sample.target.keypoints, sample.target.image_size, effective_size, device)
        if sample.target.bbox is None:
            bbox_scale = 1.0
        else:
            left, top, right, bottom = sample.target.bbox
            bbox_scale = max(right - left, bottom - top) / max(sample.target.image_size)
        radius = args.threshold * bbox_scale
        semantic_pck, semantic_count = _pck(
            predictions, targets, sample.semantic_matches, radius
        )
        concept_pck, concept_count = _pck(predictions, targets, sample.concept_matches, radius)
        rows.append(
            {
                "category": sample.pair_group,
                "semantic_pck": semantic_pck,
                "concept_pck": concept_pck,
                "semantic_keypoints": semantic_count,
                "concept_keypoints": concept_count,
            }
        )
    return _aggregate(rows)


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def main() -> None:
    args = parse_args()
    if args.max_pairs_per_category < 0 or args.image_size <= 0 or args.threshold < 0:
        raise SystemExit("pair limit, image size, and threshold must be non-negative (image size > 0)")

    model, device, patch_size = load_dinov2(args.device)
    effective_size = max(patch_size, round(args.image_size / patch_size) * patch_size)
    pair_types = ("intra", "cross") if args.pair_type == "both" else (args.pair_type,)
    metrics = {
        pair_type: evaluate_pair_type(args, pair_type, model, device, effective_size, patch_size)
        for pair_type in pair_types
    }

    for pair_type, result in metrics.items():
        overall = result["overall"]
        semantic_name = "SOC" if pair_type == "intra" else "Cross-SOC"
        print(
            f"{pair_type:5s} | pairs={overall['pairs']:5d} | "
            f"CC={_format_percent(overall['concept_pck'])} | "
            f"{semantic_name}={_format_percent(overall['semantic_pck'])}"
        )

    output = {
        "dataset": "SOCOv1",
        "model": "dinov2_vitb14",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "data_root": str(args.data_root.resolve()),
            "pair_type": args.pair_type,
            "categories": args.categories,
            "max_pairs_per_category": args.max_pairs_per_category,
            "requested_image_size": args.image_size,
            "effective_image_size": effective_size,
            "patch_size": patch_size,
            "threshold": args.threshold,
            "normalization": "target_bbox_max_dimension",
            "reduction": "pair_then_category",
            "device": str(device),
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
