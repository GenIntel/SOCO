"""Side-by-side SOCO annotation visualization."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt

from .data import SOCOPair


def plot_pair(sample: SOCOPair, mode: Literal["semantic", "concept"] = "semantic"):
    """Plot source and target matches with one color per correspondence group."""
    if mode not in {"semantic", "concept"}:
        raise ValueError("mode must be 'semantic' or 'concept'")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, view, title in zip(axes, (sample.source, sample.target), ("Source", "Target")):
        axis.imshow(view.image)
        axis.set_title(f"{title}: {view.category}/{view.image_name}")
        axis.axis("off")

    matches = sample.semantic_matches if mode == "semantic" else sample.concept_matches
    colors = plt.get_cmap("tab20")
    for index, match in enumerate(matches):
        color = colors(index % 20)
        for axis, view, indices in zip(
            axes,
            (sample.source, sample.target),
            (match.source_indices, match.target_indices),
        ):
            points = [view.keypoints[i].position for i in indices]
            axis.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                s=42,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=match.label,
            )

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        figure.legend(unique.values(), unique.keys(), loc="lower center", ncol=min(5, len(unique)))
        figure.subplots_adjust(bottom=0.2, top=0.82)
    metric = "CC" if mode == "concept" else "Cross-SOC" if sample.pair_type == "cross" else "SOC"
    figure.suptitle(f"{metric}: {sample.pair_group}/{sample.pair_id}")
    return figure, axes
