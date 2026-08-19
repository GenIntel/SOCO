"""Small smoke tests for the public loading and plotting paths."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import pytest

from soco import SOCOPairs, load_filename_mapping, load_keypoint_taxonomy, plot_pair


def _point(name: str, point_id: int, position):
    return {
        "id": point_id,
        "name": name,
        "pos": position,
        "concept_id": 10,
        "concept_name": "wheel",
    }


@pytest.fixture
def mini_soco(tmp_path: Path) -> Path:
    root = tmp_path / "SOCOv1"
    for category, size in (("car", (10, 8)), ("bus", (12, 8))):
        image_dir = root / "Images" / category
        image_dir.mkdir(parents=True)
        Image.new("RGB", size, color="white").save(image_dir / f"{category}_1.JPEG")

    source_points = [_point("front_wheel", 1, [2, 6]), _point("rear_wheel", 2, [8, 6])]
    target_points = [_point("front_wheel", 1, [3, 6]), _point("rear_wheel", 2, [9, 6])]
    for pair_type, source_category in (("intra", "car"), ("cross", "bus")):
        pair_dir = root / "PairAnnotations" / pair_type / "car"
        pair_dir.mkdir(parents=True)
        payload = {
            "pair_id": 1,
            "category": "car",
            "pair_group": "car",
            "src_category": source_category,
            "trg_category": "car",
            "src_image": f"{source_category}_1.JPEG",
            "trg_image": "car_1.JPEG",
            "src_bndbox": [1, 1, 9, 7],
            "trg_bndbox": [1, 1, 9, 7],
            "src_keypoints": source_points,
            "trg_keypoints": target_points,
            "semantic_overlap": ["front_wheel", "rear_wheel"],
            "concept_matches": [
                {
                    "concept_name": "wheel",
                    "src_keypoints": ["front_wheel", "rear_wheel"],
                    "trg_keypoints": ["front_wheel", "rear_wheel"],
                }
            ],
        }
        (pair_dir / "00001.json").write_text(json.dumps(payload), encoding="utf-8")

    source = root / "PairAnnotations/intra/car/00001.json"
    for split in ("train", "test"):
        pair_dir = root / "PairAnnotations/trainsplits" / split / "car"
        pair_dir.mkdir(parents=True)
        (pair_dir / "00001.json").write_text(source.read_text(), encoding="utf-8")

    metadata = root / "Metadata"
    metadata.mkdir()
    (metadata / "keypoint_taxonomy.json").write_text('{"instances": []}', encoding="utf-8")
    (metadata / "filename_mapping.json").write_text('{"car": {}}', encoding="utf-8")
    return root


@pytest.mark.parametrize("pair_type", ["intra", "cross", "train", "test"])
def test_loads_released_pair_layouts(mini_soco, pair_type):
    sample = SOCOPairs(mini_soco, pair_type=pair_type)[0]
    assert sample.source.image.mode == "RGB"
    assert sample.source.keypoints[0].position == (2.0, 6.0)
    assert sample.semantic_matches[0].source_indices == (0,)
    assert sample.concept_matches[0].target_indices == (0, 1)


def test_selection_and_metadata(mini_soco):
    assert len(SOCOPairs(mini_soco, max_pairs_per_category=0)) == 1
    assert SOCOPairs(mini_soco, pair_type="cross")[0].source.category == "bus"
    assert load_keypoint_taxonomy(mini_soco) == {"instances": []}
    assert load_filename_mapping(mini_soco) == {"car": {}}
    with pytest.raises(ValueError, match="Unknown categories"):
        SOCOPairs(mini_soco, categories=["plane"])


@pytest.mark.parametrize(
    ("pair_type", "mode", "title"),
    [("intra", "semantic", "SOC"), ("cross", "semantic", "Cross-SOC"), ("cross", "concept", "CC")],
)
def test_plot_matches_and_titles(mini_soco, pair_type, mode, title):
    sample = SOCOPairs(mini_soco, pair_type=pair_type)[0]
    figure, axes = plot_pair(sample, mode)
    matches = sample.semantic_matches if mode == "semantic" else sample.concept_matches
    expected = [sum(len(match.source_indices) for match in matches), sum(len(match.target_indices) for match in matches)]
    actual = [sum(len(collection.get_offsets()) for collection in axis.collections) for axis in axes]
    assert actual == expected
    assert figure._suptitle.get_text().startswith(f"{title}:")
    figure.canvas.draw()
    plt.close(figure)
