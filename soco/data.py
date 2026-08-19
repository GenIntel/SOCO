"""Lazy loader for SOCO's self-contained pair annotations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

from PIL import Image

PairType = Literal["intra", "cross", "train", "test"]

_PAIR_DIRS = {
    "intra": Path("PairAnnotations/intra"),
    "cross": Path("PairAnnotations/cross"),
    "train": Path("PairAnnotations/trainsplits/train"),
    "test": Path("PairAnnotations/trainsplits/test"),
}


@dataclass(frozen=True)
class Keypoint:
    id: int
    name: str
    position: tuple[float, float]
    concept_id: int | None
    concept_name: str | None


@dataclass(frozen=True)
class Match:
    """A named source-to-target match; index tuples allow one-to-many CC."""

    label: str
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]


@dataclass(frozen=True)
class SOCOView:
    category: str
    image_name: str
    image_path: Path
    image: Image.Image
    bbox: tuple[float, float, float, float] | None
    keypoints: tuple[Keypoint, ...]

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image.size


@dataclass(frozen=True)
class SOCOPair:
    pair_id: int
    pair_group: str
    pair_type: PairType
    annotation_path: Path
    source: SOCOView
    target: SOCOView
    semantic_matches: tuple[Match, ...]
    concept_matches: tuple[Match, ...]


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _keypoints(items: Sequence[Mapping[str, Any]]) -> tuple[Keypoint, ...]:
    return tuple(
        Keypoint(
            id=int(item["id"]),
            name=str(item["name"]),
            position=tuple(map(float, item["pos"])),
            concept_id=int(item["concept_id"]) if item.get("concept_id") is not None else None,
            concept_name=item.get("concept_name"),
        )
        for item in items
    )


class SOCOPairs(Sequence[SOCOPair]):
    """A deterministic sequence that parses pair JSON and images on access."""

    def __init__(
        self,
        root: str | Path,
        pair_type: PairType = "intra",
        categories: Sequence[str] | None = None,
        max_pairs_per_category: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        if pair_type not in _PAIR_DIRS:
            raise ValueError(f"pair_type must be one of {tuple(_PAIR_DIRS)}")
        if max_pairs_per_category is not None and max_pairs_per_category < 0:
            raise ValueError("max_pairs_per_category must be non-negative or None")

        self.pair_type = pair_type
        self.image_root = self.root / "Images"
        self.pair_root = self.root / _PAIR_DIRS[pair_type]
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Missing SOCO image directory: {self.image_root}")
        if not self.pair_root.is_dir():
            raise FileNotFoundError(f"Missing SOCO pair directory: {self.pair_root}")

        available = tuple(sorted(path.name for path in self.pair_root.iterdir() if path.is_dir()))
        self.categories = tuple(categories) if categories is not None else available
        unknown = sorted(set(self.categories) - set(available))
        if unknown:
            raise ValueError(f"Unknown categories for {pair_type}: {', '.join(unknown)}")

        limit = None if max_pairs_per_category in (None, 0) else max_pairs_per_category
        paths: list[Path] = []
        for category in self.categories:
            category_paths = sorted((self.pair_root / category).glob("*.json"))
            paths.extend(category_paths if limit is None else category_paths[:limit])
        self.annotation_paths = tuple(paths)

    def __len__(self) -> int:
        return len(self.annotation_paths)

    def __iter__(self) -> Iterator[SOCOPair]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int | slice) -> SOCOPair | list[SOCOPair]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        path = self.annotation_paths[index]
        return self._parse_pair(_read_json(path), path)

    def _parse_pair(self, data: Mapping[str, Any], path: Path) -> SOCOPair:
        default_category = data.get("category")
        source_category = str(data.get("src_category") or default_category)
        target_category = str(data.get("trg_category") or default_category)
        source_points = _keypoints(data["src_keypoints"])
        target_points = _keypoints(data["trg_keypoints"])
        source_index = {point.name: i for i, point in enumerate(source_points)}
        target_index = {point.name: i for i, point in enumerate(target_points)}

        semantic = tuple(
            Match(str(name), (source_index[str(name)],), (target_index[str(name)],))
            for name in data["semantic_overlap"]
        )
        concept = tuple(
            Match(
                str(group["concept_name"]),
                tuple(source_index[str(name)] for name in group["src_keypoints"]),
                tuple(target_index[str(name)] for name in group["trg_keypoints"]),
            )
            for group in data["concept_matches"]
        )

        return SOCOPair(
            pair_id=int(data["pair_id"]),
            pair_group=str(data.get("pair_group", default_category)),
            pair_type=self.pair_type,
            annotation_path=path,
            source=self._view(source_category, str(data["src_image"]), data.get("src_bndbox"), source_points),
            target=self._view(target_category, str(data["trg_image"]), data.get("trg_bndbox"), target_points),
            semantic_matches=semantic,
            concept_matches=concept,
        )

    def _view(
        self,
        category: str,
        image_name: str,
        bbox: Sequence[float] | None,
        keypoints: tuple[Keypoint, ...],
    ) -> SOCOView:
        image_path = self.image_root / category / image_name
        with Image.open(image_path) as image:
            rgb = image.convert("RGB").copy()
        return SOCOView(
            category=category,
            image_name=image_name,
            image_path=image_path,
            image=rgb,
            bbox=tuple(map(float, bbox)) if bbox is not None else None,
            keypoints=keypoints,
        )


def load_keypoint_taxonomy(root: str | Path) -> Mapping[str, Any]:
    return _read_json(Path(root).expanduser() / "Metadata/keypoint_taxonomy.json")


def load_filename_mapping(root: str | Path) -> Mapping[str, Any]:
    return _read_json(Path(root).expanduser() / "Metadata/filename_mapping.json")
