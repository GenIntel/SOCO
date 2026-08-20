# SOCO: Benchmarking Semantic Object Correspondence in Vision Foundation Models

[Paper](https://arxiv.org/abs/2605.31597) · [Project page](https://genintel.github.io/SOCO/) · [SOCO dataset](https://huggingface.co/datasets/GenIntelLab/SOCO) · [SOCO-LVLM dataset](https://huggingface.co/datasets/GenIntelLab/SOCO-LVLM)

Minimal utilities for loading SOCO, inspecting its annotations, and running example evaluations for vision foundation models (VFMs) and large vision-language models (LVLMs).

SOCO contains 100 categories, 4,000 images, per-view keypoint annotations, 20k intra-category pairs, 20k cross-category pairs, and predefined 10k/10k intra-category train/test splits. See [Dataset structure](docs/dataset_structure.md) for the annotation contract and terminology.

## Repository layout

```text
soco/             SOCO loaders and visualization helpers
scripts/          Small VFM evaluation examples
notebooks/        Dataset exploration examples
VLMEvalKit/       SOCO-LVLM evaluation integration
docs/             Technical documentation for the code and datasets
```

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/) are expected.

```bash
# Loading and visualization
uv sync

# Notebook environment
uv sync --extra notebook

# Add DINOv2 inference dependencies
uv sync --extra dinov2
```

Download SOCO from its [Hugging Face repository](https://huggingface.co/datasets/GenIntelLab/SOCO):

```bash
huggingface-cli download GenIntelLab/SOCO --repo-type dataset --local-dir data/SOCOv1
cd data/SOCOv1
unzip Images.zip
unzip KeypointAnnotations.zip
unzip PairAnnotations.zip
```

Point the tools at the extracted directory:

```bash
export SOCO_ROOT=/path/to/SOCOv1
```

## Load pairs

Pair annotations are parsed lazily. Images are returned as RGB PIL images and all coordinates remain in their original image coordinate system.

```python
from soco import SOCOPairs

intra = SOCOPairs("/path/to/SOCOv1", pair_type="intra")
cross = SOCOPairs("/path/to/SOCOv1", pair_type="cross")

sample = intra[0]
print(sample.source.category, sample.target.category)
print(sample.semantic_matches)
print(sample.concept_matches)
```

Open `notebooks/explore_soco.ipynb` for deterministic intra- and cross-category examples with semantic and concept-match visualizations.

## Evaluation

### Vision foundation models

For large-scale probing, additional backbones, and distributed evaluation, use [OmniProbe](https://github.com/GenIntel/OmniProbe). This repository is the small, dataset-focused entry point, including a DINOv2 example aligned with OmniProbe's SOCO protocol.

#### DINOv2 nearest-neighbor example

The default is a quick, deterministic evaluation: one pair per category from both pair types. On first use, Torch Hub downloads the DINOv2 repository and weights.

```bash
uv sync --extra dinov2
uv run python scripts/evaluate_dinov2.py --data-root "$SOCO_ROOT"
```

Results are printed and written to `results/dinov2_vitb14_soco.json`. Evaluate the complete released pair sets with:

```bash
uv run python scripts/evaluate_dinov2.py \
  --data-root "$SOCO_ROOT" \
  --pair-type both \
  --max-pairs-per-category 0
```

The example follows the zero-shot protocol from the [SOCO paper](https://arxiv.org/abs/2605.31597): frozen dense features, cosine nearest-neighbor matching, and bounding-box-normalized PCK@0.1. The requested 800-pixel input becomes 798 pixels for the DINOv2-B/14 patch grid. Metrics are averaged per pair, then per category, then across categories.

### Large vision-language models

The `VLMEvalKit/` integration contains the SOCO-LVLM evaluation for visual, visual-plus-description, and description-only queries. Its code and dependencies are kept separate from the minimal loader environment.

## Tests

```bash
uv sync --group test
uv run --group test pytest
```

## Citation and licenses

The code in this repository is MIT licensed.
SOCO is a separate CC BY 4.0 dataset.

Please cite the dataset paper when using it:

```bibtex
@article{duenkel2026soco,
  title         = {SOCO: Benchmarking Semantic Object Correspondence in Vision Foundation Models},
  author        = {D{\"u}nkel, Olaf and Sunagad, Basavaraj and Wang, Haoran and
                   Hoffmann, David T. and Theobalt, Christian and Kortylewski, Adam},
  journal       = {arXiv preprint arXiv:2605.31597},
  year          = {2026}
}
```
