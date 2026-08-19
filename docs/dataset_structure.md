# Dataset structure

SOCOv1 is distributed as three archives plus unpacked metadata. After extraction,
the expected directory tree is:

```text
SOCOv1/
  Images/<category>/*.JPEG
  KeypointAnnotations/<category>/*.json
  PairAnnotations/
    intra/<category>/*.json
    cross/<pair-group>/*.json
    trainsplits/
      train/<category>/*.json
      test/<category>/*.json
  Metadata/
    filename_mapping.json
    keypoint_taxonomy.json
```

`intra` contains within-category pairs. `cross` contains inter-category pairs and
is called Cross-SOC in the paper. Train/test annotations are predefined subsets of
the intra-category data for learned probes; the nearest-neighbor example does not
train a probe.

## Pair annotations

Each pair JSON file is self-contained. Its main fields are:

- `pair_id`, `pair_group`: pair identity and grouping used for aggregation.
- `src_category`, `trg_category`, `src_image`, `trg_image`: image references.
- `src_imsize`, `trg_imsize`: `[width, height]` in pixels.
- `src_bndbox`, `trg_bndbox`: object boxes `[left, top, right, bottom]`.
- `src_keypoints`, `trg_keypoints`: visible points and their coordinates.
- `semantic_overlap`: names present in both views with the same object-relative
  identity.
- `concept_matches`: groups of source and target keypoint names representing all
  valid matches for one local semantic concept.

A keypoint includes an integer `id`, unique `name`, position `pos = [x, y]`, concept
identifier/name, structural or functional `kind`, and whether it is shared across
categories. Coordinates use the original, unpadded image coordinate system.

## Match terminology

- **Concept correspondence (CC):** match the same local concept. A source point may
  have multiple correct target points, such as any visible wheel center.
- **Semantic object correspondence (SOC):** match the exact semantic keypoint,
  including object-relative identity such as front-left versus rear-right.
- **Cross-SOC:** apply exact semantic matching across related categories.

The loader represents both forms with the same `Match` type. Its `source_indices`
and `target_indices` tuples contain one index each for SOC and one or more indices
for CC. `label` retains the semantic keypoint or concept name, so matches remain
auditable against the JSON files.

Evaluation JSON stores PCK values as percentages on a 0–100 scale.

## Metadata

`keypoint_taxonomy.json` describes category, concept, keypoint, inheritance, and
language-template metadata. `filename_mapping.json` maps source dataset filenames to
the normalized SOCO filenames. Use `load_keypoint_taxonomy` and
`load_filename_mapping` to read them.

The canonical dataset card, download, license, and citation are maintained at
<https://huggingface.co/datasets/GenIntelLab/SOCO>.
