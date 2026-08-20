# SOCO-LVLM Evaluation

This directory contains the LVLM evaluation code for
ECCV 2026 Spotlight paper [SOCO: Benchmarking Semantic Object Correspondence in Vision Foundation
Models](https://genintel.github.io/SOCO/).


SOCO-LVLM tests whether a model can find the target-image keypoint that
semantically corresponds to a visual or textual query.

This code extends
[VLMEvalKit at commit `2e35a57`](https://github.com/open-compass/VLMEvalKit/tree/2e35a5746119ef1cb99dca2109b5be6c7a293817)
with the SOCO-LVLM datasets and scoring script.

## Evaluation Settings

The target image and candidate keypoints are shown in every setting.

| Setting | Dataset | Query |
| --- | --- | --- |
| **Vis.** | `soco_lvlm_img_c` | A keypoint marked in a source image |
| **Vis.+Desc.** | `soco_lvlm_imgtxt_c` | The marked source keypoint and its description |
| **Desc.** | `soco_lvlm_txt_c` | The keypoint description only |

## Quick Start

### 1. Install

Python 3.10 is recommended. Local models require a compatible GPU and CUDA
environment.

```bash
conda create -n soco_lvlm_eval python=3.10 -y
conda activate soco_lvlm_eval

cd SOCO/VLMEvalKit
pip install -e .
```

### 2. Select a model

Edit `MODEL` in [`eval_soco_c.sh`](eval_soco_c.sh):

```bash
MODEL=Qwen3-VL-4B-Instruct
```

Supported model names are defined in [`vlmeval/config.py`](vlmeval/config.py).
You can also edit `DATASETS` in `eval_soco_c.sh` to evaluate only a subset of
the three settings, or change `CUDA_VISIBLE_DEVICES=0` to select another GPU.

### 3. Run

```bash
bash eval_soco_c.sh
```

The script runs inference for the selected settings and then computes the
SOCO-LVLM metrics. Dataset files are downloaded automatically from
[GenIntelLab/SOCO-LVLM](https://huggingface.co/datasets/GenIntelLab/SOCO-LVLM)
on first use.

## Results

Model predictions are stored under:

```text
outputs/<model>/<timestamp-and-commit>/<model>_<dataset>.xlsx
```

The scoring script uses the latest workbook for each model and setting and
reports:

- **Average accuracy:** accuracy over all individual questions.
- **Circular accuracy:** accuracy over question groups; a group is correct only
  if every circularized candidate ordering is answered correctly.

Reports are written to:

```text
outputs/soco_lvlm_c.md
outputs/soco_lvlm_c.csv
```

## Citation

If you use this benchmark or evaluation code, please cite SOCO and VLMEvalKit:

```bibtex
@article{duenkel2026soco,
  title         = {SOCO: Benchmarking Semantic Object Correspondence in Vision Foundation Models},
  author        = {D{\"u}nkel, Olaf and Sunagad, Basavaraj and Wang, Haoran and
                   Hoffmann, David T. and Theobalt, Christian and Kortylewski, Adam},
  journal       = {arXiv preprint arXiv:2605.31597},
  year          = {2026}
}

@inproceedings{duan2024vlmevalkit,
  title     = {VLMEvalKit: An Open-Source Toolkit for Evaluating Large Multi-Modality Models},
  author    = {Duan, Haodong and Yang, Junming and Qiao, Yuxuan and Fang, Xinyu and
               Chen, Lin and Liu, Yuan and Dong, Xiaoyi and Zang, Yuhang and
               Zhang, Pan and Wang, Jiaqi and others},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
  pages     = {11198--11201},
  year      = {2024}
}
```
