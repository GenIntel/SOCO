#!/usr/bin/env bash


DATASETS=("soco_lvlm_img_c" 
        "soco_lvlm_imgtxt_c" 
        "soco_lvlm_txt_c")

MODEL=Qwen3-VL-4B-Instruct

for DATASET in "${DATASETS[@]}"; do
    echo "Evaluating dataset: $DATASET with $MODEL"
    CUDA_VISIBLE_DEVICES=0 python run.py --data "$DATASET" --model $MODEL --mode infer
done

python evaluate_soco_c.py --base ./outputs
