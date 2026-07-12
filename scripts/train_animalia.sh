#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

data_set=Animalia
seed=1027
output_dir=exp/

python main.py \
    --data_set=$data_set \
    --output_dir=$output_dir/$data_set/"animalia_seed($seed)" \
    --seed=$seed \
