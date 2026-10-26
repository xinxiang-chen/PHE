#!/bin/bash
# Hash code length sweep on a single multi-GPU machine (e.g. a RunPod pod), no Slurm.
# Configs (dataset x code length) are spread round-robin over the GPUs; each GPU runs
# its share one after another. Configs that already have checkpoints/checkpoint_last.pth
# are skipped, so re-running the script after a pod restart only runs what is left.
#
# Usage (from the PHE repo root, ideally inside tmux):
#   export PHE_DATA_ROOT=/workspace/datasets     # holds cub/, oxford_pets/, stanford_cars/, dino/
#   bash scripts/sweep_hash_len_runpod.sh
# Or detached:
#   nohup bash scripts/sweep_hash_len_runpod.sh > log/runpod/sweep.out 2>&1 &
# Optional overrides:
#   DATASETS="cub scars" LENGTHS="512 1024" GPUS="0 1" bash scripts/sweep_hash_len_runpod.sh
#
# Per-config logs: log/runpod/<dataset>_L<L>_seed<seed>.out
# Results:         exp/sweep_hash_len/<dataset>/<dataset>_L<L>_seed<seed>/

set -uo pipefail

DATASETS=(${DATASETS:-cub pets scars})
LENGTHS=(${LENGTHS:-128 256 512 1024})
GPUS=(${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')})
PYTHON=${PYTHON:-python}
NUM_WORKERS=${NUM_WORKERS:-8}

export PHE_DATA_ROOT=${PHE_DATA_ROOT:-/workspace/datasets}

# Same seeds as scripts/train_{cub,pets,scars}.sh
seed_of() {
    case $1 in
        pets) echo 1029 ;;
        *)    echo 1027 ;;
    esac
}

cd "$(dirname "$0")/.."
mkdir -p log/runpod

# Code length is the outer loop so that each GPU gets a mix of short and long codes
jobs=()
for L in "${LENGTHS[@]}"; do
    for data_set in "${DATASETS[@]}"; do
        jobs+=("$data_set $L")
    done
done

echo "PHE_DATA_ROOT=${PHE_DATA_ROOT}  GPUs=${GPUS[*]}  configs=${#jobs[@]}"

run_queue() {
    local gpu=$1; shift
    local data_set L seed name output_dir
    for job in "$@"; do
        read -r data_set L <<< "$job"
        seed=$(seed_of "$data_set")
        name="${data_set}_L${L}_seed${seed}"
        output_dir=exp/sweep_hash_len/$data_set/$name

        if [ -f "$output_dir/checkpoints/checkpoint_last.pth" ]; then
            echo "[gpu $gpu] skip $name (already finished)"
            continue
        fi

        echo "[gpu $gpu] start $name  $(date '+%F %T')"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u main.py \
            --data_set=$data_set \
            --hash_code_length=$L \
            --output_dir=$output_dir \
            --seed=$seed \
            --num_workers=$NUM_WORKERS \
            > "log/runpod/${name}.out" 2>&1
        echo "[gpu $gpu] done  $name  exit=$?  $(date '+%F %T')"
    done
}

for g in "${!GPUS[@]}"; do
    queue=()
    for (( i = g; i < ${#jobs[@]}; i += ${#GPUS[@]} )); do
        queue+=("${jobs[$i]}")
    done
    [ ${#queue[@]} -gt 0 ] && run_queue "${GPUS[$g]}" "${queue[@]}" &
done

wait
echo "All configs finished  $(date '+%F %T')"
