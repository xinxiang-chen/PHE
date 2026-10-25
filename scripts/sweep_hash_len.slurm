#!/bin/bash
# Hash code length sweep on CUB and Pets (CARC Discovery, Slurm job array).
# 8 tasks = 2 datasets x 4 code lengths, one GPU each.
#
# Submit from the PHE repo root:
#   mkdir -p log/slurm
#   sbatch --account=<project_id> scripts/sweep_hash_len.slurm
# Run a subset, e.g. only Pets:
#   sbatch --account=<project_id> --array=4-7 scripts/sweep_hash_len.slurm
#
#SBATCH --job-name=phe_hashlen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0-7
#SBATCH --output=log/slurm/%x_%A_%a.out
#SBATCH --error=log/slurm/%x_%A_%a.out

set -euo pipefail

DATASETS=(cub pets)
SEEDS=(1027 1029)    # same seeds as scripts/train_cub.sh and scripts/train_pets.sh
LENGTHS=(128 256 512 1024)

idx=${SLURM_ARRAY_TASK_ID}
ds_idx=$(( idx / ${#LENGTHS[@]} ))
data_set=${DATASETS[$ds_idx]}
seed=${SEEDS[$ds_idx]}
L=${LENGTHS[$(( idx % ${#LENGTHS[@]} ))]}

# Directory holding cub/, oxford_pets/ and dino/dino_vitbase16_pretrain.pth (see config.py)
export PHE_DATA_ROOT=${PHE_DATA_ROOT:-$HOME/Datasets}

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate phe

cd "${SLURM_SUBMIT_DIR}"
output_dir=exp/sweep_hash_len/$data_set/"${data_set}_L${L}_seed${seed}"

echo "task=${idx} data_set=${data_set} L=${L} seed=${seed} node=$(hostname)"
echo "output_dir=${output_dir}  PHE_DATA_ROOT=${PHE_DATA_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv

python main.py \
    --data_set=$data_set \
    --hash_code_length=$L \
    --output_dir=$output_dir \
    --seed=$seed \
    --num_workers=8
