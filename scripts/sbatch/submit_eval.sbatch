#!/bin/bash
#SBATCH --job-name=gazerl_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# ==============================================================================
# GazeRL SLURM Evaluation Launcher
# ==============================================================================
# Usage:
#   sbatch scripts/sbatch/submit_eval.sbatch outputs/checkpoint_final coco_2014 val
# ==============================================================================

set -euo pipefail

# Activate Python environment
CONDA_ENV="${CONDA_ENV:-/data/teaywright/conda/envs/gaze}"
export PATH="${CONDA_ENV}/bin:${PATH}"

CHECKPOINT="${1:-outputs/checkpoint_final}"
DATASET="${2:-coco_2014}"
SPLIT="${3:-val}"
shift 3 || true
EXTRA_ARGS="${*:-}"

mkdir -p logs

echo "========================================================================"
echo "GazeRL Evaluation Job ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset: ${DATASET} (${SPLIT})"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo 'No GPU detected')"
echo "========================================================================"

python scripts/eval.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    ${EXTRA_ARGS}

echo "Evaluation job completed at $(date)"
