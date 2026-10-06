#!/bin/bash
#SBATCH --job-name=gazerl_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ==============================================================================
# GazeRL SLURM Training Launcher
# ==============================================================================
# Usage:
#   sbatch scripts/sbatch/submit_train.sbatch configs/presets/gaze_bfh.yaml
#   sbatch scripts/sbatch/submit_train.sbatch configs/presets/gaze_shaping.yaml --wandb
# ==============================================================================

set -euo pipefail

# Activate Python environment
CONDA_ENV="${CONDA_ENV:-/data/teaywright/conda/envs/gaze}"
export PATH="${CONDA_ENV}/bin:${PATH}"

CONFIG_PATH="${1:-configs/presets/gaze_bfh.yaml}"
shift || true
EXTRA_ARGS="${*:-}"

mkdir -p logs outputs

echo "========================================================================"
echo "GazeRL Training Job ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "Host: $(hostname)"
echo "Config: ${CONFIG_PATH}"
echo "Extra Args: ${EXTRA_ARGS}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo 'No GPU detected')"
echo "========================================================================"

python scripts/train.py --config "${CONFIG_PATH}" ${EXTRA_ARGS}

echo "Training job completed at $(date)"
