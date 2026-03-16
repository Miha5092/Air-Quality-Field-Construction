#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=a40
#SBATCH --job-name=Diffusion_time_gaussian

export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH

source activate

srun python -m src.usage.diffusion_training --epochs 240 --experiment_name time_gaussian --noise time_gaussian

# srun python -m src.usage.diffusion_training --epochs 240 --experiment_name gaussian --noise gaussian

# srun python -m src.usage.diffusion_training --epochs 240 --experiment_name perlin --noise perlin

# srun python -m src.usage.diffusion_training --epochs 240 --experiment_name correlated --noise correlated

# ["gaussian", "time_gaussian", "perlin", "correlated"]