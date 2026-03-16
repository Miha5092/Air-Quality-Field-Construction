#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=a40
#SBATCH --job-name=Diffusion

export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH

source activate

srun python -m src.usage.diffusion_training -es -v -s --epochs 240 --experiment_name short_training