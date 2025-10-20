#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=a40
#SBATCH --job-name=ViT_timesteps

export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH

source activate

python -m experiments.timestep --model_type vitae --tested_timesteps 3 4 6