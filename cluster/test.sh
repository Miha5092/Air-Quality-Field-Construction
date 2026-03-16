#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=gpgpu,gpgpuB,gpgpuM
#SBATCH --job-name=test

export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH

source activate

srun python -m experiments.timestep --model_type vunet --tested_timesteps 1