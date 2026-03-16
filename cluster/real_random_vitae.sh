#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=a40
#SBATCH --job-name=Vitae_noised

export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH

source activate

srun python -m experiments.real_dataset_evaluation --model_types vitae --epochs 300 --experiment_name gaussian --noise gaussian --full_noise

srun python -m experiments.real_dataset_evaluation --model_types vitae --epochs 300 --experiment_name time_gaussian --noise time_gaussian --full_noise

srun python -m experiments.real_dataset_evaluation --model_types vitae --epochs 300 --experiment_name perlin --noise perlin --full_noise

srun python -m experiments.real_dataset_evaluation --model_types vitae --epochs 300 --experiment_name correlated --noise correlated --full_noise

# ["gaussian", "time_gaussian", "perlin", "correlated"]