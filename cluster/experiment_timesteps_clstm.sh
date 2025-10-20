#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=a40
#SBATCH --job-name=CLSTM_timesteps

# ---- Logging configuration ----
unset TORCH_LOGS  # avoids conflicts
export TORCHINDUCTOR_LOG_LEVEL=ERROR  # hide autotune noise, keep real errors
export TORCHINDUCTOR_VERBOSE=0        # make sure inductor doesn’t print debug info

export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH
source activate

python -m experiments.timestep --model_type clstm --tested_timesteps 3 4 6


# --tested_timesteps 1 12 

# --tested_timesteps 2 8

# --tested_timesteps 3 4 6