# Air Pollution Reconstruction

## Introduction
This repository contains scripts to train and evaluate deterministic and generative models for the reconstruction of air pollution field. This also contains code to create plots and results as shown in the paper

## Installation

### Requirements

The libraries and their versions used throughout this project can be found in [requirements.txt](requirements.txt). All experiments were run on a Linux computer with Python 3.

### Training and Evaluation

The main training loops are defined in the [src/usage/](src/usage/) directory. There you can also find the hyperparameter loops used to tune the models.

The experimental setups we presented in the paper can be found in the [experiments/](experiments/) directory. There you can find:

 - [fine_tuning.py](experiments/fine_tuning.py) - a generalized hyperparameter tunning setup.
 - [noise_magnitude.py](experiments/noise_magnitude.py) - the setup used to find the optimal noise parameters for data augmentation for training.
 - [random.py](experiments/random.py) - the setup to evaluate performance for random sensor placement on the synthethic dataset.
 - [real_dataset_evaluation.py](experiments/real_dataset_evaluation.py) - the setup to evaluate the performance on the real-world dataset.
 - [timestep.py](experiments/timestep.py) - the setup to evaluate the impact of different time horizons.

**To train a model on the synthetic dataset and evaluate it on the real-world dataset** you can use the following command:

```bash
python -m experiments.real_dataset_evaluation --experiment_name <your_name> --model_types <list_models> --epochs <n_epochs> --noise <noise_type>
```

## Architecture and Training Overview

<img src="https://github.com/Miha5092/master-thesis/blob/03ddf9757b5387b79e9372f3b71ba5a951a5bf6c/assets/training.png" width="800">

Figure shows training method of various models using simulation data. Panel (a) illustrates the feature extraction step, Panel (b) illustrates the output generation by deterministic and generative models.

## Results

<table>
  <tr>
    <!-- Top-left (2 gifs) -->
    <td>
      <img src="https://github.com/Miha5092/master-thesis/blob/5a86e0394894c9106263c6bb8656800e48217cdd/assets/animation_NO2.gif" width="250" />
      <img src="https://github.com/Miha5092/master-thesis/blob/5a86e0394894c9106263c6bb8656800e48217cdd/assets/0_backsample.gif" width="220" />
    </td>
    <td width="50%">
      <img src="https://github.com/Miha5092/master-thesis/blob/5a86e0394894c9106263c6bb8656800e48217cdd/assets/animation_O3.gif" width="250" />
      <img src="https://github.com/Miha5092/master-thesis/blob/5a86e0394894c9106263c6bb8656800e48217cdd/assets/1_backsample.gif" width="220" />
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/Miha5092/master-thesis/blob/5a86e0394894c9106263c6bb8656800e48217cdd/assets/animation_PM10.gif" width="250" />
      <img src="https://github.com/Miha5092/master-thesis/blob/5a86e0394894c9106263c6bb8656800e48217cdd/assets/2_backsample.gif" width="220" />
    </td>
    <td>
      <img src="https://github.com/Miha5092/master-thesis/blob/5a86e0394894c9106263c6bb8656800e48217cdd/assets/animation_PM25.gif" width="250" />
      <img src="https://github.com/Miha5092/master-thesis/blob/5a86e0394894c9106263c6bb8656800e48217cdd/assets/3_backsample.gif" width="220" />
    </td>
  </tr>
</table>
Plots GIFs for NO2, O3, PM10, PM2.5 pollutants showing the sampling procedure in diffusion process with the corresponding PSD plot for all intermediate steps
