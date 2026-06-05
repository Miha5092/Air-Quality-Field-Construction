# Air Pollution Reconstruction

## Introduction
This repository contains scripts to train and evaluate deterministic and generative models for the reconstruction of air pollution field. This also contains code to create plots and results as shown in the paper

## Installation

### Requirements

...........

### Train models

...........

<img src="https://github.com/Miha5092/master-thesis/blob/03ddf9757b5387b79e9372f3b71ba5a951a5bf6c/assets/training.png" width="800">

Figure shows training method of various models using simulation data. Panel (a) illustrates the feature extraction step, Panel (b) illustrates the output generation by deterministic and generative models.

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
