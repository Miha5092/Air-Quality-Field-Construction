import torch
import numpy as np
import torch.nn as nn
import os
import logging

from torch.utils.data import DataLoader

from src.datasets.real_obs_dataset import load_data as load_real_data
from src.datasets.vitae_dataset import unscale
from src.datasets.vitae_dataset import load_data as load_sparse_simulated
from src.datasets.voronoi_datasets import load_data as load_voronoi_simulated
from src.utils.evaluation import compute_relative_error, compute_rmse, compute_rrmse, compute_mean_fractional_error, compute_mean_fractional_bias, compute_SSIM
from src.models.diffusion import EvaluateDiffusionModel

@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_scaling_type: str,
    timesteps: int,
    experiment_name: str = None,
) -> None:
    
    results_on_simulated = evaluate_on_simulated(model, data_scaling_type, timesteps)
    results_on_real = evaluate_on_real(model, data_scaling_type, timesteps)
    
    # Decide where to save the results
    save_dirs_map = {
        "VCNN": "paper_results/predictions/vunet",
        "VUnet": "paper_results/predictions/vunet",
        "VCNN_classic": "paper_results/predictions/vcnn",
        "ConvLSTM": "paper_results/predictions/clstm",
        "OptimizedModule": "paper_results/predictions/clstm",
        "ViTAE": "paper_results/predictions/vitae",
        "Diffusion": "paper_results/predictions/diffusion"
    }

    # Decide where to save the results
    preds_dir = save_dirs_map.get(model.__class__.__name__, "paper_results/predictions/unknown_model")
    os.makedirs(preds_dir, exist_ok=True)
    preds_file = os.path.join(preds_dir, f"{experiment_name}_results.npz")

    np.savez_compressed(
        preds_file,

        # ------ Saving the results on the simulated data ------

        # Save the data used to compute the metrics
        simulated_observations=results_on_simulated["observations"],
        simulated_ground_truths=results_on_simulated["ground_truths"],
        simulated_predictions=results_on_simulated["predictions"],
        # Save the relative error
        simulated_global_re=results_on_simulated["global_re"],
        simulated_pollutants_re=results_on_simulated["pollutants_re"],
        # Save the RMSe
        simulated_global_rmse=results_on_simulated["global_rmse"],
        simulated_pollutants_rmse=results_on_simulated["pollutants_rmse"],
        # Save the MFE
        simulated_global_mfe=results_on_simulated["global_mfe"],
        simulated_pollutants_mfe=results_on_simulated["pollutants_mfe"],
        # Save the MFB
        simulated_global_mfb=results_on_simulated["global_mfb"],
        simulated_pollutants_mfb=results_on_simulated["pollutants_mfb"],
        # Save SSIM
        simulated_global_ssim = results_on_simulated["global_ssim"],
        simulated_pollutants_ssim = results_on_simulated["pollutants_ssim"],


        # ------ Saving the results on the real data ------

        # Save the data used to compute the metrics
        real_observations=results_on_real["observations"],
        real_ground_truths=results_on_real["ground_truths"],
        real_predictions=results_on_real["predictions"],
        real_target_masks=results_on_real["target_masks"],
        # Save the relative error
        real_global_re=results_on_real["global_re"],
        real_pollutants_re=results_on_real["pollutants_re"],
        # Save the RMSe
        real_global_rmse=results_on_real["global_rmse"],
        real_pollutants_rmse=results_on_real["pollutants_rmse"],
        # Save the MFE
        real_global_mfe=results_on_real["global_mfe"],
        real_pollutants_mfe=results_on_real["pollutants_mfe"],
        # Save the MFB
        real_global_mfb=results_on_real["global_mfb"],
        real_pollutants_mfb=results_on_real["pollutants_mfb"]
    )

    logging.info(f"Saved evaluation results to {preds_file}")

@torch.no_grad()
def ensemble_evaluate(
    model: nn.Module,
    data_scaling_type: str,
    timesteps: int,
    experiment_name: str = None,
) -> None:
    
    inf_size_list = [2,5,10,15,20,40]
    for k in inf_size_list:
        results_on_simulated = evaluate_on_diffusion(model, data_scaling_type, timesteps, inf_size=k)
        
        # Decide where to save the results
        save_dirs_map = {
            "Diffusion": "paper_results/predictions/diffusion/ensemble"
        }

        # Decide where to save the results
        preds_dir = save_dirs_map.get(model.__class__.__name__, "paper_results/predictions/unknown_model")
        os.makedirs(preds_dir, exist_ok=True)
        preds_file = os.path.join(preds_dir, "inf_"+str(inf_size_list[k])+"_results.npz")

        np.savez_compressed(
            preds_file,

            # ------ Saving the results on the simulated data ------
            # Save the relative error
            simulated_global_re=results_on_simulated["global_re"],
            simulated_pollutants_re=results_on_simulated["pollutants_re"],
            # Save the RMSe
            simulated_global_rmse=results_on_simulated["global_rmse"],
            simulated_pollutants_rmse=results_on_simulated["pollutants_rmse"],
            # Save the MFE
            simulated_global_mfe=results_on_simulated["global_mfe"],
            simulated_pollutants_mfe=results_on_simulated["pollutants_mfe"],
            # Save the MFB
            simulated_global_mfb=results_on_simulated["global_mfb"],
            simulated_pollutants_mfb=results_on_simulated["pollutants_mfb"],
            # Save SSIM
            simulated_global_ssim = results_on_simulated["global_ssim"],
            simulated_pollutants_ssim = results_on_simulated["pollutants_ssim"],


        )

        logging.info(f"Saved evaluation results to {preds_file}")


model_names_map = {
    "VCNN": "vunet",
    "VUnet": "vunet",
    "VCNN_classic": "vcnn",
    "ConvLSTM": "clstm",
    "OptimizedModule": "clstm",
    "ViTAE": "vitae",
    "Diffusion": "diffusion"
}

@torch.no_grad()
def evaluate_on_simulated(
    model: nn.Module,
    data_scaling_type: str,
    timesteps: int,
) -> dict:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_type = model_names_map.get(model.__class__.__name__)
    
    if model_type == "vitae":
        _, _, dataset, stats = load_sparse_simulated(sensor_type="real-random", scaling_type=data_scaling_type, timesteps=timesteps)
    elif model_type == "clstm":
        _, _, dataset, stats = load_voronoi_simulated(sensor_type="real-random", scaling_type=data_scaling_type, timesteps=timesteps, channel_timesteps=False)
    elif model_type == "diffusion":
        _, _, dataset, stats = load_voronoi_simulated(sensor_type="real-random", scaling_type=data_scaling_type, timesteps=timesteps, channel_timesteps=True, diffusion=True)
    else:
        _, _, dataset, stats = load_voronoi_simulated(sensor_type="real-random", scaling_type=data_scaling_type, timesteps=timesteps, channel_timesteps=True)

    dataloader = DataLoader(
        dataset,
        batch_size=64 if model_type != "clstm" else 32,
        shuffle=False
    )
    if model_type == "diffusion":
        model_eval = EvaluateDiffusionModel(model.denoiser_model, model.cond_model, device=device)
    else:
        model.eval()
        model.to(device)

    observations, ground_truths, predictions = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            obs = batch[0]
            ground_truth = batch[1]
            if model_type=="diffusion":
                mask = batch[2]
                mask = mask.to(device)

            obs = obs.to(device).float()

            if model_type=="diffusion":
                preds = model_eval.ensemble_prediction(ground_truth, mask, obs, num_inf_steps=10, num_ensem_steps=20, device=device)
            else:
                preds = model(obs)

            # If we are working with the ConvLSTM model, we need to take the last timestep
            if model_type == "clstm":
                preds = preds[:, -1]
                ground_truth = ground_truth[:, -1]

            # The ViTAE model outputs both the encoder and decoder predictions, we only need the decoder predictions
            if model_type == "vitae":
                preds = preds[-1]

            # TODO: Add any required post-processing for the other models here

            # The diffusion model outputs all the samples from ensemble [1,E]. Choose the last sample
            if model_type=="diffusion":
                preds = preds[-1]

            # Scale the data back to its original values
            if data_scaling_type == "min-max":
                channel_count = stats["data_min"].shape[1] if "data_min" in stats else stats["Y_min"].shape[1]
                obs_shape = obs.shape
            else:
                channel_count = stats["data_mean"].shape[1] if "data_mean" in stats else stats["Y_mean"].shape[1]
                obs_shape = obs.shape

            stats = _convert_stats(stats)

            obs = unscale(obs.reshape(-1, channel_count, obs.shape[-2], obs.shape[-1]).cpu().detach().numpy(), data_scaling_type, **stats).reshape(*obs_shape)
            ground_truth = unscale(ground_truth.float().numpy(), data_scaling_type, **stats)
            preds = unscale(preds.cpu().detach().numpy(), data_scaling_type, **stats)

            # Store the observations, ground truth, target masks, and predictions
            observations.append(obs)
            ground_truths.append(ground_truth)
            predictions.append(preds)

    observations = np.concatenate(observations).astype(np.float32)
    ground_truths = torch.from_numpy(np.concatenate(ground_truths).astype(np.float32))
    predictions = torch.from_numpy(np.concatenate(predictions).astype(np.float32))

    # ------- Compute the metrics -------

    # These metrics are computed globally (all pollutants together)
    global_re = np.mean(compute_relative_error(ground_truths, predictions))
    global_rmse = compute_rmse(ground_truths, predictions, None)
    global_rrmse = compute_rrmse(ground_truths, predictions, None)
    global_mfe = compute_mean_fractional_error(ground_truths, predictions, None)
    global_mfb = compute_mean_fractional_bias(ground_truths, predictions, None)
    global_ssim = np.mean(compute_SSIM(ground_truths, predictions))

    # These metrics are computed for each pollutant separately
    pollutants_re, pollutants_rmse, pollutants_rrmse, pollutants_mfe, pollutants_mfb, pollutants_ssim = [], [], [], [], [], []
    for i in range(4):
        pollutant_ground_truths = ground_truths[:, i]
        pollutant_predictions = predictions[:, i]

        pollutant_re = np.mean(compute_relative_error(pollutant_ground_truths, pollutant_predictions))    
        pollutant_rmse = compute_rmse(pollutant_ground_truths, pollutant_predictions, None)
        pollutant_rrmse = compute_rrmse(pollutant_ground_truths, pollutant_predictions, None)
        pollutant_mfe = compute_mean_fractional_error(pollutant_ground_truths, pollutant_predictions, None)
        pollutant_mfb = compute_mean_fractional_bias(pollutant_ground_truths, pollutant_predictions, None)
        pollutants_ssim = np.mean(compute_SSIM(ground_truths, pollutant_predictions))

        pollutants_re.append(pollutant_re)
        pollutants_rmse.append(pollutant_rmse)
        pollutants_rrmse.append(pollutant_rrmse)
        pollutants_mfe.append(pollutant_mfe)
        pollutants_mfb.append(pollutant_mfb)
        pollutants_ssim.append(pollutants_ssim)

    return {
        # Save the data used to compute the metrics
        "observations": observations,
        "ground_truths": ground_truths.numpy(),
        "predictions": predictions.numpy(),
        # Save the relative error
        "global_re": global_re,
        "pollutants_re": pollutants_re,
        # Save the RMSE
        "global_rmse": global_rmse,
        "pollutants_rmse": pollutants_rmse,
        # Save the RRMSE
        "global_rrmse": global_rrmse,
        "pollutants_rrmse": pollutants_rrmse,
        # Save the MFE
        "global_mfe": global_mfe,
        "pollutants_mfe": pollutants_mfe,
        # Save the MFB
        "global_mfb": global_mfb,
        "pollutants_mfb": pollutants_mfb,
        # Save the SSIM
        "global_ssim": global_ssim,
        "pollutant_ssim": pollutants_ssim,
    }

@torch.no_grad()
def evaluate_on_real(
    model: nn.Module,
    data_scaling_type: str,
    timesteps: int,
) -> dict:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_type = model_names_map.get(model.__class__.__name__)
    
    dataset, stats = load_real_data(model_type=model_type, sensor_type="real-random", timesteps=timesteps, val_set=False)
    dataloader = DataLoader(
        dataset,
        batch_size=64 if model_type != "clstm" else 32,
        shuffle=False
    )
    if model_type == "diffusion":
        model_eval = EvaluateDiffusionModel(model.denoiser_model, model.cond_model, device=device)
    else:
        model.eval()
        model.to(device)

    observations, ground_truths, predictions, target_masks = [], [], [], []

    with torch.no_grad():
        for obs, ground_truth, target_mask in dataloader:
            if model_type == "diffusion":
                obs_mask = obs[1].to(device).float()
                obs = obs[0].to(device).float()
                preds = model_eval.ensemble_prediction(obs*obs_mask, obs_mask, obs, num_inf_steps=10, num_ensem_steps=20, device=device)
            else:
                obs = obs.to(device).float()
                preds = model(obs)

            # If we are working with the ConvLSTM model, we need to take the last timestep
            if model_type == "clstm":
                preds = preds[:, -1]

            # The ViTAE model outputs both the encoder and decoder predictions, we only need the decoder predictions
            if model_type == "vitae":
                preds = preds[-1]

            # The diffusion model outputs all the samples from ensemble [1,E]. Choose the last sample
            if model_type=="diffusion":
                preds = preds[-1]

            # Scale the data back to its original values
            if model_type == "diffusion":
                channel_count = stats["data_std"].shape[1]
            else:
                channel_count = stats["data_min"].shape[1]
            obs_shape = obs.shape

            obs = unscale(obs.reshape(-1, channel_count, obs.shape[-2], obs.shape[-1]).cpu().detach().numpy(), data_scaling_type, **stats).reshape(*obs_shape)
            ground_truth = unscale(ground_truth.float().numpy(), data_scaling_type, **stats)
            target_mask = target_mask.float().numpy()
            preds = unscale(preds.cpu().detach().numpy(), data_scaling_type, **stats)

            # Store the observations, ground truth, target masks, and predictions
            observations.append(obs)
            ground_truths.append(ground_truth)
            target_masks.append(target_mask)
            predictions.append(preds)

    observations = np.concatenate(observations).astype(np.float32)
    ground_truths = torch.from_numpy(np.concatenate(ground_truths).astype(np.float32))
    target_masks = torch.from_numpy(np.concatenate(target_masks).astype(np.float32))
    predictions = torch.from_numpy(np.concatenate(predictions).astype(np.float32))

    # ------- Compute the metrics -------

    # These metrics are computed globally (all pollutants together)
    global_re = np.mean(compute_relative_error(ground_truths * target_masks, predictions * target_masks))
    global_rmse = compute_rmse(ground_truths, predictions, target_masks)
    global_rrmse = compute_rrmse(ground_truths, predictions, target_masks)
    global_mfe = compute_mean_fractional_error(ground_truths, predictions, target_masks)
    global_mfb = compute_mean_fractional_bias(ground_truths, predictions, target_masks)

    # These metrics are computed for each pollutant separately
    pollutants_re, pollutants_rmse, pollutants_rrmse, pollutants_mfe, pollutants_mfb = [], [], [], [], []
    for i in range(4):
        pollutant_ground_truths = ground_truths[:, i]
        pollutant_predictions = predictions[:, i]
        pollutant_target_masks = target_masks[:, i]

        pollutant_re = np.mean(compute_relative_error(pollutant_ground_truths * pollutant_target_masks, pollutant_predictions * pollutant_target_masks))    
        pollutant_rmse = compute_rmse(pollutant_ground_truths, pollutant_predictions, pollutant_target_masks)
        pollutant_rrmse = compute_rrmse(pollutant_ground_truths, pollutant_predictions, pollutant_target_masks)
        pollutant_mfe = compute_mean_fractional_error(pollutant_ground_truths, pollutant_predictions, pollutant_target_masks)
        pollutant_mfb = compute_mean_fractional_bias(pollutant_ground_truths, pollutant_predictions, pollutant_target_masks)

        pollutants_re.append(pollutant_re)
        pollutants_rmse.append(pollutant_rmse)
        pollutants_rrmse.append(pollutant_rrmse)
        pollutants_mfe.append(pollutant_mfe)
        pollutants_mfb.append(pollutant_mfb)

    return {
        # Save the data used to compute the metrics
        "observations": observations,
        "ground_truths": ground_truths.numpy(),
        "predictions": predictions.numpy(),
        "target_masks": target_masks.numpy(),
        # Save the relative error
        "global_re": global_re,
        "pollutants_re": pollutants_re,
        # Save the RMSE
        "global_rmse": global_rmse,
        "pollutants_rmse": pollutants_rmse,
        # Save the RRMSE
        "global_rrmse": global_rrmse,
        "pollutants_rrmse": pollutants_rrmse,
        # Save the MFE
        "global_mfe": global_mfe,
        "pollutants_mfe": pollutants_mfe,
        # Save the MFB
        "global_mfb": global_mfb,
        "pollutants_mfb": pollutants_mfb
    }

@torch.no_grad()
def evaluate_on_diffusion(model: nn.Module,
    data_scaling_type: str,
    timesteps: int,
    inf_size: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_type = model_names_map.get(model.__class__.__name__)
    
    if model_type == "diffusion":
        _, _, dataset, stats = load_voronoi_simulated(sensor_type="real-random", scaling_type=data_scaling_type, timesteps=timesteps, channel_timesteps=True, diffusion=True)

    dataloader = DataLoader(
        dataset,
        batch_size=64 if model_type != "clstm" else 32,
        shuffle=False
    )
    model_eval = EvaluateDiffusionModel(model.denoiser_model, model.cond_model, device=device)

    observations, ground_truths = [], []
    predictions = {}
    for k in range(20):
        predictions[k] = []

    with torch.no_grad():
        for batch in dataloader:
            obs = batch[0]
            ground_truth = batch[1]
            mask = batch[2]
            mask = mask.to(device)

            obs = obs.to(device).float()

            # The diffusion model outputs all the samples from ensemble [1,E]
            preds = model_eval.ensemble_prediction(ground_truth, mask, obs, num_inf_steps=inf_size, num_ensem_steps=20, device=device)

            # Scale the data back to its original values
            if data_scaling_type == "min-max":
                channel_count = stats["data_min"].shape[1] if "data_min" in stats else stats["Y_min"].shape[1]
                obs_shape = obs.shape
            else:
                channel_count = stats["data_mean"].shape[1] if "data_mean" in stats else stats["Y_mean"].shape[1]
                obs_shape = obs.shape

            stats = _convert_stats(stats)

            obs = unscale(obs.reshape(-1, channel_count, obs.shape[-2], obs.shape[-1]).cpu().detach().numpy(), data_scaling_type, **stats).reshape(*obs_shape)
            ground_truth = unscale(ground_truth.float().numpy(), data_scaling_type, **stats)
            preds = unscale(preds.cpu().detach().numpy(), data_scaling_type, **stats)

            for k in range(20):
                preds[k] = unscale(preds[k].cpu().detach().numpy(), data_scaling_type, **stats)
                predictions[k].append(preds[k])

            # Store the observations, ground truth, target masks, and predictions
            observations.append(obs)
            ground_truths.append(ground_truth)

    observations = np.concatenate(observations).astype(np.float32)
    ground_truths = torch.from_numpy(np.concatenate(ground_truths).astype(np.float32))

    # ------- Compute the metrics -------
    global_re, global_rmse, global_rrmse, global_mfe, global_mfb, global_ssim = [], [], [], [], [], []

    for k in range(20):
        predictions[k] = torch.from_numpy(np.concatenate(predictions[k]).astype(np.float32))

        global_re = np.mean(compute_relative_error(ground_truths, predictions[k]))
        global_rmse = compute_rmse(ground_truths, predictions[k], None)
        global_rrmse = compute_rrmse(ground_truths, predictions[k], None)
        global_mfe = compute_mean_fractional_error(ground_truths, predictions[k], None)
        global_mfb = compute_mean_fractional_bias(ground_truths, predictions[k], None)
        global_ssim = np.mean(compute_SSIM(ground_truths, predictions[k]))

        global_re.append(global_re)
        global_rmse.append(global_rmse)
        global_rrmse.append(global_rrmse)
        global_mfe.append(global_mfe)
        global_mfb.append(global_mfb)
        global_ssim.append(global_ssim)

    # These metrics are computed for each pollutant separately
    pollutants_re, pollutants_rmse, pollutants_rrmse, pollutants_mfe, pollutants_mfb, pollutants_ssim = [], [], [], [], [], []
    for k in range(20):
        pollutants_re_k = []
        pollutants_rmse_k = []
        pollutants_rrmse_k = []
        pollutants_mfe_k = []
        pollutants_mfb_k = []
        pollutants_ssim_k = []
        for i in range(4):
            pollutant_ground_truths = ground_truths[:, i]
            pollutant_predictions = predictions[k][:, i]

            pollutant_re_k = np.mean(compute_relative_error(pollutant_ground_truths, pollutant_predictions))    
            pollutant_rmse_k = compute_rmse(pollutant_ground_truths, pollutant_predictions, None)
            pollutant_rrmse_k = compute_rrmse(pollutant_ground_truths, pollutant_predictions, None)
            pollutant_mfe_k = compute_mean_fractional_error(pollutant_ground_truths, pollutant_predictions, None)
            pollutant_mfb_k = compute_mean_fractional_bias(pollutant_ground_truths, pollutant_predictions, None)
            pollutant_ssim_k = np.mean(compute_SSIM(pollutant_ground_truths, pollutant_predictions))   

            pollutants_re_k.append(pollutant_re_k)
            pollutants_rmse_k.append(pollutant_rmse_k)
            pollutants_rrmse_k.append(pollutant_rrmse_k)
            pollutants_mfe_k.append(pollutant_mfe_k)
            pollutants_mfb_k.append(pollutant_mfb_k)
            pollutants_ssim_k.append(pollutant_ssim_k)

        pollutants_re.append(pollutants_re_k)
        pollutants_rmse.append(pollutants_rmse_k)
        pollutants_rrmse.append(pollutants_rrmse_k)
        pollutants_mfe.append(pollutants_mfe_k)
        pollutants_mfb.append(pollutants_mfb_k)
        pollutants_ssim.append(pollutants_ssim_k)

    return {
        # Save the data used to compute the metrics
        # Save the relative error
        "global_re": global_re,
        "pollutants_re": pollutants_re,
        # Save the RMSE
        "global_rmse": global_rmse,
        "pollutants_rmse": pollutants_rmse,
        # Save the RRMSE
        "global_rrmse": global_rrmse,
        "pollutants_rrmse": pollutants_rrmse,
        # Save the MFE
        "global_mfe": global_mfe,
        "pollutants_mfe": pollutants_mfe,
        # Save the MFB
        "global_mfb": global_mfb,
        "pollutants_mfb": pollutants_mfb,
        # Save SSIM
        "global_ssim": global_ssim,
        "pollutants_ssim": pollutants_ssim
    }

def _convert_stats(stats: dict[str, any]) -> dict:
    """
    Convert stats values from lists to numpy arrays.
    """
    converted_stats = {}
    for key, value in stats.items():
        if key.startswith("Y"):
            key = key.replace("Y", "data")
        
        converted_stats[key] = value

    return converted_stats