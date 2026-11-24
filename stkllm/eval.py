import json
import os

import numpy as np
import torch

from stkllm.batch import unpack_batch
from stkllm.metrics import compute_horizon_metrics_report, masked_mape_np

def move_batch(batch, device):
    features, labels, prompts = unpack_batch(batch)
    return features.to(device), labels.to(device), prompts

def batch_loss(prediction, labels, loss_name, missing_value):
    name = str(loss_name).strip().lower()
    if name in ("masked_mae", "mae_masked"):
        from stkllm.metrics import masked_mae
        return masked_mae(prediction, labels, missing_value)
    if name in ("masked_mse", "mse"):
        return torch.nn.functional.mse_loss(prediction, labels)
    if name == "rmse":
        return torch.nn.functional.mse_loss(prediction, labels)
    return torch.nn.functional.l1_loss(prediction, labels)

def average_loss(model, loader, device, predict, loss_name, missing_value):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            features, labels, prompts = move_batch(batch, device)
            prediction = predict(features, prompts)
            losses.append(float(batch_loss(prediction, labels, loss_name, missing_value).detach()))
    model.train()
    if not losses:
        return None
    return float(sum(losses) / len(losses))

def epoch_metrics(model, loader, device, predict, metric_method, missing_value):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            features, labels, prompts = move_batch(batch, device)
            predictions.append(predict(features, prompts).detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())
    model.train()
    if not predictions:
        return None
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    if metric_method == "mask":
        from stkllm.metrics import masked_mae, masked_rmse
        return {
            "mae": float(masked_mae(torch.tensor(prediction), torch.tensor(target), missing_value)),
            "rmse": float(masked_rmse(torch.tensor(prediction), torch.tensor(target), missing_value)),
            "mape": float(masked_mape_np(target, prediction, missing_value)),
        }
    error = prediction - target
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(masked_mape_np(target, prediction, 0)),
    }

def save_predictions(model, loader, target_tensor, device, predict, metric_method, mean, std, out_dir, tag):
    model.eval()
    predictions, inputs = [], []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            features, _, prompts = move_batch(batch, device)
            inputs.append(features[:, :, 0:1].cpu().numpy())
            predictions.append(predict(features, prompts).detach().cpu().numpy())
    model.train()
    prediction = np.concatenate(predictions, axis=0)
    restored_input = np.concatenate(inputs, axis=0) * std + mean
    target = target_tensor.cpu().numpy()
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "output_%s.npz" % tag),
        input=restored_input,
        prediction=prediction,
        data_target_tensor=target,
    )
    report = compute_horizon_metrics_report(prediction, target, metric_method, 0.0)
    report_path = os.path.join(out_dir, "metrics_%s.json" % tag)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    overall = report["overall"]
    print(
        "MAE %.6f RMSE %.6f MAPE %.6f"
        % (overall["mae"], overall["rmse"], overall["mape"])
    )
    return report
