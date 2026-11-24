import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
                       
                                                      
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels,
                                 null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae_test(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(y_pred, y_true).astype('float32'),
                      )
        mae = np.nan_to_num(mask * mae)
        return np.mean(mae)

def masked_rmse_test(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
                               
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = ((y_pred- y_true)**2)
        mse = np.nan_to_num(mask * mse)
        return np.sqrt(np.mean(mse))

def compute_pointwise_metrics(y_true, y_pred, metric_method='unmask', null_val=0.0):
                                                               
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if metric_method == 'mask':
        mae = float(masked_mae_test(y_true, y_pred, null_val))
        rmse = float(masked_rmse_test(y_true, y_pred, null_val))
        mape = float(masked_mape_np(y_true, y_pred, null_val))
    else:
        mae = float(mean_absolute_error(y_true.reshape(-1, 1), y_pred.reshape(-1, 1)))
        rmse = float(mean_squared_error(y_true.reshape(-1, 1), y_pred.reshape(-1, 1)) ** 0.5)
        mape = float(masked_mape_np(y_true.reshape(-1, 1), y_pred.reshape(-1, 1), 0))
    return {'mae': mae, 'rmse': rmse, 'mape': mape}

def compute_horizon_metrics_report(prediction, data_target, metric_method='unmask', null_val=0.0):
\
\
\
\
\
       
    prediction = np.asarray(prediction, dtype=np.float32)
    data_target = np.asarray(data_target, dtype=np.float32)
    if prediction.shape != data_target.shape:
        raise ValueError('prediction/target shape mismatch: %s vs %s' % (prediction.shape, data_target.shape))
    if prediction.ndim != 3:
        raise ValueError('expected shape (batch, nodes, horizon), got %s' % (prediction.shape,))

    num_steps = int(prediction.shape[2])
    step_wise = []
    cumulative_avg = []
    for step_idx in range(num_steps):
        step_wise.append(
            compute_pointwise_metrics(
                data_target[:, :, step_idx],
                prediction[:, :, step_idx],
                metric_method,
                null_val,
            )
        )
        cumulative_avg.append(
            compute_pointwise_metrics(
                data_target[:, :, : step_idx + 1],
                prediction[:, :, : step_idx + 1],
                metric_method,
                null_val,
            )
        )

    overall = compute_pointwise_metrics(data_target, prediction, metric_method, null_val)
    return {
        "num_steps": num_steps,
        "step_wise": step_wise,
        "cumulative_avg": cumulative_avg,
        "overall": overall,
    }

def format_horizon_metrics_report(report):
                                                        
    lines = []
    lines.append('=== Per-step metrics ===')
    for idx, metrics in enumerate(report['step_wise'], start=1):
        lines.append(
            "step %d | MAE: %.6f | RMSE: %.6f | MAPE: %.6f"
            % (idx, metrics['mae'], metrics['rmse'], metrics['mape'])
        )

    lines.append('=== Cumulative average (steps 1..k) ===')
    for idx, metrics in enumerate(report['cumulative_avg'], start=1):
        lines.append(
            "前%d步平均 | MAE: %.6f | RMSE: %.6f | MAPE: %.6f"
            % (idx, metrics['mae'], metrics['rmse'], metrics['mape'])
        )

    overall = report['overall']
    lines.append(
        "overall (all %d steps) | MAE: %.6f | RMSE: %.6f | MAPE: %.6f"
        % (report['num_steps'], overall['mae'], overall['rmse'], overall['mape'])
    )
    return '\n'.join(lines)
