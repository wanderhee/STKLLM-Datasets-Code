import configparser
import os
from datetime import datetime
from time import time

import numpy as np
import torch

import stkllm.paths as paths
from stkllm.ablation import read_ablation_flags
from stkllm.data import get_adjacency_matrix, load_splits, normalize_adjacency
from stkllm.eval import average_loss, batch_loss, epoch_metrics, move_batch, save_predictions
from stkllm.gpu import resolve_torch_devices
from stkllm.model.build import make_stkllm
from stkllm.temporal import build_temporal_encoder_config
from stkllm.variant import read_out_head

def read_config(path_value):
    path = paths.abs_under_root(path_value)
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise RuntimeError("读不到配置: %s" % path)
    return path, parser

def apply_llama_path(config):
    value = config["Training"].get("llama_model_path", "").strip()
    if not value:
        value = paths.LLAMA_MODEL_PATH
    if not os.path.isabs(value):
        raise ValueError("LLaMA 权重路径必须是绝对路径: %s" % value)
    paths.LLAMA_MODEL_PATH = value
    return value

def _device_map(config, gpu_ids):
    mapping = config["Training"].get("llama_device_map", "auto").strip().lower()
    if len(gpu_ids) > 1 and mapping == "auto":
        return "balanced"
    return mapping

def _build_kwargs(config, device, gpu_ids, skip_llama=False):
    data_cfg = config["Data"]
    train_cfg = config["Training"]
    num_vertices = int(data_cfg["num_of_vertices"])
    len_input = int(data_cfg["len_input"])
    in_channels = int(train_cfg["in_channels"])
    nb_filter = int(train_cfg["nb_chev_filter"])
    id_path = None
    if config.has_option("Data", "id_filename") and data_cfg.get("id_filename", "").strip():
        id_path = paths.abs_under_root(data_cfg["id_filename"])
    adjacency, _ = get_adjacency_matrix(
        paths.abs_under_root(data_cfg["adj_filename"]), num_vertices, id_path
    )
    use_qlora = train_cfg.getboolean("use_qlora", fallback=True) and not skip_llama
    checkpoint = train_cfg.getboolean("llama_gradient_checkpointing", fallback=use_qlora)
    chunk = train_cfg.get("llama_attn_chunk_size", "").strip()
    use_temporal, use_spatial, use_gate = read_ablation_flags(config)
    return dict(
        DEVICE=device,
        len_input=len_input,
        time_strides=int(train_cfg["num_of_hours"]),
        in_channels=in_channels,
        nb_chev_filter=nb_filter,
        adj_mx=normalize_adjacency(adjacency),
        num_of_vertices=num_vertices,
        d_model=int(train_cfg["d_model"]),
        num_for_predict=int(data_cfg["num_for_predict"]),
        lora_layers=int(train_cfg.get("lora_layers", 0)),
        llama_device_map=_device_map(config, gpu_ids),
        temporal_encoder_cfg=build_temporal_encoder_config(config, len_input, in_channels, nb_filter),
        use_event_prompt=data_cfg.getboolean("use_event_prompt", fallback=False),
        event_prompt_max_length=int(data_cfg.get("event_prompt_max_length", 64)),
        use_napg=data_cfg.getboolean("use_napg", fallback=False),
        napg_attn_heads=int(data_cfg.get("napg_attn_heads", 4)),
        napg_scale_init=float(data_cfg.get("napg_scale_init", 3.0)),
        use_temporal_encoder=use_temporal,
        use_spatial_encoder=use_spatial,
        use_st_gated_residual=use_gate,
        out_head=read_out_head(config),
        llama_gradient_checkpointing=checkpoint,
        llama_attn_chunk_size=int(chunk) if chunk else None,
        use_qlora=use_qlora,
        skip_llama=skip_llama,
    )

def _load(config):
    data_cfg = config["Data"]
    train_cfg = config["Training"]
    ratio = float(train_cfg["train_ratio"]) if config.has_option("Training", "train_ratio") else 1.0
    return load_splits(
        paths.abs_under_root(data_cfg["graph_signal_matrix_filename"]),
        int(train_cfg["num_of_hours"]),
        int(train_cfg["num_of_days"]),
        int(train_cfg["num_of_weeks"]),
        int(train_cfg["batch_size"]),
        train_ratio=ratio,
    )

def _adam(model, config):
    train_cfg = config["Training"]
    base_lr = float(train_cfg["learning_rate"])
    napg_lr = float(config["Data"].get("napg_lr", base_lr * 10))
    use_napg = config["Data"].getboolean("use_napg", fallback=False)
    napg_params, others = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if use_napg and ("napg_gate" in name or name.endswith("napg_scale")):
            napg_params.append(param)
        else:
            others.append(param)
    if napg_params:
        return torch.optim.Adam(
            [{"params": others, "lr": base_lr}, {"params": napg_params, "lr": napg_lr}]
        )
    return torch.optim.Adam(others, lr=base_lr)

def _reduce_lr(optimizer, factor, min_lr):
    previous = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = max(group["lr"] * factor, min_lr)
    return previous, [group["lr"] for group in optimizer.param_groups]

def _save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "trainable_only": True,
            "state_dict": {
                name: param.detach().cpu()
                for name, param in model.named_parameters()
                if param.requires_grad
            },
        },
        path,
    )

def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    del missing, unexpected

def _run_dir(kind, name):
    folder = os.path.join(
        paths.ROOT, "experiments", kind, "%s_%s" % (name, datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    os.makedirs(folder, exist_ok=True)
    return folder

def _schedule_step(optimizer, config, no_improve):
    train_cfg = config["Training"]
    kind = train_cfg.get("lr_schedule_type", "plateau").strip().lower()
    if kind == "plateau":
        patience = int(train_cfg.get("lr_patience", 0))
        if patience > 0 and no_improve > 0 and no_improve % patience == 0:
            _reduce_lr(
                optimizer, float(train_cfg.get("lr_factor", 0.5)), float(train_cfg.get("min_lr", 1e-6))
            )
    elif kind == "epoch_decay":
        decay = float(train_cfg.get("epoch_lr_decay", 0.98))
        floor = float(train_cfg.get("min_lr", 1e-6))
        for group in optimizer.param_groups:
            group["lr"] = max(group["lr"] * decay, floor)

def train_single(config_path, ctx_override=None, epochs_override=None):
    path, config = read_config(config_path)
    apply_llama_path(config)
    train_cfg = config["Training"]
    ctx = ctx_override or train_cfg.get("ctx", "0")
    use_cuda, gpu_ids, _, device = resolve_torch_devices(ctx)
    if not use_cuda:
        raise RuntimeError("训练需要 CUDA。")
    bundles, mean, std = _load(config)
    model = make_stkllm(**_build_kwargs(config, device, gpu_ids))
    optimizer = _adam(model, config)
    epochs = int(epochs_override or train_cfg["epochs"])
    loss_name = train_cfg.get("loss_function", "mae")
    missing = float(train_cfg.get("missing_value", 0.0))
    metric = train_cfg.get("metric_method", "unmask")
    patience = int(train_cfg.get("early_stop_patience", 0))
    min_delta = float(train_cfg.get("early_stop_min_delta", 0.0))
    print_every = int(train_cfg.get("print_every", 50))
    run_dir = _run_dir("single", config["Data"]["dataset_name"])
    best_path = os.path.join(run_dir, "best_val.params")
    best = float("inf")
    stale = 0
    started = time()

    def predict(features, prompts):
        return model(features, event_prompts=prompts)

    for epoch in range(epochs):
        val_loss = average_loss(model, bundles["val"]["loader"], device, predict, loss_name, missing)
        print("epoch %d val_loss %.6f" % (epoch, val_loss))
        if val_loss < best - min_delta:
            best = val_loss
            stale = 0
            _save(model, best_path)
        else:
            stale += 1
        _schedule_step(optimizer, config, stale)
        model.train()
        losses = []
        for step, batch in enumerate(bundles["train"]["loader"]):
            features, labels, prompts = move_batch(batch, device)
            optimizer.zero_grad()
            loss = batch_loss(predict(features, prompts), labels, loss_name, missing)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if print_every > 0 and (step + 1) % print_every == 0:
                print(
                    "epoch %d step %d/%d loss %.4f time %.1fs"
                    % (epoch, step + 1, len(bundles["train"]["loader"]), losses[-1], time() - started)
                )
        metrics = epoch_metrics(model, bundles["test"]["loader"], device, predict, metric, missing)
        print(
            "epoch %d train %.4f test MAE %.4f RMSE %.4f MAPE %.4f"
            % (epoch, float(np.mean(losses)), metrics["mae"], metrics["rmse"], metrics["mape"])
        )
        if patience > 0 and stale >= patience:
            break
    if os.path.exists(best_path):
        load_checkpoint(model, best_path, device)
    save_predictions(
        model, bundles["test"]["loader"], bundles["test"]["target"], device, predict, metric, mean, std, run_dir, "test"
    )
    return run_dir

class _Cycle(object):
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def next_batch(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)

def train_multi(config_paths, ctx, epochs_override=None, consistency_weight=0.1, perturb_delta=0.1, perturb_sigma=0.01):
    if not config_paths:
        raise ValueError("多数据集训练必须显式传入配置，没有默认组合。")
    if not ctx:
        raise ValueError("多数据集训练必须显式传入 --ctx。")
    from stkllm.model.multi import MultiModeSTKLLM

    parsed = []
    for config_path in config_paths:
        path, config = read_config(config_path)
        parsed.append((path, config, config["Data"]["dataset_name"]))
    names = [item[2] for item in parsed]
    if len(set(names)) != len(names):
        raise ValueError("数据集名称重复: %s" % names)
    apply_llama_path(parsed[0][1])
    use_cuda, gpu_ids, _, device = resolve_torch_devices(ctx)
    if not use_cuda:
        raise RuntimeError("训练需要 CUDA。")

    bundles = {}
    primary_kwargs = _build_kwargs(parsed[0][1], device, gpu_ids)
    primary = make_stkllm(**primary_kwargs)
    towers = {names[0]: primary}
    for index, (path, config, name) in enumerate(parsed):
        loaded, _, _ = _load(config)
        bundles[name] = {
            "config": config,
            "train": loaded["train"]["loader"],
            "val": loaded["val"]["loader"],
            "test": loaded["test"],
            "cycle": _Cycle(loaded["train"]["loader"]),
            "loss_name": config["Training"].get("loss_function", "mae"),
            "missing": float(config["Training"].get("missing_value", 0.0)),
            "metric": config["Training"].get("metric_method", "unmask"),
        }
        if index == 0:
            continue
        kwargs = _build_kwargs(config, device, gpu_ids, skip_llama=True)
        for key in ("d_model", "num_for_predict", "in_channels", "len_input"):
            if kwargs[key] != primary_kwargs[key]:
                raise ValueError("%s 的 %s 与主数据集不一致" % (name, key))
        towers[name] = make_stkllm(**kwargs)

    model = MultiModeSTKLLM(
        primary,
        names[0],
        towers,
        consistency_weight=consistency_weight,
        perturb_delta=perturb_delta,
        perturb_sigma=perturb_sigma,
    )
    optimizer = _adam(model, parsed[0][1])
    epochs = int(epochs_override or parsed[0][1]["Training"]["epochs"])
    steps = max(len(bundles[name]["train"]) for name in names)
    run_dir = _run_dir("multi", "_".join(names))
    best_path = os.path.join(run_dir, "best_val.params")
    best = float("inf")
    started = time()
    probes = {}
    for name in names:
        features, _, _ = move_batch(next(iter(bundles[name]["val"])), device)
        probes[name] = features

    def predict_mode(name, features, prompts, summaries):
        prediction, _ = model.forward_mode(name, features, prompts, summaries=summaries, apply_consistency=False)
        return prediction

    for epoch in range(epochs):
        model.train()
        totals = {name: 0.0 for name in names}
        for step in range(steps):
            batches = {}
            for name in names:
                batches[name] = move_batch(bundles[name]["cycle"].next_batch(), device)
            summaries = model.collect_summaries({name: batches[name][0] for name in names})
            optimizer.zero_grad()
            step_loss = 0.0
            for name in names:
                features, labels, prompts = batches[name]
                prediction, consistency = model.forward_mode(
                    name, features, prompts, summaries=summaries, apply_consistency=True
                )
                pred_loss = batch_loss(prediction, labels, bundles[name]["loss_name"], bundles[name]["missing"])
                loss = model.objective(name, pred_loss, consistency)
                loss.backward()
                step_loss += float(loss.detach())
                totals[name] += float(pred_loss.detach())
            optimizer.step()
            if (step + 1) % 20 == 0:
                print("epoch %d step %d/%d loss %.4f time %.1fs" % (epoch, step + 1, steps, step_loss, time() - started))
        scores = []
        fresh = model.collect_summaries(probes)
        for name in names:
            score = average_loss(
                model,
                bundles[name]["val"],
                device,
                lambda features, prompts, name=name: predict_mode(name, features, prompts, fresh),
                bundles[name]["loss_name"],
                bundles[name]["missing"],
            )
            scores.append(score)
            print(
                "epoch %d %s train_mae %.4f val_mae %.4f s %.4f"
                % (epoch, name, totals[name] / steps, score, float(model.log_s[name].detach()))
            )
        mean_score = float(sum(scores) / len(scores))
        print("epoch %d mean val mae %.4f" % (epoch, mean_score))
        if mean_score < best:
            best = mean_score
            _save(model, best_path)
    return run_dir

def test_single(config_path, ckpt_path, ctx_override=None):
    path, config = read_config(config_path)
    apply_llama_path(config)
    ctx = ctx_override or config["Training"].get("ctx", "0")
    use_cuda, gpu_ids, _, device = resolve_torch_devices(ctx)
    if not use_cuda:
        raise RuntimeError("测试需要 CUDA。")
    bundles, mean, std = _load(config)
    model = make_stkllm(**_build_kwargs(config, device, gpu_ids))
    load_checkpoint(model, paths.abs_under_root(ckpt_path), device)
    metric = config["Training"].get("metric_method", "unmask")

    def predict(features, prompts):
        return model(features, event_prompts=prompts)

    return save_predictions(
        model,
        bundles["test"]["loader"],
        bundles["test"]["target"],
        device,
        predict,
        metric,
        mean,
        std,
        os.path.dirname(paths.abs_under_root(ckpt_path)),
        "test",
    )
