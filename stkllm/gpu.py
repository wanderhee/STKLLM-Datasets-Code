from __future__ import annotations

import os

def parse_ctx_gpu_ids(ctx_value):
    gpu_ids = []
    for token in str(ctx_value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            gpu_ids.append(int(token))
        except ValueError:
            continue
    if not gpu_ids:
        gpu_ids = [0]

    seen = set()
    ordered = []
    for gpu_id in gpu_ids:
        if gpu_id not in seen:
            seen.add(gpu_id)
            ordered.append(gpu_id)
    return ordered

def apply_cuda_visible_devices_from_ctx(ctx_value):
\
\
\
\
       
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"]

    physical_ids = parse_ctx_gpu_ids(ctx_value)
    visible = ",".join(str(gpu_id) for gpu_id in physical_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return visible

def resolve_torch_devices(ctx_value):
    import torch

    physical_ids = parse_ctx_gpu_ids(ctx_value)
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        return False, [], physical_ids, torch.device("cpu")

    logical_count = torch.cuda.device_count()
    if logical_count <= 0:
        return False, [], physical_ids, torch.device("cpu")

    gpu_ids = list(range(min(len(physical_ids), logical_count)))
    if not gpu_ids:
        gpu_ids = [0]
    return use_cuda, gpu_ids, physical_ids, torch.device("cuda:{}".format(gpu_ids[0]))
