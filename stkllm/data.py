import csv
import os

import numpy as np
import torch

def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    if distance_df_filename.endswith(".npy"):
        return np.load(distance_df_filename), None
    size = int(num_of_vertices)
    adjacency = np.zeros((size, size), dtype=np.float32)
    distance = np.zeros((size, size), dtype=np.float32)
    id_dict = None
    if id_filename:
        with open(id_filename, "r") as handle:
            id_dict = {int(node_id): index for index, node_id in enumerate(handle.read().strip().split("\n"))}
    with open(distance_df_filename, "r") as handle:
        handle.readline()
        for row in csv.reader(handle):
            if len(row) != 3:
                continue
            src, dst, dist = int(row[0]), int(row[1]), float(row[2])
            if id_dict is not None:
                src, dst = id_dict[src], id_dict[dst]
            adjacency[src, dst] = 1
            distance[src, dst] = dist
    return adjacency, distance

def normalize_adjacency(adjacency):
    matrix = torch.tensor(adjacency, dtype=torch.float32)
    matrix = matrix + torch.eye(matrix.size(0))
    degree = matrix.sum(dim=1).clamp_min(1e-6)
    scale = torch.diag(degree.pow(-0.5))
    return scale.matmul(matrix).matmul(scale)

class GraphForecastDataset(torch.utils.data.Dataset):
    def __init__(self, features, target, prompts=None):
        self.features = features
        self.target = target
        self.prompts = prompts

    def __len__(self):
        return int(self.features.shape[0])

    def __getitem__(self, index):
        if self.prompts is None:
            return self.features[index], self.target[index]
        prompt = "" if self.prompts[index] is None else str(self.prompts[index])
        return self.features[index], self.target[index], prompt

def collate_graph_batch(batch):
    if len(batch[0]) == 2:
        features, target = zip(*batch)
        return torch.stack(features, 0), torch.stack(target, 0)
    features, target, prompts = zip(*batch)
    return torch.stack(features, 0), torch.stack(target, 0), list(prompts)

def _window_path(signal_path, hours, days, weeks):
    folder = os.path.dirname(signal_path)
    stem = os.path.basename(signal_path).split(".")[0]
    return os.path.join(folder, "%s_r%s_d%s_w%s.npz" % (stem, hours, days, weeks))

def load_splits(signal_path, hours, days, weeks, batch_size, train_ratio=1.0, shuffle=True):
    path = _window_path(signal_path, hours, days, weeks)
    stored = np.load(path, allow_pickle=True)

    def take(name, keep=None):
        array = stored[name]
        if array.ndim == 4:
            array = array[:, :, 0:1, :]
        if keep is not None:
            array = array[:keep]
        return array

    train_count = int(stored["train_x"].shape[0])
    keep = train_count
    train_ratio = float(train_ratio)
    if train_ratio <= 0 or train_ratio > 1:
        raise ValueError("train_ratio must be in (0, 1], got %s" % train_ratio)
    if train_ratio < 1:
        keep = max(1, int(train_count * train_ratio))

    def prompts(key, keep_rows=None):
        if key not in stored:
            return None
        values = stored[key]
        if keep_rows is not None:
            values = values[:keep_rows]
        return values

    splits = {
        "train": (take("train_x", keep), take("train_target", keep), prompts("train_event_prompts", keep)),
        "val": (take("val_x"), take("val_target"), prompts("val_event_prompts")),
        "test": (take("test_x"), take("test_target"), prompts("test_event_prompts")),
    }
    bundles = {}
    for name, (features, target, prompt) in splits.items():
        dataset = GraphForecastDataset(
            torch.from_numpy(np.array(features, dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(target, dtype=np.float32, copy=True)),
            None if prompt is None else np.array(prompt, copy=True),
        )
        bundles[name] = {
            "loader": torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle if name == "train" else False,
                collate_fn=collate_graph_batch,
            ),
            "target": dataset.target,
        }
    mean = np.array(stored["mean"][:, :, 0:1, :], dtype=np.float32, copy=True)
    std = np.array(stored["std"][:, :, 0:1, :], dtype=np.float32, copy=True)
    stored.close()
    return bundles, mean, std
