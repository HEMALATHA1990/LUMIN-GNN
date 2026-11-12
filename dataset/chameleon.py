
"""
Chameleon-Filtered dataset loader
Dataset: Platonov et al. (2023), ICLR
Source: yandex-research/heterophilous-graphs
"""

from pathlib import Path
import numpy as np
import torch


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "yandex-research/heterophilous-graphs/"
    "main/data/chameleon_filtered.npz"
)


def get_array(npz, possible_names, description):
    """Find a dataset array using supported key names."""
    for name in possible_names:
        if name in npz.files:
            return npz[name]

    raise KeyError(
        f"Cannot find {description}. "
        f"Available NPZ keys: {npz.files}"
    )


def load_chameleon_filtered(
    root="dataset",
    split_idx=0
):
    """
    Load Chameleon-Filtered and one official split.

    Args:
        root: Dataset storage directory.
        split_idx: Official split index (0-9).

    Returns:
        Dictionary containing graph data and split masks.
    """

    if not 0 <= split_idx < 10:
        raise ValueError("split_idx must be between 0 and 9")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    dataset_path = root / "chameleon_filtered.npz"

    if not dataset_path.exists():
        from urllib.request import urlretrieve
        print("Downloading official Chameleon-Filtered dataset...")
        urlretrieve(DATA_URL, dataset_path)

    with np.load(dataset_path, allow_pickle=False) as data:

        print("Available dataset keys:", data.files)

        x = get_array(
            data,
            ["node_features", "features", "x"],
            "node features"
        )

        y = get_array(
            data,
            ["node_labels", "labels", "y"],
            "node labels"
        )

        edges = get_array(
            data,
            ["edges", "edge_index", "edge"],
            "graph edges"
        )

        train_masks = get_array(
            data,
            ["train_masks", "train_mask"],
            "training masks"
        )

        val_masks = get_array(
            data,
            ["val_masks", "valid_masks", "val_mask"],
            "validation masks"
        )

        test_masks = get_array(
            data,
            ["test_masks", "test_mask"],
            "testing masks"
        )

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long).view(-1)

    edges = np.asarray(edges)

    if edges.ndim != 2:
        raise ValueError("Invalid edge array dimensions")

    if edges.shape[0] == 2:
        edge_index = edges
    elif edges.shape[1] == 2:
        edge_index = edges.T
    else:
        raise ValueError("Invalid edge_index format")

    edge_index = torch.tensor(
        edge_index, dtype=torch.long
    )

    def select_mask(mask_array, name):
        mask_array = np.asarray(mask_array)

        if mask_array.ndim != 2:
            raise ValueError(
                f"{name} must contain all 10 split masks"
            )

        if mask_array.shape[0] == 10:
            selected = mask_array[split_idx]
        elif mask_array.shape[1] == 10:
            selected = mask_array[:, split_idx]
        else:
            raise ValueError(
                f"{name} does not contain 10 official splits: "
                f"{mask_array.shape}"
            )

        return torch.tensor(
            selected, dtype=torch.bool
        ).view(-1)

    train_mask = select_mask(train_masks, "train_masks")
    val_mask = select_mask(val_masks, "val_masks")
    test_mask = select_mask(test_masks, "test_masks")

    num_nodes = x.size(0)

    assert y.size(0) == num_nodes
    assert train_mask.numel() == num_nodes
    assert val_mask.numel() == num_nodes
    assert test_mask.numel() == num_nodes

    assert not torch.any(train_mask & val_mask)
    assert not torch.any(train_mask & test_mask)
    assert not torch.any(val_mask & test_mask)

    print("\nChameleon-Filtered Dataset")
    print("--------------------------")
    print("Number of nodes:", num_nodes)
    print("Number of edges:", edge_index.size(1))
    print("Number of features:", x.size(1))
    print("Number of classes:", torch.unique(y).numel())
    print("Official split:", split_idx)
    print("Training nodes:", train_mask.sum().item())
    print("Validation nodes:", val_mask.sum().item())
    print("Testing nodes:", test_mask.sum().item())

    return {
        "x": x,
        "y": y,
        "edge_index": edge_index,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "split_idx": split_idx
    }
