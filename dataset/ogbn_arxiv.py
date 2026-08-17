from ogb.nodeproppred import (
    PygNodePropPredDataset
)


def load_data():

    dataset = PygNodePropPredDataset(
        name="ogbn-arxiv",
        root="data"
    )

    data = dataset[0]
    split_idx = dataset.get_idx_split()

    return data, dataset, split_idx