from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_data():

    dataset = Planetoid(
        root="data/Cora",
        name="Cora",
        transform=NormalizeFeatures()
    )

    data = dataset[0]

    return data, dataset