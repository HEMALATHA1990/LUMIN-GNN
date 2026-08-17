from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_data():

    dataset = Planetoid(
        root="data/CiteSeer",
        name="CiteSeer",
        transform=NormalizeFeatures()
    )

    data = dataset[0]

    return data, dataset