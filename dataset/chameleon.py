from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures


def load_data():

    dataset = WikipediaNetwork(
        root="data",
        name="chameleon",
        geom_gcn_preprocess=True,
        transform=NormalizeFeatures()
    )

    data = dataset[0]

    return data, dataset