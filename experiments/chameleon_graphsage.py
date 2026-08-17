import numpy as np
from datasets.chameleon import load_data
from models.graphsage import GraphSAGE
from train import train_model
from utils import get_device

device = get_device()

data, dataset = load_data()
data = data.to(device)

results = []

for split in range(10):

    train_mask = data.train_mask[:, split]
    val_mask = data.val_mask[:, split]
    test_mask = data.test_mask[:, split]

    model = GraphSAGE(
        dataset.num_features,
        256,
        dataset.num_classes,
        dropout=0.5
    ).to(device)

    acc = train_model(
        model,
        data,
        train_mask,
        val_mask,
        test_mask
    )

    results.append(acc)

print()
print("GraphSAGE Results")
print("Mean:", np.mean(results))
print("Std:", np.std(results))