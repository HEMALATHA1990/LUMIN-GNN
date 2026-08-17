from datasets.ogbn_arxiv import load_data
from models.gat import GAT
from train import train_model
from utils import get_device

device = get_device()

data, dataset, split_idx = load_data()
data = data.to(device)

train_mask = split_idx["train"]
val_mask = split_idx["valid"]
test_mask = split_idx["test"]

model = GAT(
    dataset.num_features,
    256,
    dataset.num_classes,
    heads=8,
    dropout=0.5
).to(device)

train_model(
    model,
    data,
    train_mask,
    val_mask,
    test_mask
)