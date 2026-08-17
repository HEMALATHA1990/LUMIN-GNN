from datasets.citeseer import load_data
from models.gcn import GCN
from train import train_model
from utils import get_device

device = get_device()

data, dataset = load_data()
data = data.to(device)

model = GCN(
    dataset.num_features,
    256,
    dataset.num_classes,
    dropout=0.5
).to(device)

train_model(
    model,
    data,
    data.train_mask,
    data.val_mask,
    data.test_mask
)