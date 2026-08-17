from datasets.cora import load_data
from models.mixhop import MixHop
from train import train_model
from utils import get_device

device = get_device()

data, dataset = load_data()
data = data.to(device)

model = MixHop(
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