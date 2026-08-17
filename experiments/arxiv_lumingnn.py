from datasets.ogbn_arxiv import load_data
from models.lumin_gnn import LUMINGNN
from train import train_model
from utils import get_device, set_seed

device = get_device()

set_seed(42)

data, dataset, split_idx = load_data()
data = data.to(device)

train_mask = split_idx["train"]
val_mask = split_idx["valid"]
test_mask = split_idx["test"]

model = LUMINGNN(
    in_channels=dataset.num_features,
    hidden_channels=256,
    out_channels=dataset.num_classes,
    K=3,
    dropout=0.3
).to(device)

train_model(
    model=model,
    data=data,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    lr=0.005,
    weight_decay=1e-5,
    epochs=150,
    patience=20,
    lambda_uncertainty=0.05,
    smoothness_coeff=0.05,
    label_smoothing=0.1,
    grad_clip=1.0,
    scheduler="cosine"
)
