from datasets.cora import load_data
from models.lumin_gnn import LUMINGNN
from train import train_model
from utils import get_device, set_seed

device = get_device()
set_seed(42)

data, dataset = load_data()
data = data.to(device)

train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

model = LUMINGNN(
    in_channels=dataset.num_features,
    hidden_channels=128,
    out_channels=dataset.num_classes,
    K=2,
    dropout=0.5
).to(device)

train_model(
    model=model,
    data=data,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    lr=0.005,
    weight_decay=5e-4,
    epochs=300,
    patience=30,
    lambda_uncertainty=0.03,
    smoothness_coeff=0.05,
    label_smoothing=0.0,
    grad_clip=1.0,
    scheduler=None
)
