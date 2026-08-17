import torch


@torch.no_grad()
def evaluate(
    model,
    data,
    train_mask,
    val_mask,
    test_mask
):
    model.eval()

    out = model(data)

    if isinstance(out, tuple):
        out = out[0]

    pred = out.argmax(dim=1)

    train_acc = (
        pred[train_mask]
        == data.y[train_mask]
    ).float().mean().item()

    val_acc = (
        pred[val_mask]
        == data.y[val_mask]
    ).float().mean().item()

    test_acc = (
        pred[test_mask]
        == data.y[test_mask]
    ).float().mean().item()

    return (
        train_acc,
        val_acc,
        test_acc
    )