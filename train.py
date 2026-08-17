import copy
import torch
import torch.nn.functional as F


def accuracy(pred, y):
    return (pred == y).float().mean().item()


def train_model(
    model,
    data,
    train_mask,
    val_mask,
    test_mask,
    lr=0.005,
    weight_decay=5e-4,
    epochs=300,
    patience=30,
    lambda_uncertainty=0.05,
    smoothness_coeff=0.05,
    label_smoothing=0.0,
    grad_clip=1.0,
    scheduler=None,
):

    device = next(model.parameters()).device

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    if scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )

    elif scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=20
        )

    else:
        scheduler = None

    best_val = 0.0
    best_test = 0.0
    best_epoch = 0
    counter = 0

    best_state = None

    train_losses = []
    val_losses = []
    test_losses = []

    train_accs = []
    val_accs = []
    test_accs = []

    for epoch in range(epochs):

        #################################################
        # Training
        #################################################
        model.train()

        optimizer.zero_grad()

        out = model(
            data.x,
            data.edge_index
        )

        y = data.y
        if y.dim() > 1:
            y = y.squeeze()

        loss = F.cross_entropy(
            out[train_mask],
            y[train_mask],
            label_smoothing=label_smoothing
        )

        #################################################
        # Optional Smoothness Regularization
        #################################################
        if smoothness_coeff > 0:

            row, col = data.edge_index

            smooth_loss = (
                out[row] - out[col]
            ).pow(2).mean()

            loss = (
                loss
                +
                smoothness_coeff * smooth_loss
            )

        #################################################
        # Optional Uncertainty Loss
        #################################################
        if (
            lambda_uncertainty > 0
            and hasattr(model, "logvar")
        ):
            kl_loss = -0.5 * torch.mean(
                1
                +
                model.logvar.weight
                -
                model.mu.weight.pow(2)
                -
                model.logvar.weight.exp()
            )

            loss = (
                loss
                +
                lambda_uncertainty * kl_loss
            )

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip
            )

        optimizer.step()

        #################################################
        # Evaluation
        #################################################
        model.eval()

        with torch.no_grad():

            out = model(
                data.x,
                data.edge_index
            )

            pred = out.argmax(dim=1)

            train_acc = accuracy(
                pred[train_mask],
                y[train_mask]
            )

            val_acc = accuracy(
                pred[val_mask],
                y[val_mask]
            )

            test_acc = accuracy(
                pred[test_mask],
                y[test_mask]
            )

            train_loss = F.cross_entropy(
                out[train_mask],
                y[train_mask]
            ).item()

            val_loss = F.cross_entropy(
                out[val_mask],
                y[val_mask]
            ).item()

            test_loss = F.cross_entropy(
                out[test_mask],
                y[test_mask]
            ).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        #################################################
        # Scheduler
        #################################################
        if scheduler is not None:

            if isinstance(
                scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(val_acc)

            else:
                scheduler.step()

        #################################################
        # Early Stopping
        #################################################
        if val_acc > best_val:

            best_val = val_acc
            best_test = test_acc
            best_epoch = epoch

            counter = 0

            best_state = copy.deepcopy(
                model.state_dict()
            )

        else:
            counter += 1

        #################################################
        # Logging
        #################################################
        if epoch % 20 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Train {train_acc:.4f} | "
                f"Val {val_acc:.4f} | "
                f"Test {test_acc:.4f} | "
                f"Loss {loss.item():.4f}"
            )

        if counter >= patience:
            print(
                f"\nEarly stopping at epoch {epoch}"
            )
            break

    #################################################
    # Restore Best Model
    #################################################
    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nBest Epoch:", best_epoch)
    print(f"Best Validation Accuracy: {best_val*100:.2f}%")
    print(f"Best Test Accuracy: {best_test*100:.2f}%")

    return {
        "best_val": best_val,
        "best_test": best_test,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "test_accs": test_accs,
    }
