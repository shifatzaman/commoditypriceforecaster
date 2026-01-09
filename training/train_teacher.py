import torch
import torch.nn as nn

def train_teacher(
    model,
    train_loader,
    val_loader,
    epochs=30,
    lr=1e-3,
    patience=5,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()  # MAE on log-diff targets

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        for x, y, _last_price in train_loader:   # ✅ changed
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---- validate ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, _last_price in val_loader:  # ✅ changed
                x, y = x.to(device), y.to(device)
                val_losses.append(loss_fn(model(x), y).item())

        val_loss = sum(val_losses) / max(1, len(val_losses))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model