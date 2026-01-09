import torch
import torch.nn as nn

def distill_student(
    student,
    train_loader,
    teacher_ens_train,  # numpy array (N_train, H)
    epochs=20,
    lr=1e-3,
    alpha_start=0.7,
    alpha_end=0.2,
    device=None,
    verbose=False,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    hard_loss = nn.L1Loss()          # MAE
    soft_loss = nn.HuberLoss()       # robust to spikes

    teacher_ens_train_t = torch.tensor(teacher_ens_train, dtype=torch.float32, device=device)

    n_seen = 0
    for epoch in range(epochs):
        student.train()
        # linear schedule
        alpha = alpha_start + (alpha_end - alpha_start) * (epoch / max(1, epochs-1))

        offset = 0
        epoch_loss = 0.0
        for x, y in train_loader:
            b = x.shape[0]
            x = x.float().to(device)
            y = y.float().to(device)

            # Align teacher batch by running offset in the same order as the loader (shuffle must be False)
            t_batch = teacher_ens_train_t[offset:offset+b]
            offset += b

            pred = student(x)
            loss = hard_loss(pred, y) + alpha * soft_loss(pred, t_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * b
            n_seen += b

        if verbose:
            print(f"epoch {epoch+1}/{epochs} alpha={alpha:.3f} loss={epoch_loss/max(1,offset):.4f}")

    return student
