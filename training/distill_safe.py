# training/distill_safe.py
import numpy as np
import torch
import torch.nn as nn

def distill_student_safe(
    student,
    train_loader,                 # yields (x_diff, y_diff, last_price)
    teacher_train_targets_diff,   # (N_train, H) teacher targets in DIFF space
    epochs=25,
    lr=1e-3,
    alpha_start=0.7,
    alpha_end=0.2,
    device=None,
):
    """
    Safe KD:
    - Hard loss on y_diff (MAE)
    - Soft loss on teacher diff targets (Huber)
    - Uses a decaying alpha schedule

    IMPORTANT: train_loader must use shuffle=False OR you must build targets per-batch.
               Easiest safe choice: set shuffle=False for distillation loader.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device)

    opt = torch.optim.Adam(student.parameters(), lr=lr)
    hard = nn.L1Loss()
    soft = nn.HuberLoss()

    teacher_t = torch.tensor(teacher_train_targets_diff, dtype=torch.float32, device=device)

    for ep in range(epochs):
        student.train()
        alpha = alpha_start + (alpha_end - alpha_start) * (ep / max(1, epochs - 1))

        offset = 0
        for x, y, _last_price in train_loader:
            b = x.shape[0]
            x = x.to(device)
            y = y.to(device)

            t = teacher_t[offset:offset+b]
            offset += b

            pred = student(x)
            loss = hard(pred, y) + alpha * soft(pred, t)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return student