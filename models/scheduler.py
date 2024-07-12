import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


def build_scheduler(
    optimizer: torch.optim.Optimizer, lr_scheduler_type, max_epochs, warmup_epochs=10, mode="min", monitor="val.loss"
):
    lr = optimizer.param_groups[0]["lr"]
    if lr_scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0, last_epoch=-1)

    elif lr_scheduler_type == "step":
        scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    elif lr_scheduler_type == "plateau":
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, patience=10, factor=0.1, verbose=True
            ),
            "monitor": monitor,
        }
    elif lr_scheduler_type == "warmup_cosine":
        # warmup_epochs = max_epochs // 10
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=lr / 10,
            eta_min=0,
            last_epoch=-1,
        )

    else:
        raise ValueError("Valid schedulers are: cosine, step, plateau, warmup_cosine")

    return scheduler
