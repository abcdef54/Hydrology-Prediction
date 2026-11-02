"""Learning-rate schedules for sequence training."""
import math
import torch


def warmup_cosine(optimizer, *, total_epochs, warmup_epochs, min_lr):
    """Linear warmup, then cosine from the peak to min_lr by the last epoch.

    Schedulers are stepped after training each epoch. LambdaLR calls the
    function at construction, so index 0 is the rate for training epoch 1.
    """
    if not isinstance(total_epochs, int) or not isinstance(warmup_epochs, int):
        raise ValueError('Epoch counts must be integers')
    if warmup_epochs < 1 or total_epochs < warmup_epochs + 2:
        raise ValueError('Cosine schedule requires warmup and at least two decay epochs')
    if not math.isfinite(min_lr) or min_lr < 0:
        raise ValueError('Minimum learning rate must be finite and nonnegative')
    if any(not math.isfinite(group['lr']) or group['lr'] <= 0 or min_lr > group['lr']
           for group in optimizer.param_groups):
        raise ValueError('Minimum learning rate must not exceed a positive base learning rate')
    lambdas = []
    for group in optimizer.param_groups:
        ratio = min_lr / group['lr']
        def factor(epoch, ratio=ratio):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = min(1., (epoch-warmup_epochs)/(total_epochs-warmup_epochs-1))
            return ratio + (1-ratio)*0.5*(1+math.cos(math.pi*progress))
        lambdas.append(factor)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
