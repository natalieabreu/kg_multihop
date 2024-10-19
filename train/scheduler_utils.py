import math
import warnings
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR




class LambdaLRWithMin(LambdaLR):
    def __init__(self, optimizer, lr_lambda, min_lr=0.0, last_epoch=-1, verbose=False):
        self.min_lr = min_lr
        super().__init__(optimizer, lr_lambda, last_epoch, verbose)
        

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        # Change (base_lr) to (base_lr-min_lr)
        return [self.min_lr + (base_lr-self.min_lr) * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


# Modification of huggingface functions to allow for minimum lr
def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, min_lr: float = 0.0
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLRWithMin(optimizer, lr_lambda, min_lr, last_epoch)

def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, min_lr=0.0):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLRWithMin(optimizer, lr_lambda, min_lr, last_epoch)


def _get_constant_schedule_with_cooldown_lambda(
    current_step: int, *, num_warmup_steps: int, num_cooldown_steps: int, num_training_steps: int
):
    if current_step < num_warmup_steps:
        return (float(current_step) / float(max(1, num_warmup_steps)))
    elif current_step < num_training_steps - num_cooldown_steps:
        return 1.0
    else:
        return (float(num_training_steps - current_step) / float(max(1, num_cooldown_steps)))
    
def get_constant_schedule_with_cooldown(optimizer, num_warmup_steps, num_training_steps, num_cooldown_steps, last_epoch=-1, min_lr=0):
    """
    Create a schedule with a constant learning rate, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer, and a cooldown period during which it decreases linearly from the
    initial lr set in the optimizer to 0.
    """
    lr_lambda = partial(
        _get_constant_schedule_with_cooldown_lambda,
        num_warmup_steps=num_warmup_steps,
        num_cooldown_steps=num_cooldown_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLRWithMin(optimizer, lr_lambda, min_lr, last_epoch)