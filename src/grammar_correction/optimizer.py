"""Optimizer module for the Transformer model.

This module contains the NoamOpt class, which implements the learning rate
scheduling used in the "Attention Is All You Need" paper.
"""

from typing import Optional
import torch
from torch.optim import Optimizer


class NoamOpt:
    """Optim wrapper that implements the Noam learning rate schedule.
    
    This schedule increases the learning rate linearly for the first `warmup`
    steps, and decreases it thereafter proportionally to the inverse square
    root of the step number.
    """

    def __init__(self, model_size: int, factor: float, warmup: int, optimizer: Optimizer):
        """Initializes the NoamOpt wrapper.

        Args:
            model_size (int): The hidden size of the model (d_model).
            factor (float): A scaling factor for the learning rate.
            warmup (int): The number of warmup steps.
            optimizer (Optimizer): The inner optimizer to wrap (e.g., Adam).
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self) -> None:
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step: Optional[int] = None) -> float:
        """Calculate the learning rate for a specific step.

        Args:
            step (Optional[int]): The current step number. If None, uses internal step.

        Returns:
            float: The calculated learning rate.
        """
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self) -> None:
        """Clears the gradients of all optimized torch.Tensors."""
        self.optimizer.zero_grad()
