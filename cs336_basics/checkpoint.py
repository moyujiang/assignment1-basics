import torch
from torch import nn, Tensor
import os
from typing import BinaryIO, IO


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """Save a training checkpoint to a file.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        iteration: The current training iteration.
        out: Path or file-like object to save the checkpoint to.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load a training checkpoint from a file.

    Args:
        src: Path or file-like object to load the checkpoint from.
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.
    
    Returns:
        The training iteration stored in the checkpoint.
    """
    checkpoint = torch.load(src, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration