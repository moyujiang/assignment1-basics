import torch
import numpy as np
import numpy.typing as npt
from pathlib import Path


def load_dataset(
    data_path: str | Path,
    vocab_size: int,
    dtype: np.dtype = np.uint16,
    use_mmap: bool = True,
) -> npt.NDArray:
    """Load a dataset from disk, optionally using memory mapping for large files.
    
    Args:
        data_path: Path to the dataset file (.npy format).
        vocab_size: Expected vocabulary size for validation.
        dtype: Expected data type of the dataset (default: uint16).
        use_mmap: If True, use memory mapping to avoid loading entire file into memory.
                 If False, load the entire array into memory.
    
    Returns:
        A numpy array or memory-mapped array of the dataset.
        
    Raises:
        ValueError: If data validation fails.
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    # Load with or without memory mapping
    if use_mmap:
        dataset = np.load(data_path, mmap_mode='r', allow_pickle=False)
    else:
        dataset = np.load(data_path, allow_pickle=False)
    
    # Validate dtype
    if dataset.dtype != dtype:
        raise ValueError(
            f"Dataset dtype mismatch. Expected {dtype}, got {dataset.dtype}"
        )
    
    # Validate that all values are within vocabulary range [0, vocab_size)
    min_val = dataset.min()
    max_val = dataset.max()
    
    if min_val < 0 or max_val >= vocab_size:
        raise ValueError(
            f"Dataset contains invalid token IDs. "
            f"Expected range [0, {vocab_size}), but got [{min_val}, {max_val}]"
        )
    
    return dataset


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch from a dataset.
    
    Supports both regular numpy arrays and memory-mapped arrays (created by load_dataset).
    Each sample consists of a context_length sequence, with inputs being token IDs at 
    positions [i, i+1, ..., i+context_length-1] and targets at [i+1, i+2, ..., i+context_length].

    Args:
        dataset: A numpy array or memory-mapped array of token IDs.
        batch_size: Number of sequences in the batch.
        context_length: Length of each sequence.
        device: Torch device to place tensors on.

    Returns:
        A tuple of (inputs, targets) tensors, each of shape (batch_size, context_length).
    """
    dataset_length = dataset.shape[0]
    
    if context_length >= dataset_length:
        raise ValueError(
            f"context_length ({context_length}) must be less than "
            f"dataset length ({dataset_length})"
        )
    
    # Sample random starting positions
    start_indices = np.random.randint(0, dataset_length - context_length, size=batch_size)
    
    # Use advanced indexing for more efficient batch creation
    # Create index arrays for inputs and targets
    idx = start_indices[:, None] + np.arange(context_length)
    inputs = dataset[idx]
    targets = dataset[idx + 1]

    return torch.from_numpy(inputs).to(device=device, dtype=torch.long), \
           torch.from_numpy(targets).to(device=device, dtype=torch.long)