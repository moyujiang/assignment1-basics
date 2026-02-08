import torch
import einops
from torch import nn, Tensor

def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute the cross-entropy loss between logits and targets.

    Args:
        logits (Float[Tensor, "... C"]): The predicted unnormalized log probabilities for each class.
        targets (Long[Tensor, "..."]): The ground truth class indices.

    Returns:
        Float[Tensor, ""]: The computed cross-entropy loss.
    """
    logits_flat = einops.rearrange(logits, "... C -> ( ... ) C")
    targets_flat = einops.rearrange(targets, "... -> ( ... )")

    # More memory-efficient formulation:
    # CE = -log softmax(logits)[target] = logsumexp(logits) - logits[target]
    log_denom = torch.logsumexp(logits_flat, dim=-1)
    target_logits = logits_flat.gather(dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)
    return (log_denom - target_logits).mean()

def perplexity(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute the perplexity given logits and targets.

    Args:
        logits (Float[Tensor, "... C"]): The predicted unnormalized log probabilities for each class.
        targets (Long[Tensor, "..."]): The ground truth class indices.

    Returns:
        Float[Tensor, ""]: The computed perplexity.
    """
    ce_loss = cross_entropy(logits, targets)
    perplexity_value = torch.exp(ce_loss)
    return perplexity_value
