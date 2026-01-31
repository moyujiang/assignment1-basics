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

    logits_max = torch.max(logits_flat, dim=-1, keepdim=True).values
    logits_stable = logits_flat - logits_max

    exp_logits = torch.exp(logits_stable)
    sum_exp_logits = torch.sum(exp_logits, dim=-1, keepdim=True)
    log_probs = logits_stable - torch.log(sum_exp_logits)

    nll_loss = -log_probs[torch.arange(targets_flat.shape[0]), targets_flat]
    loss = torch.mean(nll_loss)
    return loss

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