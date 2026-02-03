"""Text generation and decoding functions for language models."""

import torch
from torch import Tensor


def apply_temperature(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """Apply temperature scaling to logits.
    
    Args:
        logits: Unnormalized log probabilities of shape (..., vocab_size).
        temperature: Scaling factor. Values > 1 increase randomness, < 1 make deterministic.
    
    Returns:
        Temperature-scaled logits.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    return logits / temperature


def top_p_sampling(probs: Tensor, p: float = 0.95) -> Tensor:
    """Apply top-p (nucleus) sampling mask.
    
    Masks out tokens until cumulative probability exceeds p.
    
    Args:
        probs: Probabilities of shape (..., vocab_size), summing to 1.
        p: Nucleus threshold (0, 1]. Include top tokens until cumsum > p.
    
    Returns:
        Masked probabilities (others set to 0), renormalized.
    """
    if not (0 < p <= 1):
        raise ValueError(f"p must be in (0, 1], got {p}")
    
    if p >= 1.0:
        return probs  # No filtering needed
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Compute cumulative sum
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Keep tokens until cumulative probability reaches p
    # (i.e., include the token that crosses the threshold)
    mask = (cumsum_probs - sorted_probs) < p
    # Always keep at least the top token
    mask[..., 0] = True
    
    # Zero out non-selected probabilities
    sorted_probs[~mask] = 0.0
    
    # Scatter back to original indices
    masked_probs = torch.zeros_like(probs)
    masked_probs.scatter_(-1, sorted_indices, sorted_probs)
    
    # Renormalize
    prob_sum = masked_probs.sum(dim=-1, keepdim=True)
    return masked_probs / (prob_sum + 1e-10)


def generate(
    model: torch.nn.Module,
    input_ids: Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int = None,
    device: str = "cpu",
) -> Tensor:
    """Generate text from a language model.
    
    Args:
        model: Language model (e.g., TransformerLM).
        input_ids: Starting token sequence of shape (batch, seq_len).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Temperature scaling for sampling. 1.0 = no scaling.
        top_p: Top-p threshold for nucleus sampling. 1.0 = disabled.
        eos_token_id: End-of-sequence token ID. Generation stops when encountered.
        device: Device to run inference on.
    
    Returns:
        Generated sequences including the input, shape (batch, seq_len + num_generated).
    """
    model.eval()
    context_length = model.context_length
    batch_size = input_ids.shape[0]
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(device) if device is not None else input_ids.device
    if input_ids.device != model_device:
        input_ids = input_ids.to(model_device)
    
    with torch.no_grad():
        generated = input_ids.clone()
        finished = None
        if eos_token_id is not None:
            finished = generated[:, -1] == eos_token_id
        
        for _ in range(max_new_tokens):
            if finished is not None and finished.all():
                break
            # Truncate to context length if needed
            if generated.shape[1] > context_length:
                input_for_model = generated[:, -context_length:]
            else:
                input_for_model = generated
            
            # Get logits for next token
            logits = model(input_for_model)  # (batch, seq_len, vocab_size)
            next_logits = logits[:, -1, :]   # (batch, vocab_size)
            
            # Clamp logits to prevent overflow in softmax
            max_logits = 20.0  # softmax is invariant to adding constants
            next_logits = torch.clamp(next_logits, min=-max_logits, max=max_logits)
            
            # Subtract max for numerical stability (standard trick)
            next_logits = next_logits - next_logits.max(dim=-1, keepdim=True)[0]
            
            # Apply temperature
            next_logits = apply_temperature(next_logits, temperature)
            
            # Convert to probabilities with numerical stability
            probs = torch.softmax(next_logits, dim=-1)
            
            # If softmax produced NaN/Inf, use uniform distribution
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.ones_like(next_logits) / next_logits.shape[-1]
            else:
                # Apply top-p sampling
                if top_p < 1.0:
                    probs = top_p_sampling(probs, top_p)
                
                # Final safety check and normalization
                probs = torch.clamp(probs, min=0.0)
                prob_sum = probs.sum(dim=-1, keepdim=True)
                
                if (prob_sum == 0).any():
                    # Fallback to uniform if sum is 0
                    probs = torch.ones_like(probs) / probs.shape[-1]
                else:
                    probs = probs / prob_sum
            
            # Sample next tokens
            next_token_ids = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # If EOS, stop generating for finished sequences
            if finished is not None:
                eos_reached = next_token_ids.squeeze(-1) == eos_token_id
                finished = finished | eos_reached
                if finished.any():
                    next_token_ids = next_token_ids.clone()
                    next_token_ids[finished, 0] = eos_token_id

            # Append to sequence
            generated = torch.cat([generated, next_token_ids], dim=1)
    
    return generated
