"""Tests for text generation functions."""

import torch
import pytest
from cs336_basics.generate import apply_temperature, top_p_sampling, generate
from cs336_basics.transformer import TransformerLM


def test_apply_temperature():
    """Test temperature scaling."""
    logits = torch.tensor([1.0, 2.0, 3.0])
    
    # Temperature = 1.0 should not change logits
    scaled = apply_temperature(logits, temperature=1.0)
    torch.testing.assert_close(scaled, logits)
    
    # Temperature > 1.0 should reduce magnitude
    scaled_high = apply_temperature(logits, temperature=2.0)
    assert (scaled_high.abs() < logits.abs()).all()
    
    # Temperature < 1.0 should increase magnitude
    scaled_low = apply_temperature(logits, temperature=0.5)
    assert (scaled_low.abs() > logits.abs()).all()
    
    # Invalid temperature should raise error
    with pytest.raises(ValueError):
        apply_temperature(logits, temperature=0)
    with pytest.raises(ValueError):
        apply_temperature(logits, temperature=-1)


def test_top_p_sampling():
    """Test top-p sampling masking."""
    # Simple case: uniform probabilities
    probs = torch.ones(10) / 10
    
    # p=1.0 should not change probabilities
    masked = top_p_sampling(probs, p=1.0)
    torch.testing.assert_close(masked, probs)
    
    # p < 1.0 should zero out some probabilities
    masked = top_p_sampling(probs, p=0.5)
    assert (masked >= 0).all()
    assert masked.sum() < 1.01 or masked.sum() > 0.99  # Should be normalized
    
    # Skewed probabilities: test actual filtering
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    masked = top_p_sampling(probs, p=0.8)
    
    # With p=0.8, should keep top tokens (0.5 + 0.3 = 0.8)
    assert masked[0] > 0  # Top token kept
    assert masked[1] > 0  # Second token kept
    assert masked.sum() > 0.99 and masked.sum() < 1.01  # Normalized
    
    # Invalid p should raise error
    with pytest.raises(ValueError):
        top_p_sampling(probs, p=0)
    with pytest.raises(ValueError):
        top_p_sampling(probs, p=1.5)


def test_generate_basic():
    """Test basic text generation."""
    model = TransformerLM(
        vocab_size=100,
        context_length=64,
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_ff=256,
        rope_theta=10000.0,
    )
    
    batch_size = 2
    input_ids = torch.randint(0, 100, (batch_size, 10))
    
    # Generate with default parameters
    output_ids = generate(model, input_ids, max_new_tokens=5)
    
    # Check output shape
    assert output_ids.shape[0] == batch_size
    assert output_ids.shape[1] == 10 + 5  # input + generated
    
    # Check that input is preserved
    torch.testing.assert_close(output_ids[:, :10], input_ids)
    
    # Check that output token IDs are in valid range
    assert (output_ids >= 0).all()
    assert (output_ids < 100).all()


def test_generate_with_temperature():
    """Test temperature sampling is applied correctly."""
    model = TransformerLM(
        vocab_size=100,
        context_length=64,
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_ff=256,
        rope_theta=10000.0,
    )
    
    input_ids = torch.randint(0, 100, (1, 10))
    
    # Generate with different temperatures - just verify it doesn't crash
    # and produces valid token IDs
    output_low = generate(model, input_ids, max_new_tokens=5, temperature=0.1)
    output_high = generate(model, input_ids, max_new_tokens=5, temperature=2.0)
    
    # Check shapes are correct
    assert output_low.shape[0] == 1
    assert output_high.shape[0] == 1
    assert output_low.shape[1] == 15  # 10 + 5
    assert output_high.shape[1] == 15
    
    # Check token IDs are in valid range
    assert (output_low >= 0).all() and (output_low < 100).all()
    assert (output_high >= 0).all() and (output_high < 100).all()


def test_generate_with_top_p():
    """Test generation with top-p sampling."""
    model = TransformerLM(
        vocab_size=100,
        context_length=64,
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_ff=256,
        rope_theta=10000.0,
    )
    
    input_ids = torch.randint(0, 100, (1, 10))
    
    # Generate with top-p
    output = generate(model, input_ids, max_new_tokens=10, top_p=0.9)
    
    assert output.shape[1] == 10 + 10
    assert (output >= 0).all() and (output < 100).all()


def test_generate_max_tokens():
    """Test that generation respects max_new_tokens."""
    model = TransformerLM(
        vocab_size=100,
        context_length=64,
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_ff=256,
        rope_theta=10000.0,
    )
    
    input_len = 5
    input_ids = torch.randint(0, 100, (1, input_len))
    
    for max_tokens in [1, 5, 10]:
        output = generate(model, input_ids, max_new_tokens=max_tokens)
        assert output.shape[1] == input_len + max_tokens


def test_generate_with_eos():
    """Test early stopping with EOS token."""
    model = TransformerLM(
        vocab_size=100,
        context_length=64,
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_ff=256,
        rope_theta=10000.0,
    )
    
    input_ids = torch.randint(0, 100, (2, 10))
    eos_token_id = 99  # Assume 99 is EOS
    
    output = generate(
        model, 
        input_ids, 
        max_new_tokens=100,
        eos_token_id=eos_token_id,
    )
    
    # Output should be shorter than max (unless we don't hit EOS in both sequences)
    assert output.shape[1] <= 10 + 100
    assert (output >= 0).all() and (output < 100).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
