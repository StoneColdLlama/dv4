"""
dv4/inference/generate.py

Generation loop for DV4 models with topic-aware mask switching.

Supports:
  - Greedy decoding
  - Temperature sampling
  - Top-k / top-p sampling
  - Mid-prompt topic switching (routes once per prompt for PoC)

Author: Peter Norman / twoswans.com.au
Architecture: DV4 (Dual-Vocab 4-bit)
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
from .router import InferenceRouter


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits except the top-k."""
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_val = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_val, float('-inf'))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift right to keep the first token above threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float('-inf'))


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    Sample next token from logits.
    
    Args:
        logits:      Raw logits, shape (batch, vocab_size)
        temperature: Sampling temperature (1.0 = no change, <1 = sharper)
        top_k:       Keep only top-k tokens (0 = disabled)
        top_p:       Nucleus sampling threshold (1.0 = disabled)
    
    Returns:
        Sampled token ids, shape (batch,)
    """
    if temperature == 0.0:
        # Greedy
        return logits.argmax(dim=-1)

    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model,
    tokenizer,
    router: InferenceRouter,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> str:
    """
    Generate text from a prompt using DV4 topic-aware inference.
    
    Steps:
      1. Route prompt to topic → activate flip mask
      2. Tokenise prompt
      3. Autoregressively generate tokens
      4. Decode and return
    
    Args:
        model:          Trained DV4Transformer
        tokenizer:      HuggingFace tokeniser
        router:         InferenceRouter (handles mask switching)
        prompt:         Input text prompt
        max_new_tokens: Maximum tokens to generate
        temperature:    Sampling temperature
        top_k:          Top-k filtering
        top_p:          Nucleus sampling
        device:         Torch device
        verbose:        Print routing and generation info
    
    Returns:
        Generated text (prompt + completion)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Step 1: Route prompt and activate the correct flip mask
    topic_id = router.route_and_activate(prompt, verbose=verbose)
    topic_name = router.registry.topics[topic_id].name

    if verbose:
        print(f"Generating with topic mask: {topic_name}")

    # Step 2: Tokenise
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    original_len = input_ids.shape[1]

    if verbose:
        print(f"Prompt tokens: {original_len}")

    # Step 3: Autoregressive generation
    generated = input_ids
    eos_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        # Truncate if exceeding max_seq_len
        if generated.shape[1] >= model.max_seq_len:
            if verbose:
                print(f"  Hit max_seq_len at step {step}")
            break

        # Forward pass (no grad)
        logits, _ = model(generated)

        # Take logits at the last position
        next_logits = logits[:, -1, :]  # (batch, vocab_size)

        # Sample next token
        next_token = sample_token(next_logits, temperature, top_k, top_p)
        next_token = next_token.unsqueeze(-1)  # (batch, 1)

        # Append to sequence
        generated = torch.cat([generated, next_token], dim=1)

        # Stop at EOS
        if eos_id is not None and (next_token == eos_id).all():
            if verbose:
                print(f"  EOS at step {step}")
            break

    # Step 4: Decode
    output_tokens = generated[0].tolist()
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)

    if verbose:
        new_tokens = generated.shape[1] - original_len
        print(f"Generated {new_tokens} new tokens")

    return output_text


# ---------------------------------------------------------------------------
# Batch generation (for evaluation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    router: InferenceRouter,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 0.0,  # Greedy by default for eval
    device: Optional[torch.device] = None,
) -> List[str]:
    """
    Generate text for a list of prompts.
    Note: routes each prompt independently (may switch masks between prompts).
    
    Args:
        prompts: List of input prompts
    
    Returns:
        List of generated texts (same length as prompts)
    """
    results = []
    for prompt in prompts:
        result = generate(
            model, tokenizer, router, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            verbose=False,
        )
        results.append(result)
    return results
