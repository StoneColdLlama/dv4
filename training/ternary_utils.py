"""
dv4/training/ternary_utils.py

Training utilities for DV4 ternary weight training.

Covers:
  - Gradient health monitoring (STE can produce dead gradients)
  - Ternary distribution tracking across training
  - Weight clipping (shadow weights should stay bounded)
  - Scale factor monitoring

Author: Peter Norman / twoswans.com.au
Architecture: DV4 (Dual-Vocab 4-bit)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from ..model.dv4_linear import DV4Linear


# ---------------------------------------------------------------------------
# Shadow weight management
# ---------------------------------------------------------------------------

def clip_shadow_weights(model: nn.Module, clip_val: float = 1.0):
    """
    Clip shadow weights to [-clip_val, +clip_val].
    
    Shadow weights can drift far from zero during training, which causes
    the ternary threshold to grow large and pushes more weights to ±1,
    reducing the number of zero weights (losing sparsity benefit).
    
    Call this after each optimiser step.
    
    Args:
        model:    DV4 transformer model
        clip_val: Maximum absolute value for shadow weights
    """
    for module in model.modules():
        if isinstance(module, DV4Linear):
            with torch.no_grad():
                module.weight.clamp_(-clip_val, clip_val)


# ---------------------------------------------------------------------------
# Gradient diagnostics
# ---------------------------------------------------------------------------

def check_gradient_health(model: nn.Module) -> Dict[str, float]:
    """
    Check gradient statistics across all DV4Linear layers.
    
    The STE can cause gradient issues:
      - Dead gradients: grad_norm ≈ 0 (weights not learning)
      - Exploding gradients: grad_norm >> 1 (training instability)
    
    Returns dict with per-layer gradient norms and a summary.
    Call after loss.backward() but before optimizer.step().
    
    Returns:
        Dict mapping layer names to gradient norms,
        plus 'mean', 'max', 'dead_fraction' summary keys.
    """
    stats = {}
    norms = []
    dead_count = 0
    total_count = 0

    for name, module in model.named_modules():
        if isinstance(module, DV4Linear):
            if module.weight.grad is not None:
                grad_norm = module.weight.grad.norm().item()
                stats[name] = grad_norm
                norms.append(grad_norm)

                # A layer is "dead" if its gradient norm is essentially zero
                if grad_norm < 1e-8:
                    dead_count += 1
                total_count += 1
            else:
                stats[name] = 0.0
                dead_count += 1
                total_count += 1

    if norms:
        stats['mean'] = sum(norms) / len(norms)
        stats['max'] = max(norms)
        stats['dead_fraction'] = dead_count / total_count if total_count > 0 else 1.0
    else:
        stats['mean'] = 0.0
        stats['max'] = 0.0
        stats['dead_fraction'] = 1.0

    return stats


# ---------------------------------------------------------------------------
# Ternary distribution tracking
# ---------------------------------------------------------------------------

def get_model_ternary_stats(model: nn.Module) -> Dict[str, float]:
    """
    Aggregate ternary weight distribution across all DV4Linear layers.
    
    Key metrics to watch during training:
      - zero_fraction: Should stay ~0.3–0.5. If too high, model is losing
        representational capacity. If too low, losing sparsity benefit.
      - pos_fraction / neg_fraction: Should be roughly equal (balanced polarity).
        Large imbalance suggests the model is learning a biased representation.
    
    Returns:
        Dict with 'negative', 'zero', 'positive' fractions (sum to 1.0),
        plus per-layer breakdown.
    """
    all_neg, all_zero, all_pos = 0, 0, 0
    layer_stats = {}

    for name, module in model.named_modules():
        if isinstance(module, DV4Linear):
            dist = module.ternary_distribution()
            layer_stats[name] = dist

            total = module.weight.numel()
            all_neg  += dist['negative'] * total
            all_zero += dist['zero']     * total
            all_pos  += dist['positive'] * total

    grand_total = all_neg + all_zero + all_pos
    if grand_total > 0:
        return {
            'negative': all_neg  / grand_total,
            'zero':     all_zero / grand_total,
            'positive': all_pos  / grand_total,
            'layers':   layer_stats,
        }
    return {'negative': 0.0, 'zero': 0.0, 'positive': 0.0, 'layers': {}}


def get_flip_utilisation_stats(model: nn.Module) -> Dict[str, float]:
    """
    Get flip bit utilisation statistics across all DV4Linear layers.
    
    Expected: ~0.5 for all layers (masks are 50% ones by design).
    Significant deviation indicates a mask generation bug.
    
    Returns per-layer utilisation and model-wide mean.
    """
    utils = {}
    for name, module in model.named_modules():
        if isinstance(module, DV4Linear):
            utils[name] = module.flip_utilisation()

    if utils:
        utils['mean'] = sum(utils.values()) / len(utils)
    return utils


# ---------------------------------------------------------------------------
# Training step helper
# ---------------------------------------------------------------------------

def training_step_diagnostics(
    model: nn.Module,
    loss: torch.Tensor,
    step: int,
    log_every: int = 100,
) -> Dict:
    """
    Collect diagnostics at a training step.
    
    Only computes expensive stats every log_every steps.
    Always returns loss value.
    
    Args:
        model:     DV4 transformer
        loss:      Current loss tensor (after backward)
        step:      Current training step
        log_every: How often to compute full diagnostics
    
    Returns:
        Dict with at minimum {'loss': float}, plus full stats every log_every steps
    """
    result = {'loss': loss.item(), 'step': step}

    if step % log_every == 0:
        grad_stats = check_gradient_health(model)
        ternary_stats = get_model_ternary_stats(model)

        result.update({
            'grad_mean':      grad_stats.get('mean', 0.0),
            'grad_max':       grad_stats.get('max', 0.0),
            'dead_fraction':  grad_stats.get('dead_fraction', 0.0),
            'ternary_neg':    ternary_stats['negative'],
            'ternary_zero':   ternary_stats['zero'],
            'ternary_pos':    ternary_stats['positive'],
        })

        # Warn on pathological conditions
        if grad_stats.get('dead_fraction', 0) > 0.5:
            print(f"  ⚠️  Step {step}: {grad_stats['dead_fraction']:.0%} dead gradients")
        if ternary_stats['zero'] > 0.7:
            print(f"  ⚠️  Step {step}: {ternary_stats['zero']:.0%} zero weights — representational collapse?")
        if abs(ternary_stats['positive'] - ternary_stats['negative']) > 0.2:
            print(f"  ⚠️  Step {step}: polarity imbalance "
                  f"(+{ternary_stats['positive']:.2f} / -{ternary_stats['negative']:.2f})")

    return result
