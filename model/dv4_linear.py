"""
dv4/model/dv4_linear.py

DV4Linear — the core weight primitive for the DV4 architecture.

Weight format:
  - 3 bits encode ternary value: maps to {-1, 0, +1}
  - 1 bit is the flip bit: set externally by the active topic mask
  - Together: 4 bits per weight (hence DV4 — Dual-Vocab 4-bit)

Flip bit semantics:
  - flip=0: weight behaves as standard ternary {-1, 0, +1}
  - flip=1: weight interpretation is toggled
      - If ternary == 0  → becomes 0 (zero stays zero — no signal either way)
      - If ternary == +1 → becomes -1 (polarity flipped)
      - If ternary == -1 → becomes +1 (polarity flipped)
  
  In other words: the flip bit inverts non-zero weights.
  A topic mask sets flip bits across all neurons simultaneously,
  effectively reorienting the weight matrix for that topic domain.

Training:
  - Full-precision shadow weights are maintained for gradient flow
  - Ternary quantisation uses a straight-through estimator (STE)
  - Flip bits are FROZEN during training — set by topic_masks.py
  - Only the shadow weights (and thus ternary values) are learned

Inference:
  - Shadow weights are discarded
  - Active flip mask is swapped by the router (router.py)
  - Weight application is a simple integer op — no float multiply

Author: Peter Norman / twoswans.com.au
Architecture: DV4 (Dual-Vocab 4-bit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Ternary quantisation utilities
# ---------------------------------------------------------------------------

def quantise_ternary(weight: torch.Tensor) -> torch.Tensor:
    """
    Quantise a full-precision weight tensor to ternary {-1, 0, +1}.
    
    Uses the mean absolute value as the threshold (per BitNet b1.58).
    Values within [-threshold, +threshold] become 0.
    Values above become +1, below become -1.
    
    Args:
        weight: Full-precision weight tensor (any shape)
    
    Returns:
        Ternary tensor with values in {-1, 0, +1}, same shape as input
    """
    threshold = weight.abs().mean()
    ternary = torch.zeros_like(weight)
    ternary[weight > threshold] = 1.0
    ternary[weight < -threshold] = -1.0
    return ternary


def ternary_forward_ste(weight: torch.Tensor) -> torch.Tensor:
    """
    Straight-through estimator for ternary quantisation.
    
    Forward pass: quantise to ternary
    Backward pass: pass gradients through as if identity (STE)
    
    This allows the full-precision shadow weights to be updated by
    gradient descent even though the actual computation uses ternary values.
    """
    ternary = quantise_ternary(weight)
    # STE: in forward use ternary, in backward pretend we used identity
    return weight + (ternary - weight).detach()


# ---------------------------------------------------------------------------
# Flip bit application
# ---------------------------------------------------------------------------

def apply_flip_mask(ternary_weight: torch.Tensor, flip_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a flip mask to a ternary weight tensor.
    
    For each weight:
      - If flip_mask == 0: weight unchanged
      - If flip_mask == 1: non-zero weights have polarity inverted
        (+1 → -1, -1 → +1, 0 → 0)
    
    This is equivalent to: w * (1 - 2 * flip * |sign(w)|)
    But more clearly implemented as a conditional negation.
    
    Args:
        ternary_weight: Tensor with values in {-1.0, 0.0, +1.0}
        flip_mask: Binary tensor (0 or 1), same shape as ternary_weight
    
    Returns:
        Modified weight tensor, still in {-1.0, 0.0, +1.0}
    """
    # Where flip_mask is 1, negate non-zero weights
    # Zero weights remain zero regardless of flip
    flipped = ternary_weight * (1.0 - 2.0 * flip_mask.float())
    # Restore zeros (negating 0 gives 0, so this is already correct,
    # but being explicit for clarity and future hardware mapping)
    return flipped


# ---------------------------------------------------------------------------
# DV4Linear layer
# ---------------------------------------------------------------------------

class DV4Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear using DV4 weight format.
    
    Each weight is represented as:
      - A full-precision shadow weight (for training only)
      - A ternary quantised value derived from the shadow weight
      - A flip bit from the active topic mask (frozen during training)
    
    The effective weight used in computation is:
      effective = apply_flip_mask(quantise(shadow_weight), active_flip_mask)
    
    A per-layer scale factor (one float per output neuron) is learned
    to compensate for the loss of magnitude information in ternary quantisation.
    This follows the BitNet b1.58 approach.
    
    Args:
        in_features:  Input dimension
        out_features: Output dimension
        bias:         Whether to include a bias term (default: True)
        group_size:   Number of weights sharing one scale factor (default: 128)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Full-precision shadow weights — these are what gradient descent updates
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=0.01)

        # Bias term
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Per-output-neuron scale factor (learned)
        # Shape: (out_features,) — one scale per output neuron
        self.scale = nn.Parameter(torch.ones(out_features))

        # Active flip mask — NOT a parameter, set externally by topic_masks.py
        # Shape: (out_features, in_features) — same as weight
        # Registered as a buffer so it moves with .to(device) but doesn't train
        self.register_buffer(
            'flip_mask',
            torch.zeros(out_features, in_features, dtype=torch.uint8)
        )

        # Track which topic mask is currently active (for diagnostics)
        self.active_topic_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Mask management
    # ------------------------------------------------------------------

    def set_flip_mask(self, mask: torch.Tensor, topic_id: Optional[int] = None):
        """
        Set the active flip mask for this layer.
        
        Called by topic_masks.py during training phase setup
        and by router.py during inference.
        
        Args:
            mask:     Binary tensor (0/1), shape (out_features, in_features)
            topic_id: Optional integer ID for diagnostics
        """
        assert mask.shape == self.flip_mask.shape, (
            f"Mask shape {mask.shape} doesn't match weight shape {self.flip_mask.shape}"
        )
        self.flip_mask.copy_(mask.to(self.flip_mask.device).to(torch.uint8))
        self.active_topic_id = topic_id

    def get_flip_mask(self) -> torch.Tensor:
        """Return the currently active flip mask."""
        return self.flip_mask

    # ------------------------------------------------------------------
    # Weight access
    # ------------------------------------------------------------------

    def get_ternary_weight(self) -> torch.Tensor:
        """
        Return the quantised ternary weight WITHOUT flip applied.
        Used for inspection and saving the trained model.
        """
        return quantise_ternary(self.weight.detach())

    def get_effective_weight(self) -> torch.Tensor:
        """
        Return the effective weight WITH flip mask applied.
        This is what actually gets used in the forward pass computation.
        """
        ternary = quantise_ternary(self.weight.detach())
        return apply_flip_mask(ternary, self.flip_mask)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using DV4 weight format.
        
        During training:
          - Uses STE to allow gradients to flow through quantisation
          - Flip mask is applied on top of quantised weights
          - Scale factor applied per output neuron
        
        During inference (torch.no_grad()):
          - Same computation, shadow weights are used to derive ternary
          - In a production deployment these would be pre-computed integers
        
        Args:
            x: Input tensor, shape (..., in_features)
        
        Returns:
            Output tensor, shape (..., out_features)
        """
        # Step 1: Get ternary weights via STE (training) or direct quantise (inference)
        if self.training:
            ternary_w = ternary_forward_ste(self.weight)
        else:
            ternary_w = quantise_ternary(self.weight)

        # Step 2: Apply active flip mask
        effective_w = apply_flip_mask(ternary_w, self.flip_mask)

        # Step 3: Linear transform
        # Standard F.linear: output = x @ weight.T + bias
        out = F.linear(x, effective_w, self.bias)

        # Step 4: Apply per-neuron scale
        # scale shape: (out_features,) — broadcasts over batch and sequence dims
        out = out * self.scale

        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def flip_utilisation(self) -> float:
        """
        What fraction of weights have their flip bit set?
        
        Should be ~0.5 for a random mask. If it collapses toward 0 or 1
        during training analysis, the mask design may need revisiting.
        """
        return self.flip_mask.float().mean().item()

    def ternary_distribution(self) -> dict:
        """
        Distribution of ternary weight values {-1, 0, +1}.
        
        Healthy distribution: not too many zeros (representational collapse)
        and roughly balanced +1/-1 (no strong polarity bias).
        """
        t = self.get_ternary_weight()
        total = t.numel()
        return {
            'negative': (t == -1).sum().item() / total,
            'zero':     (t ==  0).sum().item() / total,
            'positive': (t ==  1).sum().item() / total,
        }

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"topic={self.active_topic_id}, "
            f"flip_util={self.flip_utilisation():.2f}"
        )
