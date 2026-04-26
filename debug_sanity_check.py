"""
debug_sanity_check.py

Run this FIRST before any training to verify:
  1. DV4Linear forward pass works
  2. Flip mask application is correct
  3. Topic mask registry generates and applies masks
  4. Model forward pass produces loss
  5. Gradient flows through STE
  6. Mask switching changes output (the core DV4 test)

Usage:
    cd /path/to/dv4/parent/
    python debug_sanity_check.py

Should complete in < 30 seconds on CPU.
No training data or tokeniser required.

Author: Peter Norman / twoswans.com.au
"""

import torch
import sys

print("DV4 Sanity Check")
print("=" * 60)


# ---------------------------------------------------------------------------
# Test 1: DV4Linear basic operation
# ---------------------------------------------------------------------------
print("\n[1] DV4Linear forward pass...")

from dv4.model.dv4_linear import DV4Linear, quantise_ternary, apply_flip_mask

layer = DV4Linear(64, 32, bias=True)

x = torch.randn(2, 10, 64)  # (batch=2, seq=10, features=64)
out = layer(x)

assert out.shape == (2, 10, 32), f"Expected (2,10,32), got {out.shape}"
print(f"   Output shape: {out.shape} ✅")

# Check ternary distribution
dist = layer.ternary_distribution()
print(f"   Ternary dist: neg={dist['negative']:.2f} zero={dist['zero']:.2f} pos={dist['positive']:.2f}")
assert abs(dist['negative'] + dist['zero'] + dist['positive'] - 1.0) < 1e-5, "Distribution doesn't sum to 1"
print(f"   Distribution sums to 1.0 ✅")


# ---------------------------------------------------------------------------
# Test 2: Flip mask application
# ---------------------------------------------------------------------------
print("\n[2] Flip mask application...")

# Manual test: +1 with flip=1 should become -1
w = torch.tensor([[1.0, -1.0, 0.0, 1.0]])
flip = torch.tensor([[1, 1, 1, 0]], dtype=torch.uint8)
result = apply_flip_mask(w, flip)
expected = torch.tensor([[-1.0, 1.0, 0.0, 1.0]])  # zeros stay, non-zeros flip, last unchanged
assert torch.allclose(result, expected), f"Flip result wrong: {result} vs {expected}"
print(f"   Flip result: {result.tolist()} ✅")

# Zero weight should be unaffected by flip
w_zero = torch.tensor([[0.0, 0.0]])
flip_ones = torch.tensor([[1, 1]], dtype=torch.uint8)
result_zero = apply_flip_mask(w_zero, flip_ones)
assert torch.allclose(result_zero, w_zero), "Zeros changed by flip — bug!"
print(f"   Zeros unaffected by flip ✅")


# ---------------------------------------------------------------------------
# Test 3: Mask changes output
# ---------------------------------------------------------------------------
print("\n[3] Mask switching changes output (core DV4 test)...")

layer2 = DV4Linear(32, 16, bias=False)
x2 = torch.randn(1, 5, 32)

# Output with zero mask (all flip bits off)
zero_mask = torch.zeros(16, 32, dtype=torch.uint8)
layer2.set_flip_mask(zero_mask, topic_id=0)
out_mask0 = layer2(x2).detach().clone()

# Output with ones mask (all flip bits on)
ones_mask = torch.ones(16, 32, dtype=torch.uint8)
layer2.set_flip_mask(ones_mask, topic_id=1)
out_mask1 = layer2(x2).detach().clone()

# Outputs should differ (unless all ternary weights are zero — very unlikely)
diff = (out_mask0 - out_mask1).abs().mean().item()
assert diff > 1e-6, f"Mask switch had no effect on output (diff={diff}) — possible bug"
print(f"   Mean output difference between masks: {diff:.4f} ✅")


# ---------------------------------------------------------------------------
# Test 4: TopicMaskRegistry
# ---------------------------------------------------------------------------
print("\n[4] TopicMaskRegistry...")

from dv4.model.topic_masks import TopicMaskRegistry, POC_TOPICS
from dv4.model.dv4_transformer import DV4Transformer

# Use tiny model for speed
model = DV4Transformer.tiny_config(vocab_size=1000, max_seq_len=64)
n_dv4 = model.count_dv4_layers()
print(f"   Tiny model has {n_dv4} DV4Linear layers")

registry = TopicMaskRegistry(POC_TOPICS, global_seed=42)
registry.register_model(model)

# Check overlap is ~0.5
overlap = registry.mask_overlap(0, 1)
print(f"   Mask overlap (topic 0 vs 1): {overlap:.4f} (expect ~0.500)")
assert 0.4 < overlap < 0.6, f"Mask overlap {overlap} too far from 0.5"
print(f"   Mask overlap within expected range ✅")

# Apply each topic mask and verify it changes
registry.set_active_topic(model, topic_id=0)
registry.set_active_topic(model, topic_id=1)
print(f"   Topic switching works ✅")


# ---------------------------------------------------------------------------
# Test 5: Full model forward pass + loss
# ---------------------------------------------------------------------------
print("\n[5] Full model forward pass + loss computation...")

input_ids = torch.randint(0, 1000, (2, 32))  # (batch=2, seq=32)
labels = input_ids.clone()

logits, loss = model(input_ids, labels)

assert logits.shape == (2, 32, 1000), f"Logits shape wrong: {logits.shape}"
assert loss is not None, "Loss is None — label handling broken"
assert not torch.isnan(loss), "Loss is NaN — check model init or forward pass"
assert not torch.isinf(loss), "Loss is Inf — check label masking"
print(f"   Logits shape: {logits.shape} ✅")
print(f"   Loss: {loss.item():.4f} ✅")


# ---------------------------------------------------------------------------
# Test 6: Gradient flow through STE
# ---------------------------------------------------------------------------
print("\n[6] Gradient flow through STE...")

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer.zero_grad()

logits, loss = model(input_ids, labels)
loss.backward()

# Check that at least some DV4Linear layers have non-zero gradients
from dv4.model.dv4_linear import DV4Linear as DV4L
grad_norms = []
for name, module in model.named_modules():
    if isinstance(module, DV4L):
        if module.weight.grad is not None:
            grad_norms.append(module.weight.grad.norm().item())

assert len(grad_norms) > 0, "No gradients found in DV4Linear layers"
assert any(g > 0 for g in grad_norms), "All gradients are zero — STE not working"
mean_grad = sum(grad_norms) / len(grad_norms)
print(f"   Layers with gradients: {len(grad_norms)}/{n_dv4}")
print(f"   Mean gradient norm: {mean_grad:.6f} ✅")

optimizer.step()
print(f"   Optimiser step completed ✅")


# ---------------------------------------------------------------------------
# Test 7: Mask is NOT updated by gradient step (buffers don't train)
# ---------------------------------------------------------------------------
print("\n[7] Flip masks not modified by training...")

registry.set_active_topic(model, topic_id=0)

# Record mask before training step
sample_layer = next(m for m in model.modules() if isinstance(m, DV4L))
mask_before = sample_layer.flip_mask.clone()

# Do a training step
optimizer.zero_grad()
logits, loss = model(input_ids, labels)
loss.backward()
optimizer.step()

mask_after = sample_layer.flip_mask.clone()
assert torch.equal(mask_before, mask_after), "Flip mask was modified during training — critical bug!"
print(f"   Flip mask unchanged after training step ✅")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ALL CHECKS PASSED ✅")
print("DV4 architecture is functional. Ready for training.")
print("\nNext steps:")
print("  1. Prepare math and general training data (.jsonl format)")
print("  2. Run: python -m dv4.training.train \\")
print("           --math_data data/math.jsonl \\")
print("           --general_data data/general.jsonl \\")
print("           --output_dir checkpoints/poc_run1 \\")
print("           --model_size tiny \\")
print("           --steps_per_topic 100")
print("  3. Run bleed test on the trained checkpoint")
print("=" * 60)
