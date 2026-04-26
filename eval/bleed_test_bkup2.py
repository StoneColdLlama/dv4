"""
dv4/eval/bleed_test.py

Cross-topic contamination test — the key PoC validation experiment.

The central question DV4 must answer:
  "When trained under topic mask A, do the ternary weights learn something
   that is SPECIFIC to topic A, or do they bleed into topic B?"

Test protocol:
  1. Train model on math (mask 0) and general (mask 1) using TopicSchedule
  2. For each topic, collect a held-out test set
  3. Evaluate the model under:
     a. CORRECT mask  (topic A data + mask A, topic B data + mask B)
     b. WRONG mask    (topic A data + mask B, topic B data + mask A)
  4. Compare perplexity under correct vs wrong mask

Expected results if DV4 is working:
  - Correct mask perplexity << Wrong mask perplexity (clear separation)
  - The gap is the "topic specificity score"

If wrong mask perplexity ≈ correct mask perplexity:
  - The flip bits are not being utilised meaningfully
  - Or the ternary weights are topic-agnostic (masks don't matter)
  - Investigate: flip utilisation stats, ternary distribution per topic

Author: Peter Norman / twoswans.com.au
Architecture: DV4 (Dual-Vocab 4-bit)
"""

import torch
import torch.nn.functional as F
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from torch.utils.data import DataLoader

from ..model.topic_masks import TopicMaskRegistry


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    """
    Compute perplexity of model on a dataset.
    
    Perplexity = exp(mean cross-entropy loss)
    
    Lower perplexity = model assigns higher probability to the data.
    This is what we use to measure topic specificity.
    
    Args:
        model:       DV4Transformer (flip mask must already be set)
        dataloader:  DataLoader yielding {'input_ids', 'labels'} batches
        device:      Compute device
        max_batches: Limit evaluation (for speed during debug)
    
    Returns:
        Perplexity as a float
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)
        labels    = batch['labels'].to(device)

        _, loss = model(input_ids, labels)

        if loss is not None:
            # loss is mean over non-masked tokens
            # Count non-masked tokens in this batch
            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float('inf')

    mean_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(mean_loss)).item()
    return perplexity


# ---------------------------------------------------------------------------
# Bleed test
# ---------------------------------------------------------------------------

class BleedTest:
    """
    Runs the cross-topic contamination experiment.
    
    For each topic:
      - Evaluate under CORRECT mask → perplexity_correct
      - Evaluate under WRONG mask   → perplexity_wrong
      - Topic specificity = (perplexity_wrong - perplexity_correct) / perplexity_correct
    
    A high specificity score means the flip mask is doing real work.
    A score near 0 means the masks are interchangeable (DV4 not working).
    """

    def __init__(
        self,
        model,
        registry: TopicMaskRegistry,
        device: torch.device,
    ):
        self.model = model
        self.registry = registry
        self.device = device
        self.results = {}

    def run(
        self,
        topic_dataloaders: Dict[int, DataLoader],
        max_batches: Optional[int] = 50,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the full bleed test across all topic pairs.
        
        Args:
            topic_dataloaders: Dict mapping topic_id → DataLoader of held-out data
            max_batches:       Batches per evaluation (None = full dataset)
            verbose:           Print results table
        
        Returns:
            Results dict with perplexities and specificity scores
        """
        topic_ids = list(topic_dataloaders.keys())
        results = {}

        if verbose:
            print("\n" + "="*70)
            print("DV4 BLEED TEST — Cross-Topic Contamination Analysis")
            print("="*70)
            print(f"{'Topic':<12} {'Data':<12} {'Mask':<12} {'Perplexity':>12} {'Note'}")
            print("-"*70)

        for data_topic_id in topic_ids:
            data_topic_name = self.registry.topics[data_topic_id].name
            dataloader = topic_dataloaders[data_topic_id]
            results[data_topic_id] = {}

            for mask_topic_id in topic_ids:
                mask_topic_name = self.registry.topics[mask_topic_id].name

                # Activate the mask for mask_topic_id
                self.registry.set_active_topic(self.model, mask_topic_id)

                # Compute perplexity
                ppl = compute_perplexity(
                    self.model, dataloader, self.device, max_batches
                )
                results[data_topic_id][mask_topic_id] = ppl

                is_correct = data_topic_id == mask_topic_id
                note = "← CORRECT" if is_correct else "← WRONG"

                if verbose:
                    print(
                        f"  {data_topic_name:<12} {data_topic_name:<12} "
                        f"{mask_topic_name:<12} {ppl:>12.2f} {note}"
                    )

        if verbose:
            print("-"*70)

        # Compute specificity scores
        specificity = {}
        for topic_id in topic_ids:
            correct_ppl = results[topic_id][topic_id]
            wrong_ppls = [
                results[topic_id][m] for m in topic_ids if m != topic_id
            ]
            mean_wrong_ppl = sum(wrong_ppls) / len(wrong_ppls) if wrong_ppls else correct_ppl

            # Relative improvement from using the correct mask
            if correct_ppl > 0:
                spec_score = (mean_wrong_ppl - correct_ppl) / correct_ppl
            else:
                spec_score = 0.0

            specificity[topic_id] = spec_score
            topic_name = self.registry.topics[topic_id].name

            if verbose:
                print(f"\nTopic {topic_id} ({topic_name}):")
                print(f"  Correct mask perplexity:  {correct_ppl:.2f}")
                print(f"  Wrong mask perplexity:    {mean_wrong_ppl:.2f}")
                print(f"  Topic specificity score:  {spec_score:+.3f}")
                if spec_score > 0.1:
                    print(f"  ✅ DV4 flip masks are topic-specific (>{spec_score:.0%} higher PPL wrong mask)")
                elif spec_score > 0.02:
                    print(f"  ⚠️  Weak topic specificity — may need more training or different masks")
                else:
                    print(f"  ❌ No topic specificity — flip masks are not differentiating topics")

        self.results = {
            'perplexities': results,
            'specificity': specificity,
        }

        if verbose:
            print("\n" + "="*70)
            mean_spec = sum(specificity.values()) / len(specificity)
            print(f"Mean topic specificity: {mean_spec:+.3f}")
            if mean_spec > 0.05:
                print("VERDICT: DV4 flip masks are working — topics are differentiated ✅")
            else:
                print("VERDICT: DV4 flip masks are NOT differentiating topics ❌")
            print("="*70 + "\n")

        return self.results

    def save_results(self, path: str):
        """Save bleed test results to JSON."""
        # Convert tensor keys to strings for JSON serialisation
        serialisable = {
            'perplexities': {
                str(k): {str(mk): mv for mk, mv in v.items()}
                for k, v in self.results['perplexities'].items()
            },
            'specificity': {
                str(k): v for k, v in self.results['specificity'].items()
            }
        }
        with open(path, 'w') as f:
            json.dump(serialisable, f, indent=2)
        print(f"Bleed test results saved to {path}")


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_bleed_test(
    model,
    registry: TopicMaskRegistry,
    math_dataloader: DataLoader,
    general_dataloader: DataLoader,
    device: torch.device,
    output_path: Optional[str] = None,
    max_batches: int = 50,
) -> Dict:
    """
    Convenience function to run the full PoC bleed test.
    
    Args:
        model:              Trained DV4Transformer
        registry:           Topic mask registry
        math_dataloader:    Held-out math test data
        general_dataloader: Held-out general test data
        device:             Compute device
        output_path:        If provided, save results to this JSON path
        max_batches:        Limit evaluation batches for speed
    
    Returns:
        Results dict from BleedTest.run()
    """
    bleed_test = BleedTest(model, registry, device)

    topic_dataloaders = {
        0: math_dataloader,
        1: general_dataloader,
    }

    results = bleed_test.run(
        topic_dataloaders,
        max_batches=max_batches,
        verbose=True,
    )

    if output_path:
        bleed_test.save_results(output_path)

    return results
