"""
dv4/eval/run_bleed_test.py

Runner script for the DV4 cross-topic contamination (bleed) test.
Supports 2-4 topics.

Usage:
    python -m dv4.eval.run_bleed_test \
        --checkpoint /mnt/Models/dv4_500m_run2/model_final.pt \
        --math_data    dv4/data/math.jsonl \
        --general_data dv4/data/general.jsonl \
        --code_data    dv4/data/code.jsonl \
        --science_data dv4/data/science.jsonl \
        --output results/bleed_test_500m.json

Author: Peter Norman / twoswans.com.au
Architecture: DV4 (Dual-Vocab 4-bit)
"""

import sys
import json
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dv4.model.topic_masks import TopicMaskRegistry, POC_TOPICS
from dv4.model.dv4_transformer import DV4Transformer
from dv4.eval.bleed_test import compute_perplexity, BleedTest


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_len=256, max_samples=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    item = json.loads(line.strip())
                    text = item.get('text', '')
                    if text:
                        self.samples.append(text)
                except json.JSONDecodeError:
                    continue
        print(f"  Loaded {len(self.samples)} samples from {Path(path).name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(
            self.samples[idx],
            add_special_tokens=True,
            max_length=self.max_seq_len,
            truncation=True,
        )
        pad_id = self.tokenizer.pad_token_id or 0
        tokens = tokens + [pad_id] * (self.max_seq_len - len(tokens))
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == pad_id] = -100
        return {'input_ids': input_ids, 'labels': labels}


# ---------------------------------------------------------------------------
# 4-topic bleed test
# ---------------------------------------------------------------------------

def run_4topic_bleed_test(model, registry, dataloaders, device, max_batches=50):
    """
    Full N-topic bleed test.
    Tests every data/mask combination — N x N matrix of perplexities.
    """
    topic_ids = sorted(dataloaders.keys())
    topic_names = {t.topic_id: t.name for t in POC_TOPICS}

    results = {t: {} for t in topic_ids}

    print("\n" + "="*75)
    print("DV4 BLEED TEST — Cross-Topic Contamination Analysis (4 Topics)")
    print("="*75)
    print(f"{'Data Topic':<14} {'Mask Topic':<14} {'Perplexity':>12}  Note")
    print("-"*75)

    model.eval()
    for data_topic in topic_ids:
        for mask_topic in topic_ids:
            # Activate mask
            registry.set_active_topic(model, mask_topic)

            # Evaluate
            ppl = compute_perplexity(
                model, dataloaders[data_topic], device, max_batches
            )
            results[data_topic][mask_topic] = ppl

            correct = "← CORRECT" if data_topic == mask_topic else ""
            print(
                f"  {topic_names[data_topic]:<14} {topic_names[mask_topic]:<14} "
                f"{ppl:>12.2f}  {correct}"
            )

    print("-"*75)

    # Compute specificity scores
    print("\nTOPIC SPECIFICITY SCORES")
    print("="*75)
    specificity = {}
    for topic_id in topic_ids:
        correct_ppl = results[topic_id][topic_id]
        wrong_ppls = [results[topic_id][m] for m in topic_ids if m != topic_id]
        mean_wrong = sum(wrong_ppls) / len(wrong_ppls)

        spec = (mean_wrong - correct_ppl) / correct_ppl if correct_ppl > 0 else 0.0
        specificity[topic_id] = spec

        name = topic_names[topic_id]
        print(f"\n  Topic {topic_id} ({name}):")
        print(f"    Correct mask PPL:  {correct_ppl:.2f}")
        print(f"    Mean wrong PPL:    {mean_wrong:.2f}")
        print(f"    Specificity score: {spec:+.3f}")

        if spec > 0.10:
            print(f"    ✅ Strong topic specificity")
        elif spec > 0.02:
            print(f"    ⚠️  Weak specificity — needs more training")
        else:
            print(f"    ❌ No specificity detected")

    mean_spec = sum(specificity.values()) / len(specificity)
    print(f"\n{'='*75}")
    print(f"Mean topic specificity: {mean_spec:+.3f}")
    if mean_spec > 0.10:
        print("VERDICT: DV4 flip masks are working — all topics differentiated ✅")
    else:
        print("VERDICT: Insufficient topic specificity ❌")
    print("="*75)

    # Worst-case bleed (most similar topic pair)
    print("\nWORST-CASE BLEED (most similar topic pair):")
    min_spec = min(specificity.values())
    min_topic = min(specificity, key=specificity.get)
    print(f"  Topic {min_topic} ({topic_names[min_topic]}): {min_spec:+.3f}")

    # Best specificity
    max_spec = max(specificity.values())
    max_topic = max(specificity, key=specificity.get)
    print(f"Best specificity: Topic {max_topic} ({topic_names[max_topic]}): {max_spec:+.3f}")

    return {
        'perplexities': results,
        'specificity': specificity,
        'mean_specificity': mean_spec,
        'topic_names': {str(k): v for k, v in topic_names.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Run DV4 4-topic bleed test')
    parser.add_argument('--checkpoint',   type=str, required=True)
    parser.add_argument('--math_data',    type=str, required=True)
    parser.add_argument('--general_data', type=str, required=True)
    parser.add_argument('--code_data',    type=str, default=None)
    parser.add_argument('--science_data', type=str, default=None)
    parser.add_argument('--tokenizer',    type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--max_batches',  type=int, default=50)
    parser.add_argument('--batch_size',   type=int, default=4)
    parser.add_argument('--max_seq_len',  type=int, default=256)
    parser.add_argument('--model_size',   type=str, default='medium',
                        choices=['tiny', 'poc', 'medium', 'large'])
    parser.add_argument('--output',       type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDV4 Bleed Test Runner (4 Topics)")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")
    if device.type == 'cuda':
        print(f"GPU:        {torch.cuda.get_device_name(0)}")

    # Tokeniser
    print(f"\nLoading tokeniser...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"  Vocab size: {vocab_size}")

    # Load checkpoint
    print(f"\nLoading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f"  Trained to step: {ckpt['step']}")

    # Rebuild model
    print(f"\nRebuilding {args.model_size} model...")
    model_configs = {
        'tiny':   DV4Transformer.tiny_config,
        'poc':    DV4Transformer.poc_config,
        'medium': DV4Transformer.medium_config,
        'large':  DV4Transformer.large_config,
    }
    model = model_configs[args.model_size](
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    # Registry
    print(f"\nRestoring topic masks...")
    registry = TopicMaskRegistry(POC_TOPICS, global_seed=42)
    registry.register_model(model)
    if 'registry_masks' in ckpt:
        registry.masks = ckpt['registry_masks']
        print(f"  Masks restored from checkpoint")
    else:
        print(f"  Masks regenerated from seed")

    # Build dataloaders for available topics
    print(f"\nLoading evaluation data...")
    data_paths = {
        0: args.math_data,
        1: args.general_data,
        2: args.code_data,
        3: args.science_data,
    }
    active = {k: v for k, v in data_paths.items() if v is not None}

    dataloaders = {}
    for topic_id, path in active.items():
        ds = TextDataset(path, tokenizer, args.max_seq_len)
        dataloaders[topic_id] = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, drop_last=False
        )

    print(f"  Topics to evaluate: {[POC_TOPICS[t].name for t in dataloaders]}")

    # Run bleed test
    results = run_4topic_bleed_test(
        model, registry, dataloaders, device, args.max_batches
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Make serialisable
        serialisable = {
            'checkpoint': args.checkpoint,
            'step': ckpt['step'],
            'model_size': args.model_size,
            'mean_specificity': results['mean_specificity'],
            'topic_names': results['topic_names'],
            'specificity': {str(k): v for k, v in results['specificity'].items()},
            'perplexities': {
                str(k): {str(mk): mv for mk, mv in v.items()}
                for k, v in results['perplexities'].items()
            },
        }
        with open(output_path, 'w') as f:
            json.dump(serialisable, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\nINTERPRETATION GUIDE")
    print("="*60)
    print("Specificity = (mean_wrong_PPL - correct_PPL) / correct_PPL")
    print("  > 0.50  Excellent — strong topic separation")
    print("  > 0.10  Good — meaningful differentiation")
    print("  > 0.02  Weak — present but marginal")
    print("  < 0.02  None — masks not differentiating topics")
    print("="*60)


if __name__ == '__main__':
    main()
