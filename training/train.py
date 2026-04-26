"""
dv4/training/train.py

Main training loop for DV4 — supports 4 topics and GPU training.

Usage:
    python -m dv4.training.train \
        --math_data    dv4/data/math.jsonl \
        --general_data dv4/data/general.jsonl \
        --code_data    dv4/data/code.jsonl \
        --science_data dv4/data/science.jsonl \
        --output_dir   /mnt/Models/dv4_500m_run1 \
        --model_size   medium \
        --steps_per_topic 2000

Author: Peter Norman / twoswans.com.au
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dv4.model.dv4_transformer import DV4Transformer
from dv4.model.topic_masks import TopicMaskRegistry, POC_TOPICS
from dv4.training.topic_schedule import TopicSchedule, TrainingPhase
from dv4.training.ternary_utils import (
    clip_shadow_weights,
    training_step_diagnostics,
)


class TextDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_len=1024, max_samples=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        print(f"Loading dataset from {path}...")
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
        print(f"  Loaded {len(self.samples)} samples")

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


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Tokeniser
    print("\nLoading tokeniser...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"  Vocab size: {vocab_size}")

    # Model
    print(f"\nBuilding {args.model_size} DV4Transformer...")
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
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params/1e6:.1f}M")

    # Registry
    registry = TopicMaskRegistry(POC_TOPICS, global_seed=args.seed)
    registry.register_model(model)
    print(registry.summary())

    # Build topic list from available data files
    topic_data_args = {
        0: args.math_data,
        1: args.general_data,
        2: getattr(args, 'code_data', None),
        3: getattr(args, 'science_data', None),
    }
    active_topics = {k: v for k, v in topic_data_args.items() if v is not None}
    print(f"\nActive topics: {[POC_TOPICS[k].name for k in active_topics]}")

    # Datasets
    datasets = {}
    for topic_id, data_path in active_topics.items():
        datasets[topic_id] = TextDataset(
            data_path, tokenizer, args.max_seq_len, args.max_samples
        )

    # Schedule
    phases = []
    for cycle in range(args.num_cycles):
        for topic_id, dataset in datasets.items():
            phases.append(TrainingPhase(
                topic_id=topic_id,
                dataset=dataset,
                steps=args.steps_per_topic,
                phase_name=f"{POC_TOPICS[topic_id].name}_cycle{cycle}",
            ))

    schedule = TopicSchedule(
        registry=registry,
        model=model,
        phases=phases,
        batch_size=args.batch_size,
    )

    # Optimiser
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )
    total_steps = schedule.total_steps()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )

    # Training loop
    print(f"\nStarting training: {total_steps} total steps")
    model.train()
    log = []

    for phase, batch, step in schedule.iterate():
        input_ids = batch['input_ids'].to(device)
        labels    = batch['labels'].to(device)

        optimizer.zero_grad()
        logits, loss = model(input_ids, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        diag = training_step_diagnostics(model, loss, step, log_every=args.log_every)
        diag['topic'] = POC_TOPICS[phase.topic_id].name
        log.append(diag)

        optimizer.step()
        scheduler.step()
        clip_shadow_weights(model, clip_val=1.0)

        if step % args.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            vram = ""
            if device.type == 'cuda':
                used = torch.cuda.memory_allocated() / 1e9
                vram = f" | vram={used:.1f}GB"
            print(
                f"  Step {step:6d} | topic={diag['topic']:8s} | "
                f"loss={diag['loss']:.4f} | lr={lr:.2e}{vram}"
            )

        if step > 0 and step % args.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_step{step}.pt"
            save_checkpoint(model, registry, optimizer, step, log, ckpt_path)

    # Final save
    final_path = output_dir / "model_final.pt"
    save_checkpoint(model, registry, optimizer, step, log, final_path)
    print(f"\nTraining complete. Model saved to {final_path}")

    log_path = output_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved to {log_path}")

    return model, registry


def save_checkpoint(model, registry, optimizer, step, log, path):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'registry_masks': registry.masks,
        'log': log[-200:],
    }, path)
    print(f"  Checkpoint saved: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train DV4 model')
    parser.add_argument('--math_data',    type=str, required=True)
    parser.add_argument('--general_data', type=str, required=True)
    parser.add_argument('--code_data',    type=str, default=None)
    parser.add_argument('--science_data', type=str, default=None)
    parser.add_argument('--output_dir',   type=str, default='./checkpoints/run1')
    parser.add_argument('--model_size',   type=str, default='medium',
                        choices=['tiny', 'poc', 'medium', 'large'])
    parser.add_argument('--tokenizer',    type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--max_seq_len',  type=int, default=1024)
    parser.add_argument('--max_samples',  type=int, default=None)
    parser.add_argument('--steps_per_topic', type=int, default=2000)
    parser.add_argument('--num_cycles',   type=int, default=1)
    parser.add_argument('--batch_size',   type=int, default=16)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--log_every',    type=int, default=100)
    parser.add_argument('--save_every',   type=int, default=1000)
    parser.add_argument('--seed',         type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    train(args)
