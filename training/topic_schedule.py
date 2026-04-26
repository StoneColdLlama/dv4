"""
dv4/training/topic_schedule.py

TopicSchedule — controls which topic mask is active during training.

Training protocol for DV4:
  1. Activate topic mask for topic A (e.g. math)
  2. Train on topic A data for N steps
  3. Activate topic mask for topic B (e.g. general)
  4. Train on topic B data for N steps
  5. Repeat for M cycles

The ternary weights are shared across topics — only the flip mask changes.
The masks are NEVER updated during training. Only the shadow weights learn.

This module handles the interleaving schedule and dataset routing.

Author: Peter Norman / twoswans.com.au
Architecture: DV4 (Dual-Vocab 4-bit)
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, List, Iterator, Tuple, Optional
from dataclasses import dataclass

from ..model.topic_masks import TopicMaskRegistry, TopicDefinition


# ---------------------------------------------------------------------------
# Training phase definition
# ---------------------------------------------------------------------------

@dataclass
class TrainingPhase:
    """
    A single training phase: one topic, one dataset, N steps.
    
    Attributes:
        topic_id:   Which topic mask to activate
        dataset:    Dataset to draw batches from
        steps:      Number of training steps in this phase
        phase_name: Human-readable label for logging
    """
    topic_id: int
    dataset: Dataset
    steps: int
    phase_name: str = ""

    def __post_init__(self):
        if not self.phase_name:
            self.phase_name = f"topic_{self.topic_id}"


# ---------------------------------------------------------------------------
# Topic schedule
# ---------------------------------------------------------------------------

class TopicSchedule:
    """
    Manages the training schedule across topics.
    
    PoC schedule (2 topics, 1 cycle):
      Phase 0: Math mask active    → train on math data  → N steps
      Phase 1: General mask active → train on general data → N steps
    
    Can be extended to multiple cycles, more topics, or interleaved batches.
    
    Usage:
        schedule = TopicSchedule(registry, model, phases)
        for phase, batch, step in schedule.iterate(batch_size=32):
            logits, loss = model(batch['input_ids'], batch['labels'])
            # ... backward, step, etc.
    """

    def __init__(
        self,
        registry: TopicMaskRegistry,
        model: torch.nn.Module,
        phases: List[TrainingPhase],
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        self.registry = registry
        self.model = model
        self.phases = phases
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Validate all topic IDs are registered
        for phase in phases:
            if phase.topic_id not in registry.topics:
                raise ValueError(
                    f"Phase topic_id {phase.topic_id} not in registry. "
                    f"Available: {list(registry.topics.keys())}"
                )

    def iterate(self) -> Iterator[Tuple[TrainingPhase, Dict, int]]:
        """
        Iterate through all training phases, yielding (phase, batch, step).
        
        Automatically:
          - Activates the correct topic mask before each phase
          - Creates a DataLoader for each phase's dataset
          - Yields batches until phase steps are exhausted
        
        Yields:
            (phase, batch_dict, global_step)
            where batch_dict has keys 'input_ids' and 'labels'
        """
        global_step = 0

        for phase_idx, phase in enumerate(self.phases):
            topic = self.registry.topics[phase.topic_id]
            print(f"\n{'='*60}")
            print(f"Phase {phase_idx}: {phase.phase_name}")
            print(f"  Topic: {topic.name} (id={phase.topic_id})")
            print(f"  Steps: {phase.steps}")
            print(f"{'='*60}")

            # Activate the topic mask for this phase
            self.registry.set_active_topic(self.model, phase.topic_id)

            # Create dataloader for this phase
            loader = DataLoader(
                phase.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            )
            loader_iter = iter(loader)

            phase_step = 0
            while phase_step < phase.steps:
                # Get next batch, cycling through dataset if needed
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    batch = next(loader_iter)

                yield phase, batch, global_step

                phase_step += 1
                global_step += 1

        print(f"\nTopicSchedule: completed {global_step} total steps across {len(self.phases)} phases")

    def total_steps(self) -> int:
        """Total number of training steps across all phases."""
        return sum(p.steps for p in self.phases)

    @classmethod
    def poc_schedule(
        cls,
        registry: TopicMaskRegistry,
        model: torch.nn.Module,
        math_dataset: Dataset,
        general_dataset: Dataset,
        steps_per_topic: int = 1000,
        batch_size: int = 8,
        num_cycles: int = 1,
    ) -> 'TopicSchedule':
        """
        Build the standard PoC two-topic schedule.
        
        Alternates between math and general phases for num_cycles cycles.
        
        Args:
            registry:          Topic mask registry (with model registered)
            model:             DV4 transformer model
            math_dataset:      Dataset for math topic (topic_id=0)
            general_dataset:   Dataset for general topic (topic_id=1)
            steps_per_topic:   Training steps per topic per cycle
            batch_size:        Batch size
            num_cycles:        Number of full math→general cycles
        
        Returns:
            Configured TopicSchedule
        """
        phases = []
        for cycle in range(num_cycles):
            phases.append(TrainingPhase(
                topic_id=0,
                dataset=math_dataset,
                steps=steps_per_topic,
                phase_name=f"math_cycle{cycle}",
            ))
            phases.append(TrainingPhase(
                topic_id=1,
                dataset=general_dataset,
                steps=steps_per_topic,
                phase_name=f"general_cycle{cycle}",
            ))

        return cls(
            registry=registry,
            model=model,
            phases=phases,
            batch_size=batch_size,
        )
