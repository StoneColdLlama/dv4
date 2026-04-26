"""
dv4/model/topic_masks.py

TopicMaskRegistry — manages flip bit patterns for each topic domain.

Each topic gets a unique binary mask over every DV4Linear layer in the model.
The mask is generated once, locked, and never updated during training.

4-topic configuration:
  - Topic 0: MATH    — mathematical reasoning, arithmetic, algebra
  - Topic 1: GENERAL — general language, facts, commonsense
  - Topic 2: CODE    — programming, algorithms, software
  - Topic 3: SCIENCE — physics, chemistry, biology, research

Author: Peter Norman / twoswans.com.au
Architecture: DV4 (Dual-Vocab 4-bit)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Topic definition
# ---------------------------------------------------------------------------

@dataclass
class TopicDefinition:
    topic_id: int
    name: str
    description: str
    keywords: List[str] = field(default_factory=list)


# 4-topic registry
POC_TOPICS = [
    TopicDefinition(
        topic_id=0,
        name="math",
        description="Mathematical reasoning, arithmetic, algebra, word problems",
        keywords=[
            "calculate", "solve", "equation", "sum", "product", "difference",
            "quotient", "multiply", "divide", "add", "subtract", "percent",
            "fraction", "algebra", "geometry", "derivative", "integral",
            "probability", "statistics", "proof", "theorem", "formula",
            "number", "digit", "equals", "greater", "less", "value",
            "compute", "total", "average", "mean", "median", "variance",
        ],
    ),
    TopicDefinition(
        topic_id=1,
        name="general",
        description="General language understanding, facts, commonsense reasoning",
        keywords=[],  # Default fallback topic
    ),
    TopicDefinition(
        topic_id=2,
        name="code",
        description="Programming, algorithms, software engineering, debugging",
        keywords=[
            "function", "class", "variable", "loop", "array", "list",
            "dictionary", "import", "return", "print", "debug", "error",
            "algorithm", "recursion", "iteration", "compile", "runtime",
            "python", "javascript", "java", "code", "program", "script",
            "library", "module", "api", "database", "query", "stack",
            "queue", "tree", "graph", "sort", "search", "complexity",
            "def ", "int ", "str ", "bool", "null", "void", "async",
        ],
    ),
    TopicDefinition(
        topic_id=3,
        name="science",
        description="Physics, chemistry, biology, scientific reasoning",
        keywords=[
            "atom", "molecule", "element", "compound", "reaction", "energy",
            "force", "mass", "velocity", "acceleration", "gravity", "electron",
            "proton", "neutron", "cell", "dna", "protein", "evolution",
            "photosynthesis", "osmosis", "hypothesis", "experiment", "theory",
            "nucleus", "orbit", "wave", "frequency", "amplitude", "pressure",
            "temperature", "entropy", "catalyst", "enzyme", "genome", "species",
            "quantum", "relativity", "thermodynamics", "electromagnetic",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def generate_topic_mask(
    shape: Tuple[int, ...],
    topic_id: int,
    global_seed: int = 42,
) -> torch.Tensor:
    shape_hash = hash(shape) % 100000
    seed = global_seed + topic_id * 100000 + shape_hash
    rng = torch.Generator()
    rng.manual_seed(seed % (2**32))
    mask = torch.bernoulli(
        torch.full(shape, 0.5),
        generator=rng
    ).to(torch.uint8)
    return mask


# ---------------------------------------------------------------------------
# Topic mask registry
# ---------------------------------------------------------------------------

class TopicMaskRegistry:
    def __init__(
        self,
        topics: List[TopicDefinition],
        global_seed: int = 42,
    ):
        self.topics = {t.topic_id: t for t in topics}
        self.global_seed = global_seed
        self.masks: Dict[int, Dict[str, torch.Tensor]] = {
            t.topic_id: {} for t in topics
        }
        self.active_topic_id: Optional[int] = None

    def register_model(self, model: nn.Module):
        from .dv4_linear import DV4Linear
        dv4_layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, DV4Linear)
        }
        if not dv4_layers:
            raise ValueError("No DV4Linear layers found in model.")

        print(f"TopicMaskRegistry: found {len(dv4_layers)} DV4Linear layers")

        for topic_id in self.topics:
            for layer_name, layer in dv4_layers.items():
                shape = (layer.out_features, layer.in_features)
                mask = generate_topic_mask(shape, topic_id, self.global_seed)
                self.masks[topic_id][layer_name] = mask

        total_masks = len(self.topics) * len(dv4_layers)
        print(f"TopicMaskRegistry: generated {total_masks} masks "
              f"({len(self.topics)} topics × {len(dv4_layers)} layers)")

    def set_active_topic(self, model: nn.Module, topic_id: int):
        from .dv4_linear import DV4Linear
        if topic_id not in self.topics:
            raise ValueError(f"Topic ID {topic_id} not registered.")
        if not self.masks[topic_id]:
            raise RuntimeError("Masks not initialised. Call register_model() first.")

        topic_masks = self.masks[topic_id]
        for name, module in model.named_modules():
            if isinstance(module, DV4Linear):
                module.set_flip_mask(topic_masks[name], topic_id=topic_id)

        self.active_topic_id = topic_id
        topic_name = self.topics[topic_id].name
        print(f"TopicMaskRegistry: activated topic {topic_id} ({topic_name})")

    def get_active_topic(self) -> Optional[TopicDefinition]:
        if self.active_topic_id is None:
            return None
        return self.topics[self.active_topic_id]

    def mask_overlap(self, topic_a: int, topic_b: int) -> float:
        overlaps = []
        for layer_name in self.masks[topic_a]:
            mask_a = self.masks[topic_a][layer_name].float()
            mask_b = self.masks[topic_b][layer_name].float()
            overlap = (mask_a == mask_b).float().mean().item()
            overlaps.append(overlap)
        return sum(overlaps) / len(overlaps)

    def summary(self) -> str:
        lines = ["TopicMaskRegistry Summary", "=" * 40]
        for topic_id, topic in self.topics.items():
            n_layers = len(self.masks.get(topic_id, {}))
            active = " [ACTIVE]" if topic_id == self.active_topic_id else ""
            lines.append(f"  Topic {topic_id}: {topic.name}{active}")
            lines.append(f"    Description: {topic.description}")
            lines.append(f"    Layers masked: {n_layers}")
            if topic.keywords:
                lines.append(f"    Keywords: {', '.join(topic.keywords[:5])}...")
        return "\n".join(lines)

    def save(self, path: str):
        torch.save({
            'masks': self.masks,
            'topics': self.topics,
            'global_seed': self.global_seed,
        }, path)
        print(f"TopicMaskRegistry: saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'TopicMaskRegistry':
        data = torch.load(path, map_location='cpu')
        topics = list(data['topics'].values())
        registry = cls(topics, global_seed=data['global_seed'])
        registry.masks = data['masks']
        print(f"TopicMaskRegistry: loaded from {path}")
        return registry
