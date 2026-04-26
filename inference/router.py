"""
dv4/inference/router.py

TopicRouter — classifies input text to a topic and activates the
correct flip mask for inference.

PoC implementation: keyword-based router (fast, interpretable, no extra model).
Production path: replace with a small trained classifier.

The router can switch topics mid-sequence if the topic changes between tokens.
For the PoC, switching happens per-prompt (not per-token) — simpler to validate.

Author: Peter Norman / twoswans.com.au
Architecture: DV4 (Dual-Vocab 4-bit)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from ..model.topic_masks import TopicMaskRegistry, TopicDefinition


# ---------------------------------------------------------------------------
# Keyword-based router (PoC)
# ---------------------------------------------------------------------------

class KeywordRouter:
    """
    Simple keyword-based topic router for PoC validation.
    
    Scores each topic by keyword match count in the input text.
    Returns the topic with the highest score, defaulting to topic_id=1
    (general) when no math keywords are found.
    
    This is intentionally simple — the goal for the PoC is to validate
    whether the flip mask switching *works*, not to build a perfect router.
    A neural router is the production upgrade path.
    """

    def __init__(self, registry: TopicMaskRegistry):
        self.registry = registry
        # Build keyword sets per topic (lowercased for matching)
        self.topic_keywords = {
            topic_id: set(kw.lower() for kw in topic.keywords)
            for topic_id, topic in registry.topics.items()
        }

    def route(self, text: str) -> int:
        """
        Route a text string to a topic ID.
        
        Args:
            text: Input text (prompt or partial sentence)
        
        Returns:
            topic_id of the best matching topic
        """
        text_lower = text.lower()
        words = set(text_lower.split())

        scores = {}
        for topic_id, keywords in self.topic_keywords.items():
            if keywords:
                # Count keyword matches (word-level)
                score = len(words & keywords)
                # Also check for substring matches for multi-word terms
                score += sum(1 for kw in keywords if kw in text_lower and kw not in words)
                scores[topic_id] = score
            else:
                scores[topic_id] = 0  # Default/fallback topic scores 0

        # Find the topic with the highest score
        best_topic = max(scores, key=lambda t: scores[t])

        # If best score is 0, fall back to the default topic (highest id = general)
        if scores[best_topic] == 0:
            # Default: last topic in registry (general)
            best_topic = max(self.registry.topics.keys())

        return best_topic

    def route_with_confidence(self, text: str) -> Tuple[int, float]:
        """
        Route with a confidence score (0-1).
        
        Confidence is the fraction of total keyword matches accounted
        for by the winning topic. Low confidence = ambiguous input.
        
        Returns:
            (topic_id, confidence)
        """
        text_lower = text.lower()
        words = set(text_lower.split())

        scores = {}
        for topic_id, keywords in self.topic_keywords.items():
            if keywords:
                scores[topic_id] = len(words & keywords)
            else:
                scores[topic_id] = 0

        total = sum(scores.values())
        if total == 0:
            default_topic = max(self.registry.topics.keys())
            return default_topic, 0.0

        best_topic = max(scores, key=lambda t: scores[t])
        confidence = scores[best_topic] / total

        return best_topic, confidence


# ---------------------------------------------------------------------------
# Mid-sentence router (experimental)
# ---------------------------------------------------------------------------

class SentenceLevelRouter:
    """
    Routes at the sentence level within a longer input.
    
    Splits input on sentence boundaries, routes each sentence,
    and returns a list of (sentence, topic_id) pairs.
    
    This is the foundation for mid-sentence topic switching at inference.
    For the PoC, this is used for analysis only — the generation loop
    (generate.py) handles the actual mask switching.
    """

    def __init__(self, base_router: KeywordRouter):
        self.router = base_router

    def segment(self, text: str) -> List[Tuple[str, int]]:
        """
        Split text into segments and route each to a topic.
        
        Args:
            text: Full input text
        
        Returns:
            List of (segment_text, topic_id) pairs
        """
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if s.strip()]

        segments = []
        for sentence in sentences:
            topic_id = self.router.route(sentence)
            segments.append((sentence, topic_id))

        return segments

    def detect_topic_change(self, text: str) -> bool:
        """
        Returns True if the text contains a mid-sentence topic change.
        Useful for deciding whether to enable fine-grained switching.
        """
        segments = self.segment(text)
        if len(segments) <= 1:
            return False
        topic_ids = [t for _, t in segments]
        return len(set(topic_ids)) > 1


# ---------------------------------------------------------------------------
# Inference router — ties routing to mask switching
# ---------------------------------------------------------------------------

class InferenceRouter:
    """
    High-level router used during inference.
    
    Combines topic classification with flip mask activation.
    Call route_and_activate() before each generation call to ensure
    the correct mask is active.
    """

    def __init__(
        self,
        registry: TopicMaskRegistry,
        model: nn.Module,
    ):
        self.registry = registry
        self.model = model
        self.keyword_router = KeywordRouter(registry)
        self.sentence_router = SentenceLevelRouter(self.keyword_router)
        self.current_topic_id: Optional[int] = None

    def route_and_activate(self, text: str, verbose: bool = False) -> int:
        """
        Route text to a topic and activate the corresponding flip mask.
        
        Args:
            text:    Input text to classify
            verbose: Print routing decision
        
        Returns:
            Activated topic_id
        """
        topic_id, confidence = self.keyword_router.route_with_confidence(text)

        if topic_id != self.current_topic_id:
            self.registry.set_active_topic(self.model, topic_id)
            self.current_topic_id = topic_id

            if verbose:
                topic_name = self.registry.topics[topic_id].name
                print(f"Router: → topic={topic_name} (id={topic_id}, confidence={confidence:.2f})")

        return topic_id

    def get_current_topic(self) -> Optional[TopicDefinition]:
        """Return the currently active topic definition."""
        if self.current_topic_id is None:
            return None
        return self.registry.topics[self.current_topic_id]
