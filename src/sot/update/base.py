"""Abstract interface for knowledge update methods."""

from abc import ABC, abstractmethod

from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedModel


class UpdateMethod(ABC):
    """Base class for all knowledge update methods.

    All methods receive the same inputs through a common interface:
    - model: The task-tuned model to update
    - tokenizer: The model's tokenizer
    - fact_qa_pairs: Rendered QA pairs from fact triples (common NL rendering)
    - task_data: Optional replay buffer of task training data
    - cfg: Method-specific configuration
    """

    @abstractmethod
    def apply(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        task_data: list[dict] | None = None,
        cfg: DictConfig | None = None,
    ) -> PreTrainedModel:
        """Apply knowledge update and return the updated model."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable method name for logging."""
        ...
