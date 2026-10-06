"""
Abstract base classes and output data structures for Speaker and Listener models.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Union, Sequence
import torch
import torch.nn as nn
from PIL import Image


@dataclass
class SpeakerOutput:
    """Output from speaker generation and rollout."""
    text: str
    token_ids: Union[torch.Tensor, List[int]]   # Shape: (1, seq_len) or List[int]
    tokens: List[str]                          # Decoded string token list
    logprobs: Union[torch.Tensor, List[torch.Tensor]]  # Shape: (seq_len,) or list
    entropy: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None  # Shape: (seq_len,) or list
    logits: Optional[torch.Tensor] = None

    @property
    def log_probs(self) -> List[torch.Tensor]:
        if isinstance(self.logprobs, list):
            return self.logprobs
        return [self.logprobs[i] for i in range(self.logprobs.shape[0])]

    @property
    def entropies(self) -> Optional[List[torch.Tensor]]:
        if self.entropy is None:
            return None
        if isinstance(self.entropy, list):
            return self.entropy
        return [self.entropy[i] for i in range(self.entropy.shape[0])]


@dataclass
class ListenerOutput:
    """Output from listener gaze estimation or spatial grounding."""
    gaze_points: List[Optional[Tuple[float, float]]]
    gaussian_params: Optional[List[Optional[Tuple[float, float, float]]]] = None
    hit_mask: Optional[List[bool]] = None
    is_success: bool = False
    first_hit_index: Optional[int] = None
    distances_to_bbox: Optional[List[float]] = None
    raw_output_text: Optional[str] = None

    @property
    def coordinates(self) -> List[Optional[Tuple[float, float]]]:
        return self.gaze_points


class BaseSpeaker(nn.Module, ABC):
    """Abstract base class for referring expression generation models."""

    def __init__(self, config: Any = None):
        super().__init__()
        self.config = config
        self.model: Optional[nn.Module] = None
        self.processor: Optional[Any] = None

    @abstractmethod
    def apply_lora(self, lora_config: Any) -> None:
        """Attach LoRA adapters to the model."""
        pass

    @abstractmethod
    def prepare_inputs(
        self,
        image: Union[Image.Image, torch.Tensor],
        prompt: str,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Prepare multi-modal tensor inputs for forward pass or generation."""
        pass

    @abstractmethod
    def generate_referring_expression(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> SpeakerOutput:
        """Sample a referring expression and collect token log probabilities."""
        pass

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> SpeakerOutput:
        """Convenience alias for generate_referring_expression."""
        return self.generate_referring_expression(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    @abstractmethod
    def forward_logprobs(
        self,
        inputs: Dict[str, Any],
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute token-level log probabilities and entropy under current policy parameters.

        Returns:
            Tuple of (logprobs, entropy)
        """
        pass

    def compute_log_probs(
        self,
        image: Image.Image,
        prompt: str,
        token_ids: Sequence[int],
    ) -> List[torch.Tensor]:
        """Compute per-token log probabilities under current policy."""
        inputs = self.prepare_inputs(image, prompt)
        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        token_tensor = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
        logps, _ = self.forward_logprobs(inputs, token_tensor)
        return [logps[i] for i in range(len(token_ids))]

    def save_lora(self, save_dir: str) -> None:
        """Save LoRA weights to directory."""
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_dir)
        else:
            torch.save(self.model.state_dict(), f"{save_dir}/adapter_model.bin")

    def load_lora(self, checkpoint_dir: str) -> None:
        """Load trained LoRA weights."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, checkpoint_dir)


class BaseListener(nn.Module, ABC):
    """Abstract base class for listener comprehension and gaze prediction models."""

    def __init__(self, config: Any = None):
        super().__init__()
        self.config = config
        self.model: Optional[nn.Module] = None

    @abstractmethod
    def predict_gaze_sequence(
        self,
        image: Image.Image,
        referring_expression: str,
        target_bbox_normalized: Tuple[float, float, float, float],
    ) -> ListenerOutput:
        """
        Map image and generated expression to incremental gaze scanpath.

        Args:
            image: Padded letterboxed listener image (512x320)
            referring_expression: Generated expression text
            target_bbox_normalized: Target bbox in 0-100 listener space

        Returns:
            ListenerOutput containing gaze coordinates, hits, and success
        """
        pass

    def predict(
        self,
        images: List[Image.Image],
        tokens_list: List[List[str]],
        **kwargs,
    ) -> List[ListenerOutput]:
        """Batched listener prediction over images and token lists."""
        results = []
        for img, toks in zip(images, tokens_list):
            expr = " ".join(toks)
            results.append(self.predict_gaze_sequence(img, expr, target_bbox_normalized=(0, 0, 0, 0)))
        return results
