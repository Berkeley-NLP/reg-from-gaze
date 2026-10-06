"""
Molmo speaker implementation for GazeRL.
Wraps AllenAI Molmo-7B with step-by-step autoregressive generation,
token-level logprob extraction, entropy computation, and LoRA support.
"""
from typing import Dict, Any, Optional, Union, List, Tuple
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import get_peft_model, LoraConfig

from models.base import BaseSpeaker, SpeakerOutput

logger = logging.getLogger(__name__)


class MolmoSpeaker(BaseSpeaker):
    """
    Molmo-7B Speaker policy for referring expression generation.
    """

    def __init__(self, config: Any, model: Optional[nn.Module] = None, processor: Optional[Any] = None):
        super().__init__(config)
        self.temperature = getattr(config, "temperature", 1.0)
        self.top_p = getattr(config, "top_p", 0.9)
        self.max_new_tokens = getattr(config, "max_new_tokens", 20)
        self.model_path = getattr(config, "model_name_or_path", "allenai/Molmo-7B-D-0924")

        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
        else:
            self._load_model_and_processor()

    def _load_model_and_processor(self) -> None:
        logger.info("Loading Molmo speaker from %s", self.model_path)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )

    def apply_lora(self, lora_config: Any) -> None:
        """Attach LoRA adapter layers."""
        if isinstance(lora_config, dict):
            lora_cfg = LoraConfig(**lora_config)
        elif not isinstance(lora_config, LoraConfig):
            # Fallback / construct from dataclass
            lora_cfg = LoraConfig(
                r=getattr(lora_config, "r", 16),
                lora_alpha=getattr(lora_config, "lora_alpha", 32),
                lora_dropout=getattr(lora_config, "lora_dropout", 0.05),
                bias=getattr(lora_config, "bias", "none"),
                target_modules=getattr(lora_config, "target_modules", [
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
                ]),
                task_type="CAUSAL_LM",
            )
        else:
            lora_cfg = lora_config

        self.model = get_peft_model(self.model, lora_cfg)
        self.model.train()
        logger.info("LoRA attached to MolmoSpeaker. Trainable parameters:")
        self.model.print_trainable_parameters()

    def prepare_inputs(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        prompt: str,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Format image and text prompt using Molmo processor."""
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
        elif isinstance(image, np.ndarray):
            if image.shape[-1] == 4:
                image = image[:, :, :3]
            image = Image.fromarray(image)

        inputs = self.processor.process(
            images=[image],
            text=prompt,
            max_length=128,
            truncation=True,
            padding="max_length",
        )

        if device is None:
            device = next(self.model.parameters()).device

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        tensor_inputs = {}
        for k, v in inputs.items():
            if v is None:
                tensor_inputs[k] = None
                continue
            if isinstance(v, torch.Tensor):
                t = v.to(device)
            elif isinstance(v, np.ndarray):
                t = torch.from_numpy(v).to(device)
            else:
                t = torch.tensor(v, device=device)
            if k == "images":
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                t = t.to(dtype)
            else:
                if t.dim() == 1:
                    t = t.unsqueeze(0)
            tensor_inputs[k] = t

        return tensor_inputs

    def generate_referring_expression(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        validation_mode: bool = False,
    ) -> SpeakerOutput:
        """Autoregressively generate an expression while recording token logprobs."""
        max_tokens = max_new_tokens or self.max_new_tokens
        temp = temperature if temperature is not None else self.temperature
        p_val = top_p if top_p is not None else self.top_p

        inputs = self.prepare_inputs(image, prompt)
        input_ids = inputs["input_ids"]
        device = input_ids.device

        max_len = input_ids.shape[1] + max_tokens + 5
        full_ids = torch.full((1, max_len), self.processor.tokenizer.pad_token_id or 0, device=device, dtype=torch.long)
        full_ids[:, :input_ids.shape[1]] = input_ids
        input_pos = input_ids.shape[1]

        full_mask = torch.zeros_like(full_ids)
        full_mask[:, :input_ids.shape[1]] = inputs.get(
            "attention_mask",
            (input_ids != (self.processor.tokenizer.pad_token_id or 0)).long()
        )

        all_log_probs: List[torch.Tensor] = []
        all_sampled_tokens: List[torch.Tensor] = []
        all_entropies: List[torch.Tensor] = []

        past_key_values = None

        for step in range(max_tokens):
            model_kwargs = {
                "input_ids": full_ids[:, :input_pos],
                "images": inputs["images"] if step == 0 else None,
                "attention_mask": full_mask[:, :input_pos],
                "return_dict": True,
                "use_cache": True,
            }

            if past_key_values is not None:
                model_kwargs["past_key_values"] = past_key_values
                model_kwargs["input_ids"] = full_ids[:, input_pos - 1:input_pos]
                model_kwargs["position_ids"] = torch.arange(input_pos - 1, input_pos, device=device).unsqueeze(0)

            if inputs.get("image_masks") is not None:
                model_kwargs["image_masks"] = inputs["image_masks"]
            if inputs.get("image_input_idx") is not None:
                model_kwargs["image_input_idx"] = inputs["image_input_idx"]

            outputs = self.model(**model_kwargs)
            past_key_values = getattr(outputs, "past_key_values", None)

            logits = outputs.logits[:, -1, :].float()
            if temp != 1.0:
                logits = logits / max(temp, 1e-4)

            dist = torch.distributions.Categorical(logits=logits)

            if validation_mode:
                sampled_token = torch.argmax(logits, dim=-1)
            else:
                sampled_token = dist.sample()

            log_prob = dist.log_prob(sampled_token)
            entropy = dist.entropy()

            all_log_probs.append(log_prob.squeeze())
            all_sampled_tokens.append(sampled_token.squeeze())
            all_entropies.append(entropy.squeeze())

            full_ids[:, input_pos] = sampled_token
            full_mask[:, input_pos] = 1
            input_pos += 1

            if sampled_token.item() == self.processor.tokenizer.eos_token_id:
                break

        token_id_list = [t.item() for t in all_sampled_tokens]
        decoded_tokens = [self.processor.tokenizer.decode([tid], skip_special_tokens=False) for tid in token_id_list]
        full_text = self.processor.tokenizer.decode(token_id_list, skip_special_tokens=True).strip()

        token_ids_tensor = torch.tensor(token_id_list, device=device).unsqueeze(0)
        logprobs_tensor = torch.stack(all_log_probs) if all_log_probs else torch.empty(0, device=device)
        entropies_tensor = torch.stack(all_entropies) if all_entropies else torch.empty(0, device=device)

        return SpeakerOutput(
            text=full_text,
            token_ids=token_ids_tensor,
            tokens=decoded_tokens,
            logprobs=logprobs_tensor,
            entropy=entropies_tensor,
        )

    def forward_logprobs(
        self,
        inputs: Dict[str, Any],
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute token-level log probabilities for optimization."""
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -token_ids.shape[1]:, :].float()
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(token_ids)
        entropy = dist.entropy()
        return log_probs, entropy
