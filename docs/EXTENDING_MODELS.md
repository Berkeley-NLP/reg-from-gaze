# Extending GazeRL: Adding New Speakers, Listeners & Rewards

This guide details how to extend the GazeRL framework with custom multimodal speaker architectures, listener models, and credit assignment rewards.

---

## 1. Adding a New Speaker Policy

All speaker models inherit from [`BaseSpeaker`](../reg/models/base.py) in `reg/models/base.py`.

### Step 1: Create Your Speaker Class
Create a new file in `reg/models/speakers/my_speaker.py`:

```python
import torch
import torch.nn as nn
from PIL import Image
from typing import Any, List, Optional
from reg.models.base import BaseSpeaker, SpeakerOutput

class MyCustomSpeaker(BaseSpeaker):
    """Custom Vision-Language Speaker Policy."""

    def __init__(self, config: Any, model: Optional[nn.Module] = None, processor: Optional[Any] = None):
        super().__init__(config)
        self.model_path = getattr(config, "model_name_or_path", "my-org/my-vlm")
        # Initialize your processor and model
        # self.processor = ...
        # self.model = ...

    def forward_tokens(self, image: Image.Image, tokens: torch.Tensor, bbox_normalized=None):
        """Forward pass computing full sequence logits and token representations."""
        # return logits, hidden_states
        pass

    def generate_step(self, image: Image.Image, prefix_tokens: torch.Tensor, bbox_normalized=None) -> SpeakerOutput:
        """Autoregressively sample next token and return SpeakerOutput."""
        # token_id = sample(...)
        # logprob = compute_logprob(...)
        # return SpeakerOutput(token_id=token_id, logprob=logprob, text=decoded_token)
        pass

    def apply_lora(self, lora_config: Any) -> None:
        """Attach PEFT LoRA adapter layers for efficient parameter fine-tuning."""
        pass
```

### Step 2: Register in `get_speaker()`
Add your architecture identifier to the factory in [`reg/models/speakers/__init__.py`](../reg/models/speakers/__init__.py):

```python
from reg.models.speakers.my_speaker import MyCustomSpeaker

def get_speaker(config, ...):
    arch = getattr(config, "architecture", "").lower()
    if arch == "my_custom":
        return MyCustomSpeaker(config=config, model=model, processor=processor)
    # ...
```

---

## 2. Adding a New Listener Model

All listener models inherit from [`BaseListener`](../reg/models/base.py).

### Step 1: Create Your Listener Class
Create `reg/models/listeners/my_listener.py`:

```python
import torch
from PIL import Image
from typing import Any, List, Optional, Tuple
from reg.models.base import BaseListener, ListenerOutput

class MyCustomListener(BaseListener):
    """Custom listener providing grounding feedback or estimated eye gaze."""

    def __init__(self, config: Any = None):
        super().__init__(config)
        # Load listener checkpoint or initialize API client

    def step(
        self,
        image: Image.Image,
        prefix_tokens: List[str],
        target_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> ListenerOutput:
        """
        Predict grounding coordinate (x, y) on 0-100 continuous canvas scale
        and verify if point lands inside target bounding box.
        """
        # x, y = predict_gaze(image, prefix_tokens)
        # is_hit = (target_bbox[0] <= x <= target_bbox[2]) and (target_bbox[1] <= y <= target_bbox[3])
        # return ListenerOutput(predicted_coords=(x, y), is_hit=is_hit)
        pass
```

### Step 2: Register in `get_listener()`
Register your listener type in [`reg/models/listeners/__init__.py`](../reg/models/listeners/__init__.py):

```python
from reg.models.listeners.my_listener import MyCustomListener

def get_listener(config, ...):
    ltype = getattr(config, "listener_type", "").lower()
    if ltype == "my_custom":
        return MyCustomListener(config=config)
    # ...
```

---

## 3. Adding a New RL Reward Function

All credit assignment reward functions inherit from [`BaseRewardFunction`](../reg/rl/rewards/base.py).

### Step 1: Implement the Reward Function
Create `reg/rl/rewards/my_reward.py`:

```python
import torch
from typing import Dict, Any, List
from reg.rl.rewards.base import BaseRewardFunction

class MyRewardFunction(BaseRewardFunction):
    """Custom step-level or sequence-level reward function."""

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.scale = getattr(config, "reward_scale", 1.0)

    def compute_rewards(
        self,
        rollouts: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Compute step-level rewards for each token in the rollout batch.
        
        Args:
            rollouts: List of rollout dictionaries containing:
                - 'is_hit': bool indicating if target bounding box was hit
                - 'predicted_coords': list of (x, y) fixation coordinates
                - 'tokens': generated token IDs
                - 'target_bbox': normalized target bounding box
                
        Returns:
            torch.Tensor of shape (batch_size, max_seq_len) containing per-step rewards.
        """
        rewards = []
        for r in rollouts:
            seq_len = len(r["tokens"])
            step_rewards = [0.0] * seq_len
            # Custom logic: assign credit based on gaze arrival time, distance, etc.
            rewards.append(torch.tensor(step_rewards))
        return torch.stack(rewards)
```

### Step 2: Register in `get_reward_function()`
Register your reward type in [`reg/rl/rewards/__init__.py`](../reg/rl/rewards/__init__.py):

```python
from reg.rl.rewards.my_reward import MyRewardFunction

def get_reward_function(config: Any) -> BaseRewardFunction:
    rtype = getattr(config, "reward_type", "").lower()
    if rtype == "my_reward":
        return MyRewardFunction(config)
    # ...
```
