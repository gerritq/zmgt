"""
Our zero-shot joint score for the runtime analysis (src.baseline_time), without the other scores of src.zero.

Same model loading and tokenisation as src.inference.Inference (no special tokens, truncation at 1024 tokens) and the
same definition as "joint" in src.zero.AngleNormScores.scores, at one transformer layer ℓ = LAYER. Over the kept tokens
t ≥ SKIP (src.zero.SKIP = 10), with θ_t = ∠(h_{t−1}^ℓ, h_t^ℓ) (within) and φ_t = ∠(h_t^{ℓ−1}, h_t^ℓ) (across):
    joint = mean_t √(θ_t φ_t), over the tokens that have both angles (t ≥ SKIP + 1)
version "raw" returns −joint (higher = machine, src.zero.SIGNS); version "ratio" divides joint by the same score at
transformer layer 1 first (the "ratio" version of src.items). A text is skipped (NaN) as in src.zero.

The forward pass runs the whole model with all hidden states, as src.zero does, although only layers ℓ − 1 and ℓ (and
0, 1 for "ratio") are used.
"""

import torch

from src.inference import Inference
from src.zero import CONTEXT_K, MAX_SKIP_SHARE, SKIP, angle

LAYER = 8  # transformer layer ℓ (hidden_states[8]; "layer_7" in src.zero's 0-based output keys)


class JointScore:
    def __init__(self, model_name: str = "l8b", layer: int = LAYER, version: str = "raw") -> None:
        if version not in ("raw", "ratio"):
            raise ValueError(f"Invalid version: {version}")
        inference = Inference(model_name=model_name)
        self.model, self.tokenizer, self.device = inference.model, inference.tokenizer, inference.device
        self.layer, self.version = layer, version
        self.settings = {"model": inference.model_id, "layer": layer, "version": version, "skip": SKIP}

    def joint(self, hidden_states: tuple[torch.Tensor, ...], layer: int) -> torch.Tensor:
        """mean_t √(θ_t φ_t) at transformer layer `layer`, on the tokens t ≥ SKIP."""
        prev = hidden_states[layer - 1][0, SKIP:].float()
        cur = hidden_states[layer][0, SKIP:].float()
        theta = angle(cur[:-1], cur[1:])  # tokens SKIP + 1, ...
        phi = angle(prev[1:], cur[1:])  # the same tokens
        return (theta * phi).sqrt().mean()

    @torch.no_grad()
    def score(self, text: str) -> float:
        inputs = self.tokenizer(text, truncation=True, add_special_tokens=False, max_length=1024,
                                return_tensors="pt").to(self.device)
        n_tokens = inputs["input_ids"].shape[-1]
        kept = n_tokens - SKIP
        if kept < (1 - MAX_SKIP_SHARE) * n_tokens or kept < CONTEXT_K + 2:  # as src.zero
            return float("nan")
        hidden_states = self.model(**inputs, output_hidden_states=True, use_cache=False).hidden_states
        value = self.joint(hidden_states, self.layer)
        if self.version == "ratio":
            value = value / self.joint(hidden_states, 1).clamp_min(1e-12)
        return float(-value)
