from __future__ import annotations

from context_rot.datasets.schema import BenchmarkItem, CompressionResult


class MLXModel:
    def __init__(self, model_id: str, max_tokens: int = 256, temperature: float = 0.0):
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        self.model_id = f"mlx:{model_id}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._generate = generate
        self._sampler = make_sampler(temp=temperature, top_p=1.0)
        self._model, self._tokenizer = load(model_id)

    def generate(self, prompt: str, item: BenchmarkItem, compression: CompressionResult) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            model_prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            model_prompt = prompt
        return self._generate(
            self._model,
            self._tokenizer,
            prompt=model_prompt,
            max_tokens=self.max_tokens,
            sampler=self._sampler,
            verbose=False,
        )
