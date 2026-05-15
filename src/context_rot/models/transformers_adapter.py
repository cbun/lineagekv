from __future__ import annotations

from context_rot.datasets.schema import BenchmarkItem, CompressionResult


class TransformersModel:
    def __init__(self, model_id: str, max_tokens: int = 96, temperature: float = 0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = f"transformers:{model_id}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            torch_dtype=self._torch.float32,
            device_map=None,
        )
        self._model.eval()

    def generate(self, prompt: str, item: BenchmarkItem, compression: CompressionResult) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            model_prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            model_prompt = prompt
        inputs = self._tokenizer(model_prompt, return_tensors="pt", truncation=True, max_length=4096)
        with self._torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-6) if self.temperature > 0 else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)
