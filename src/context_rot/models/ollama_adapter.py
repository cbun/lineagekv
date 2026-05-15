from __future__ import annotations

import json
import urllib.error
import urllib.request

from context_rot.datasets.schema import BenchmarkItem, CompressionResult


class OllamaModel:
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 600,
        disable_thinking: bool = True,
    ):
        self.model_id = f"ollama:{model_id}"
        self.ollama_model = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.disable_thinking = disable_thinking

    def generate(self, prompt: str, item: BenchmarkItem, compression: CompressionResult) -> str:
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": not self.disable_thinking,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. Start Ollama or choose a different model adapter."
            ) from exc
        message = data.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Ollama response shape: {data}")
        return content
