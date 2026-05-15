from __future__ import annotations

from .heuristic_adapter import HeuristicModel
from .mlx_adapter import MLXModel
from .ollama_adapter import OllamaModel
from .transformers_adapter import TransformersModel


def build_model(
    model_id: str,
    mlx_model: str | None = None,
    transformers_model: str | None = None,
    ollama_model: str | None = None,
):
    if model_id == "heuristic":
        return HeuristicModel()
    if model_id == "mlx":
        if not mlx_model:
            raise ValueError("--mlx-model is required when --model mlx")
        return MLXModel(mlx_model)
    if model_id == "transformers":
        if not transformers_model:
            raise ValueError("--transformers-model is required when --model transformers")
        return TransformersModel(transformers_model)
    if model_id == "ollama":
        if not ollama_model:
            raise ValueError("--ollama-model is required when --model ollama")
        return OllamaModel(ollama_model)
    raise KeyError(f"Unknown model adapter: {model_id}")
