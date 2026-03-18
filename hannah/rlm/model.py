"""RLM model stub."""

from __future__ import annotations


class RLMModel:
    """Placeholder local strategy model."""

    def generate(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 2048) -> str:
        last_user = next((message["content"] for message in reversed(messages) if message["role"] == "user"), "")
        return (
            f"[RLM scaffold] Received: '{last_user[:80]}'. "
            "Replace hannah.rlm.model.RLMModel with the trained local model."
        )

