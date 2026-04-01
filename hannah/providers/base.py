"""Provider contracts and response adapters for Hannah."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ProviderFunctionCall:
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ProviderToolCall:
    id: str
    function: ProviderFunctionCall
    type: str = "function"

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


@dataclass(frozen=True, slots=True)
class ProviderMessage:
    role: str
    content: str
    tool_calls: list[ProviderToolCall] | None = None
    name: str | None = None
    tool_call_id: str | None = None

    def model_dump(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


class CompletionProvider(Protocol):
    """Provider contract consumed by the agent loop and sub-agents."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> object:
        ...


def coerce_provider_message(response: object) -> ProviderMessage:
    message = extract_first_message(response)
    if message is None:
        return ProviderMessage(
            role="assistant",
            content="I could not parse the model response.",
        )
    return message_to_provider_message(message)


def extract_first_message(response: object) -> object | None:
    if hasattr(response, "choices") and getattr(response, "choices"):
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        if message is not None:
            return message

    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                return first_choice.get("message")
            if hasattr(first_choice, "message"):
                return getattr(first_choice, "message", None)
    return None


def message_to_provider_message(message: object) -> ProviderMessage:
    if isinstance(message, dict):
        payload = message
    elif hasattr(message, "model_dump"):
        payload = message.model_dump()
    else:
        payload = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", ""),
            "tool_calls": getattr(message, "tool_calls", None),
            "name": getattr(message, "name", None),
            "tool_call_id": getattr(message, "tool_call_id", None),
        }

    if not isinstance(payload, dict):
        return ProviderMessage(
            role="assistant",
            content="I could not parse the model response.",
        )
    return payload_to_provider_message(payload)


def payload_to_provider_message(message: dict[str, Any]) -> ProviderMessage:
    tool_calls = coerce_tool_calls(message.get("tool_calls"))
    content = normalize_message_content(
        message.get("content"),
        has_tool_calls=bool(tool_calls),
    )
    return ProviderMessage(
        role=str(message.get("role", "assistant")),
        content=content,
        tool_calls=tool_calls or None,
        name=message.get("name"),
        tool_call_id=message.get("tool_call_id"),
    )


def coerce_tool_calls(tool_calls_payload: Any) -> list[ProviderToolCall]:
    tool_calls: list[ProviderToolCall] = []
    for index, raw_call in enumerate(tool_calls_payload or []):
        call = coerce_payload(raw_call)
        if call is None:
            continue
        function = coerce_payload(call.get("function"))
        if function is None:
            continue
        name = str(function.get("name", "")).strip()
        if not name:
            continue
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, default=str)
        tool_calls.append(
            ProviderToolCall(
                id=str(call.get("id", f"dict-tool-call-{index + 1}")),
                function=ProviderFunctionCall(name=name, arguments=arguments),
                type=str(call.get("type", "function")),
            )
        )
    return tool_calls


def coerce_payload(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        dumped = payload.model_dump()
        return dumped if isinstance(dumped, dict) else None
    if payload is None:
        return None
    result: dict[str, Any] = {}
    for key in ("id", "type", "function", "role", "content", "tool_calls", "name", "tool_call_id"):
        if hasattr(payload, key):
            result[key] = getattr(payload, key)
    return result or None


def normalize_message_content(content: Any, *, has_tool_calls: bool) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if has_tool_calls:
            return ""
        return flatten_content_blocks(content)
    if isinstance(content, dict):
        if has_tool_calls:
            return ""
        return flatten_content_blocks([content])
    if has_tool_calls:
        return ""
    return str(content)


def flatten_content_blocks(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            if isinstance(text, dict):
                nested = text.get("value") or text.get("text")
                if isinstance(nested, str):
                    parts.append(nested)
                    continue
            value = block.get("value")
            if isinstance(value, str):
                parts.append(value)
                continue
        else:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part).strip()
