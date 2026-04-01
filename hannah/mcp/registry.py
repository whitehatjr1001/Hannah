"""In-memory MCP tool registry used by the main Hannah tool registry."""

from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from hannah.mcp.manager import MCPManager


@dataclass(frozen=True)
class MCPToolBinding:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    server_name: str = "mcp"


class MCPRegistry:
    """Optional MCP-backed tool surface.

    This class keeps discovery and execution separate from the native tool registry.
    The main registry can merge these bindings when MCP is enabled without taking
    ownership of transport setup.
    """

    def __init__(self, manager: MCPManager | None = None) -> None:
        self.manager = manager or MCPManager.from_config()
        self._tools: dict[str, MCPToolBinding] = {}

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "MCPRegistry":
        return cls(MCPManager.from_config(path))

    @property
    def enabled(self) -> bool:
        return self.manager.enabled or bool(self._tools)

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any] | None,
        handler: Callable[..., Any],
        server_name: str = "mcp",
    ) -> None:
        if not callable(handler):
            raise TypeError(f"handler for MCP tool {name} must be callable")
        normalized_parameters = parameters if isinstance(parameters, dict) else {"type": "object", "properties": {}}
        self._tools[name] = MCPToolBinding(
            name=name,
            description=description,
            parameters=deepcopy(normalized_parameters),
            handler=handler,
            server_name=server_name,
        )

    def registered_tools(self) -> list[MCPToolBinding]:
        return list(self._tools.values())

    def tool_names(self) -> set[str]:
        return set(self._tools)

    def get_tool_specs(self) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for tool in self._tools.values():
            parameters = deepcopy(tool.parameters)
            if parameters.get("type") == "object" and "additionalProperties" not in parameters:
                parameters["additionalProperties"] = False
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": parameters,
                    },
                }
            )
        return specs

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            state = "enabled" if self.enabled else "disabled"
            raise ValueError(f"unknown MCP tool: {name} ({state})")
        result = tool.handler(**args)
        if inspect.isawaitable(result):
            resolved = await result
        else:
            resolved = result
        if not isinstance(resolved, dict):
            raise TypeError(f"MCP tool {name} returned {type(resolved).__name__}, expected dict")
        return resolved
