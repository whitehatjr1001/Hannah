"""Registry integration tests for optional MCP-backed tools."""

from __future__ import annotations

import asyncio

from hannah.agent.tool_registry import ToolRegistry
from hannah.mcp.manager import MCPManager
from hannah.mcp.registry import MCPRegistry


def test_tool_registry_merges_injected_mcp_tools() -> None:
    registry = MCPRegistry(MCPManager())

    async def _run(**kwargs):
        return {"source": "mcp", "kwargs": kwargs}

    registry.register_tool(
        name="mcp_bridge",
        description="Bridge to a fake MCP tool.",
        parameters={
            "type": "object",
            "properties": {
                "race": {"type": "string"},
                "lap": {"type": "integer"},
            },
            "required": ["race"],
        },
        handler=_run,
    )

    tool_registry = ToolRegistry(mcp_registry=registry)

    specs = {tool["function"]["name"] for tool in tool_registry.get_tool_specs()}
    assert "mcp_bridge" in specs


def test_tool_registry_dispatches_to_mcp_tool() -> None:
    registry = MCPRegistry(MCPManager())
    seen: dict[str, object] = {}

    async def _run(**kwargs):
        seen.update(kwargs)
        return {"source": "mcp", "kwargs": kwargs}

    registry.register_tool(
        name="mcp_bridge",
        description="Bridge to a fake MCP tool.",
        parameters={
            "type": "object",
            "properties": {
                "race": {"type": "string"},
                "lap": {"type": "integer"},
            },
            "required": ["race"],
        },
        handler=_run,
    )

    tool_registry = ToolRegistry(mcp_registry=registry)
    result = asyncio.run(tool_registry.call("mcp_bridge", {"race": "bahrain", "lap": "18"}))

    assert result == {"source": "mcp", "kwargs": {"race": "bahrain", "lap": 18}}
    assert seen == {"race": "bahrain", "lap": 18}
