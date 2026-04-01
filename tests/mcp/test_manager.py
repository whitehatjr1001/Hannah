"""Tests for the optional MCP manager and registry defaults."""

from __future__ import annotations

from hannah.mcp.manager import MCPManager
from hannah.mcp.registry import MCPRegistry


def test_mcp_manager_defaults_to_disabled_without_servers() -> None:
    manager = MCPManager()

    assert manager.enabled is False
    assert manager.server_names == []


def test_mcp_registry_is_empty_when_disabled() -> None:
    registry = MCPRegistry(MCPManager())

    assert registry.enabled is False
    assert registry.get_tool_specs() == []
    assert registry.tool_names() == set()
