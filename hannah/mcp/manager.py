"""MCP server configuration and lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from hannah.utils.console import Console

console = Console()


@dataclass(frozen=True)
class MCPServerConfig:
    """Minimal server config used to keep MCP optional."""

    name: str
    config: dict[str, Any]


class MCPManager:
    """Store optional MCP server configuration without forcing transport setup."""

    def __init__(
        self,
        servers: Sequence[Mapping[str, Any]] | None = None,
        *,
        config_path: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path or "config.yaml")
        self._servers = self._load_servers(servers)

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "MCPManager":
        return cls(config_path=path)

    @property
    def enabled(self) -> bool:
        return bool(self._servers)

    @property
    def server_names(self) -> list[str]:
        return [server.name for server in self._servers]

    def describe_servers(self) -> list[dict[str, Any]]:
        return [{"name": server.name, **server.config} for server in self._servers]

    def _load_servers(
        self,
        explicit_servers: Sequence[Mapping[str, Any]] | None,
    ) -> list[MCPServerConfig]:
        if explicit_servers is not None:
            return self._coerce_servers(explicit_servers)
        if not self.config_path.exists():
            return []
        try:
            with self.config_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
        except Exception as err:  # pragma: no cover - defensive config read
            console.print(f"[yellow]warning:[/yellow] failed to read MCP config: {err}")
            return []

        mcp_raw = raw.get("mcp", {})
        if not isinstance(mcp_raw, dict):
            return []
        servers = mcp_raw.get("servers", [])
        if not isinstance(servers, Sequence) or isinstance(servers, (str, bytes)):
            return []
        return self._coerce_servers(servers)

    def _coerce_servers(self, servers: Sequence[Mapping[str, Any]]) -> list[MCPServerConfig]:
        coerced: list[MCPServerConfig] = []
        for index, server in enumerate(servers):
            if not isinstance(server, Mapping):
                continue
            raw_name = server.get("name")
            name = str(raw_name).strip() if raw_name is not None else ""
            if not name:
                name = f"server_{index + 1}"
            config = {str(key): value for key, value in server.items() if key != "name"}
            coerced.append(MCPServerConfig(name=name, config=config))
        return coerced
