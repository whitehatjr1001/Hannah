"""Optional Rich compatibility layer."""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from rich.console import Console as RichConsole
    from rich.panel import Panel as RichPanel
    from rich.table import Table as RichTable
except Exception:
    RichConsole = None
    RichPanel = None
    RichTable = None


class Console:
    """Use Rich when available, otherwise fall back to plain printing."""

    def __init__(self) -> None:
        self._console = RichConsole() if RichConsole is not None else None

    def print(self, *args, **kwargs) -> None:
        if self._console is not None:
            self._console.print(*args, **kwargs)
            return
        text = " ".join(str(arg) for arg in args)
        print(text)


@dataclass
class SimplePanel:
    """Plain-text panel fallback."""

    renderable: str
    title: str = ""
    border_style: str = ""
    padding: tuple[int, int] = (0, 0)

    def __str__(self) -> str:
        return f"{self.title}\n{self.renderable}" if self.title else self.renderable


@dataclass
class SimpleTable:
    """Plain-text table fallback."""

    title: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)

    def add_column(self, header: str, style: str | None = None) -> None:
        del style
        self.columns.append(header)

    def add_row(self, *values: str) -> None:
        self.rows.append([str(value) for value in values])

    def __str__(self) -> str:
        lines = [self.title] if self.title else []
        if self.columns:
            lines.append(" | ".join(self.columns))
        lines.extend(" | ".join(row) for row in self.rows)
        return "\n".join(lines)


Panel = RichPanel or SimplePanel
Table = RichTable or SimpleTable
