"""Nanobot-style skills and bootstrap-doc loader for Hannah.

This module stays content-focused:
- discover skill markdown files
- load bootstrap docs from ``hannah/docs/bootstrap``
- expose compact summaries and raw content for context assembly
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

try:  # Optional dependency, but present in Hannah's requirements.
    import yaml
except Exception:  # pragma: no cover - fallback for stripped environments
    yaml = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BOOTSTRAP_DOC_ORDER = ("AGENTS", "TOOLS", "IDENTITY", "USER", "SOUL")
_SKILL_FILE = "SKILL.md"


class SkillsLoader:
    """Discover and load Hannah skills plus bootstrap docs."""

    def __init__(
        self,
        workspace: str | Path | None = None,
        builtin_skills_dir: str | Path | None = None,
        bootstrap_dir: str | Path | None = None,
    ) -> None:
        self.workspace = Path(workspace) if workspace is not None else _REPO_ROOT
        self.workspace_skills_dir = self.workspace / "skills"
        self.builtin_skills_dir = (
            Path(builtin_skills_dir)
            if builtin_skills_dir is not None
            else _REPO_ROOT / "hannah" / "skills"
        )
        self.bootstrap_dir = (
            Path(bootstrap_dir)
            if bootstrap_dir is not None
            else _REPO_ROOT / "hannah" / "docs" / "bootstrap"
        )

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, Any]]:
        """Return discovered skills in workspace-first order."""
        discovered = self._discover_skills()
        if filter_unavailable:
            discovered = [skill for skill in discovered if skill["available"]]
        return discovered

    def load_skill(self, name: str) -> str | None:
        """Return the raw contents for a skill by name."""
        for skill_root in (self.workspace_skills_dir, self.builtin_skills_dir):
            skill_file = skill_root / name / _SKILL_FILE
            if skill_file.exists():
                return skill_file.read_text(encoding="utf-8")
        return None

    def load_skills_for_context(self, skill_names: Sequence[str]) -> str:
        """Return stripped skill bodies joined for prompt/context assembly."""
        parts: list[str] = []
        for name in skill_names:
            content = self.load_skill(name)
            if not content:
                continue
            parts.append(f"### Skill: {name}\n\n{self._strip_frontmatter(content)}")
        return "\n\n---\n\n".join(parts)

    def build_skills_summary(self) -> str:
        """Return a compact summary of discovered skills."""
        lines = ["<skills>"]
        for skill in self.list_skills(filter_unavailable=False):
            lines.append(f'  <skill available="{str(skill["available"]).lower()}">')
            lines.append(f"    <name>{self._escape_xml(skill['name'])}</name>")
            lines.append(f"    <description>{self._escape_xml(skill['description'])}</description>")
            lines.append(f"    <location>{self._escape_xml(skill['path'])}</location>")
            lines.append(f"    <source>{self._escape_xml(skill['source'])}</source>")
            lines.append("  </skill>")
        lines.append("</skills>")
        return "\n".join(lines)

    def list_bootstrap_docs(self) -> list[dict[str, Any]]:
        """Return available Hannah bootstrap docs in canonical order."""
        docs: list[dict[str, Any]] = []
        for name in self._bootstrap_doc_names():
            doc_path = self.bootstrap_dir / f"{name}.md"
            if not doc_path.exists():
                continue
            content = doc_path.read_text(encoding="utf-8")
            docs.append(
                {
                    "name": name,
                    "path": str(doc_path),
                    "source": "bootstrap",
                    "description": self._extract_summary(content),
                }
            )
        return docs

    def load_bootstrap_doc(self, name: str) -> str | None:
        """Return the raw contents of one bootstrap doc."""
        doc_path = self.bootstrap_dir / f"{name}.md"
        if not doc_path.exists():
            return None
        return doc_path.read_text(encoding="utf-8")

    def load_bootstrap_docs(self, names: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Return bootstrap docs with raw content for context assembly."""
        selected_names = tuple(names) if names is not None else self._bootstrap_doc_names()
        docs: list[dict[str, Any]] = []
        for name in selected_names:
            content = self.load_bootstrap_doc(name)
            if content is None:
                continue
            docs.append(
                {
                    "name": name,
                    "path": str(self.bootstrap_dir / f"{name}.md"),
                    "source": "bootstrap",
                    "description": self._extract_summary(content),
                    "content": self._strip_frontmatter(content),
                }
            )
        return docs

    def load_bootstrap_context(self, names: Sequence[str] | None = None) -> str:
        """Return bootstrap docs joined into a single context string."""
        parts: list[str] = []
        for doc in self.load_bootstrap_docs(names):
            parts.append(f"### {doc['name']}\n\n{doc['content']}")
        return "\n\n---\n\n".join(parts)

    def build_bootstrap_summary(self) -> str:
        """Return a compact summary of the bootstrap docs."""
        lines = ["<bootstrap_docs>"]
        for doc in self.list_bootstrap_docs():
            lines.append("  <doc>")
            lines.append(f"    <name>{self._escape_xml(doc['name'])}</name>")
            lines.append(f"    <description>{self._escape_xml(doc['description'])}</description>")
            lines.append(f"    <location>{self._escape_xml(doc['path'])}</location>")
            lines.append("  </doc>")
        lines.append("</bootstrap_docs>")
        return "\n".join(lines)

    def build_context_bundle(
        self,
        skill_names: Sequence[str] | None = None,
        bootstrap_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Convenience bundle for future prompt/context assembly."""
        selected_skills = tuple(skill_names) if skill_names is not None else tuple(
            skill["name"] for skill in self.list_skills()
        )
        all_skills = self.list_skills(filter_unavailable=False)
        ordered_skills = [
            skill
            for name in selected_skills
            for skill in all_skills
            if skill["name"] == name
        ]
        return {
            "bootstrap_docs": self.load_bootstrap_docs(bootstrap_names),
            "skills": ordered_skills,
            "bootstrap_summary": self.build_bootstrap_summary(),
            "skills_summary": self.build_skills_summary(),
            "bootstrap_context": self.load_bootstrap_context(bootstrap_names),
            "skills_context": self.load_skills_for_context(selected_skills),
        }

    def _discover_skills(self) -> list[dict[str, Any]]:
        discovered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source, root in (
            ("workspace", self.workspace_skills_dir),
            ("builtin", self.builtin_skills_dir),
        ):
            if not root.exists():
                continue
            for skill_file in sorted(root.glob(f"*/{_SKILL_FILE}")):
                name = skill_file.parent.name
                if name in seen:
                    continue
                seen.add(name)
                content = skill_file.read_text(encoding="utf-8")
                metadata, body = self._split_frontmatter(content)
                available = self._requirements_met(metadata)
                discovered.append(
                    {
                        "name": name,
                        "path": str(skill_file),
                        "source": source,
                        "description": self._skill_description(metadata, body, name),
                        "available": available,
                    }
                )
        return discovered

    def _bootstrap_doc_names(self) -> tuple[str, ...]:
        names = [name for name in _DEFAULT_BOOTSTRAP_DOC_ORDER if (self.bootstrap_dir / f"{name}.md").exists()]
        extras = sorted(
            path.stem
            for path in self.bootstrap_dir.glob("*.md")
            if path.stem not in names
        )
        return tuple(names + extras)

    def _split_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        if not content.startswith("---"):
            return {}, content
        match = re.match(r"^---\n(.*?)\n---\n?(.*)$", content, re.DOTALL)
        if match is None:
            return {}, content
        raw_metadata = match.group(1)
        body = match.group(2)
        if yaml is None:
            return {}, body
        try:
            metadata = yaml.safe_load(raw_metadata) or {}
        except Exception:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return metadata, body

    def _requirements_met(self, metadata: dict[str, Any]) -> bool:
        requires = metadata.get("requires")
        if not isinstance(requires, dict):
            return True
        bins = requires.get("bins", [])
        env = requires.get("env", [])
        for binary in bins if isinstance(bins, list) else []:
            if isinstance(binary, str) and not shutil.which(binary):
                return False
        for env_name in env if isinstance(env, list) else []:
            if isinstance(env_name, str) and not os.environ.get(env_name):
                return False
        return True

    def _skill_description(self, metadata: dict[str, Any], body: str, fallback_name: str) -> str:
        description = metadata.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
        return self._extract_summary(body) or fallback_name

    def _extract_summary(self, content: str) -> str:
        body = self._strip_frontmatter(content)
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped.startswith("```"):
                continue
            return stripped
        return ""

    def _strip_frontmatter(self, content: str) -> str:
        if not content.startswith("---"):
            return content.strip()
        match = re.match(r"^---\n.*?\n---\n?(.*)$", content, re.DOTALL)
        if match is None:
            return content.strip()
        return match.group(1).strip()

    def _escape_xml(self, text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
