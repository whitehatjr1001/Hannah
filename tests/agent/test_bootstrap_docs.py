from __future__ import annotations

from pathlib import Path

from hannah.agent.skills import SkillsLoader


def test_bootstrap_docs_are_loaded_in_hannah_order(tmp_path) -> None:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    docs = {
        "AGENTS.md": "# Hannah Agent Instructions\n\nHannah is a CLI-first F1 strategy agent.\n",
        "TOOLS.md": "# Hannah Tools\n\nHannah exposes a small F1-focused tool surface.\n",
        "IDENTITY.md": "# Hannah Identity\n\nHannah Smith is a CLI-first F1 strategy agent acting like a virtual Red Bull race director.\n",
        "USER.md": "# Hannah User\n\nThe Hannah user wants decisive strategy help.\n",
        "SOUL.md": "# Hannah Soul\n\nHannah is fast, factual, and F1-first.\n",
    }
    for filename, content in docs.items():
        (bootstrap / filename).write_text(content, encoding="utf-8")

    loader = SkillsLoader(workspace=tmp_path, bootstrap_dir=bootstrap)
    loaded = loader.list_bootstrap_docs()

    assert [doc["name"] for doc in loaded] == ["AGENTS", "TOOLS", "IDENTITY", "USER", "SOUL"]
    assert all("Hannah" in doc["description"] for doc in loaded)

    context = loader.load_bootstrap_context()
    assert "### AGENTS" in context
    assert "virtual Red Bull race director" in context

    bundle = loader.load_bootstrap_docs(["AGENTS", "SOUL"])
    assert [doc["name"] for doc in bundle] == ["AGENTS", "SOUL"]
    assert bundle[0]["content"].startswith("# Hannah Agent Instructions")
