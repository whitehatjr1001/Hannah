from __future__ import annotations

from pathlib import Path

from hannah.agent.skills import SkillsLoader


def _write_skill(root: Path, name: str, content: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_skills_loader_discovers_workspace_skills_and_filters_unavailable(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    builtin = tmp_path / "builtin"
    bootstrap = tmp_path / "bootstrap"
    workspace.mkdir()
    builtin.mkdir()
    bootstrap.mkdir()

    _write_skill(
        workspace / "skills",
        "race_data",
        """---
description: Workspace race-data skill
---
# race_data

Load race telemetry and summarize it for Hannah.
""",
    )
    _write_skill(
        builtin / "skills",
        "race_data",
        """---
description: Builtin race-data skill
---
# race_data

This content should be shadowed by the workspace skill.
""",
    )
    _write_skill(
        builtin / "skills",
        "predict_winner",
        """---
description: Requires a missing binary
requires:
  bins:
    - definitely-not-a-real-binary
---
# predict_winner

This skill should be marked unavailable.
""",
    )

    loader = SkillsLoader(
        workspace=workspace,
        builtin_skills_dir=builtin / "skills",
        bootstrap_dir=bootstrap,
    )

    skills = loader.list_skills(filter_unavailable=False)
    assert [skill["name"] for skill in skills] == ["race_data", "predict_winner"]
    assert skills[0]["source"] == "workspace"
    assert skills[1]["available"] is False

    filtered = loader.list_skills()
    assert [skill["name"] for skill in filtered] == ["race_data"]

    loaded = loader.load_skill("race_data")
    assert loaded is not None
    assert "Workspace race-data skill" in loaded

    contextual = loader.load_skills_for_context(["race_data"])
    assert "Workspace race-data skill" not in contextual
    assert "Load race telemetry and summarize it for Hannah." in contextual

    summary = loader.build_skills_summary()
    assert "<skills>" in summary
    assert "predict_winner" in summary


def test_skills_loader_build_context_bundle_includes_loaded_assets(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    bootstrap = tmp_path / "bootstrap"
    workspace.mkdir()
    bootstrap.mkdir()

    _write_skill(
        workspace / "skills",
        "race_data",
        """---
description: Race-data skill
---
Fetch race context.
""",
    )
    (bootstrap / "AGENTS.md").write_text("# Hannah Agent Instructions\n\nBe direct.\n", encoding="utf-8")

    loader = SkillsLoader(workspace=workspace, bootstrap_dir=bootstrap)
    bundle = loader.build_context_bundle(skill_names=["race_data"], bootstrap_names=["AGENTS"])

    assert bundle["skills_summary"].startswith("<skills>")
    assert bundle["bootstrap_summary"].startswith("<bootstrap_docs>")
    assert bundle["skills_context"].startswith("### Skill: race_data")
    assert bundle["bootstrap_context"].startswith("### AGENTS")

