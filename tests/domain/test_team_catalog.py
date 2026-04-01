"""Current-grid catalog behavior backed by the 2026 fallback roster."""

from __future__ import annotations

import pytest

from hannah.domain.teams import (
    TeamCatalogEntry,
    build_current_resolved_roster,
    canonical_driver_code,
    get_driver_codes,
    get_driver_info,
    get_team_catalog,
)


def test_team_catalog_groups_current_grid_by_team() -> None:
    catalog = get_team_catalog()

    assert isinstance(catalog["Cadillac"], TeamCatalogEntry)
    assert catalog["Cadillac"].drivers == ("PER", "BOT")
    assert catalog["Red Bull Racing"].drivers == ("VER", "HAD")
    assert catalog["Ferrari"].color == "#DC0000"


def test_current_resolved_roster_preserves_2026_driver_order_and_metadata() -> None:
    roster = build_current_resolved_roster()

    assert roster.year == 2026
    assert roster.source == "current_f1_2026_fallback"
    assert roster.driver_codes() == get_driver_codes()
    assert roster.get("HAM").team == get_driver_info("HAM").team
    assert roster.get("PER").teammate == get_driver_info("PER").teammate


def test_canonical_driver_code_rejects_unsupported_names() -> None:
    with pytest.raises(ValueError, match="unknown driver"):
        canonical_driver_code("schumacher")
