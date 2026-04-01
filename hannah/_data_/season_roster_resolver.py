"""Season-aware roster resolution across telemetry and fallback sources."""

from __future__ import annotations

from typing import Any

from hannah.domain.teams import get_driver_codes, get_driver_info


def resolve_season_roster(
    year: int,
    *,
    fastf1_payload: dict[str, Any] | None = None,
    openf1_drivers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve a race roster with source priority FastF1 -> OpenF1 -> teams(2026 only)."""
    fastf1_roster = _resolve_from_fastf1(fastf1_payload or {})
    if fastf1_roster:
        return _build_result(year, "fastf1", fastf1_roster)

    openf1_roster = _resolve_from_openf1(openf1_drivers or [])
    if openf1_roster:
        return _build_result(year, "openf1", openf1_roster)

    if year == 2026:
        return _build_result(year, "teams", _resolve_from_current_grid())

    return {
        "season": year,
        "source": "unresolved",
        "count": 0,
        "codes": [],
        "drivers": [],
    }


def summarize_resolved_roster(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"season": None, "source": "unresolved", "count": 0, "codes": []}
    codes = payload.get("codes")
    return {
        "season": payload.get("season"),
        "source": payload.get("source", "unresolved"),
        "count": payload.get("count", len(codes) if isinstance(codes, list) else 0),
        "codes": list(codes) if isinstance(codes, list) else [],
    }


def _build_result(year: int, source: str, drivers: list[dict[str, Any]]) -> dict[str, Any]:
    codes = [driver["code"] for driver in drivers]
    return {
        "season": year,
        "source": source,
        "count": len(drivers),
        "codes": codes,
        "drivers": drivers,
    }


def _resolve_from_fastf1(payload: dict[str, Any]) -> list[dict[str, Any]]:
    codes = _collect_codes(payload.get("results"), ("Abbreviation", "abbreviation", "Driver", "driver"))
    if not codes:
        codes = _collect_codes(payload.get("laps"), ("Driver", "driver", "Abbreviation", "abbreviation"))
    return [{"code": code} for code in codes]


def _resolve_from_openf1(drivers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for driver in drivers:
        if not isinstance(driver, dict):
            continue
        code = _coerce_openf1_code(driver)
        if not code or code in seen:
            continue
        seen.add(code)
        entry: dict[str, Any] = {"code": code}
        name = driver.get("full_name")
        team = driver.get("team_name")
        if isinstance(name, str) and name.strip():
            entry["name"] = name.strip()
        if isinstance(team, str) and team.strip():
            entry["team"] = team.strip()
        resolved.append(entry)
    return resolved


def _resolve_from_current_grid() -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for code in get_driver_codes():
        info = get_driver_info(code)
        resolved.append({"code": info.code, "name": info.driver, "team": info.team})
    return resolved


def _collect_codes(records: Any, fields: tuple[str, ...]) -> list[str]:
    if not isinstance(records, list):
        return []
    codes: list[str] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        code = _extract_code_from_record(record, fields)
        if not code or code in seen:
            continue
        seen.add(code)
        codes.append(code)
    return codes


def _extract_code_from_record(record: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = record.get(field)
        code = _normalize_code(value)
        if code is not None:
            return code
    return None


def _coerce_openf1_code(driver: dict[str, Any]) -> str | None:
    for field in ("name_acronym", "broadcast_name", "driver_code", "driver"):
        code = _normalize_code(driver.get(field))
        if code is not None:
            return code
    return None


def _normalize_code(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().upper()
    if not token:
        return None
    if len(token) == 3 and token.isalpha():
        return token
    if len(token) > 3 and token.isalpha():
        return token[:3]
    return None
