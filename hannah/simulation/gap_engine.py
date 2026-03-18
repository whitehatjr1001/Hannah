"""Gap and pit-window calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OutcomeSummary:
    final_positions: np.ndarray
    total_time: np.ndarray


class GapEngine:
    """Compute undercut and overcut feasibility."""

    def undercut_feasibility(
        self,
        gap_to_ahead: float,
        pit_delta: float = 22.5,
        lap_delta: float = 0.8,
    ) -> dict:
        tyre_swing = max(lap_delta * 2.8, 0.6)
        required_gap = max(tyre_swing, min(pit_delta * 0.12, 3.5))
        feasible = gap_to_ahead <= required_gap
        recommendation = "box now" if feasible else "build delta first"
        return {
            "feasible": feasible,
            "required_gap": round(required_gap, 3),
            "recommendation": recommendation,
        }

    def overcut_feasibility(self, gap_to_behind: float, tyre_age_delta: int, deg_rate: float) -> dict:
        retained_margin = gap_to_behind - max(tyre_age_delta, 0) * max(deg_rate, 0.15)
        feasible = retained_margin > 0
        recommendation = "extend stint" if feasible else "cover rival"
        return {
            "feasible": feasible,
            "retained_margin": round(retained_margin, 3),
            "recommendation": recommendation,
        }

    def compute_deltas(self, all_times: np.ndarray, pit_laps: np.ndarray) -> OutcomeSummary:
        final_positions = np.argsort(all_times, axis=1)
        total_time = np.min(all_times, axis=1)
        return OutcomeSummary(final_positions=final_positions, total_time=total_time)

    def pit_window_from_samples(self, pit_laps: np.ndarray, percentile: float = 0.6) -> tuple[int, int]:
        """Return a deterministic pit window from Monte Carlo pit-lap samples."""
        if pit_laps.size == 0:
            return 0, 0
        lower = int(np.percentile(pit_laps, max(5.0, (1.0 - percentile) * 100.0)))
        upper = int(np.percentile(pit_laps, min(95.0, percentile * 100.0)))
        if upper < lower:
            upper = lower
        return lower, upper
