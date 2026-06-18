"""Regression tests for 08fd33_8 pass/turnover detection."""

from __future__ import annotations

import pytest

from world_cup_projects.ground_truth.scan_gt_with_ball_ensemble import scan_sequence


def _pairs(items: list[dict], *, passer_key: str, receiver_key: str) -> set[tuple[int, int]]:
    return {(int(x[passer_key]), int(x[receiver_key])) for x in items}


@pytest.fixture(scope="module")
def detected_08fd33_8() -> dict:
    try:
        return scan_sequence("08fd33_8")
    except FileNotFoundError as exc:
        pytest.skip(f"detection cache missing: {exc}")


def test_08fd33_8_post_intercept_pass(detected_08fd33_8: dict) -> None:
    passes = _pairs(
        detected_08fd33_8["passes"], passer_key="passer_tid", receiver_key="receiver_tid"
    )
    assert (23, 12) in passes


def test_08fd33_8_no_false_turnover_on_same_reception(detected_08fd33_8: dict) -> None:
    """#12 receiving from #23 must not also count as #13 losing to #12."""
    turnovers = _pairs(
        detected_08fd33_8["turnovers"],
        passer_key="passer_tid",
        receiver_key="interceptor_tid",
    )
    assert (13, 12) not in turnovers
