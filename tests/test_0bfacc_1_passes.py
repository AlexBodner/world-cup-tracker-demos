"""Regression tests for 0bfacc_1 pass/turnover detection."""

from __future__ import annotations

import pytest

from world_cup_projects.ground_truth.scan_gt_with_ball_ensemble import scan_sequence


def _pairs(items: list[dict], *, passer_key: str, receiver_key: str) -> set[tuple[int, int]]:
    return {(int(x[passer_key]), int(x[receiver_key])) for x in items}


@pytest.fixture(scope="module")
def detected_0bfacc_1() -> dict:
    try:
        return scan_sequence("0bfacc_1")
    except FileNotFoundError as exc:
        pytest.skip(f"detection cache missing: {exc}")


def test_0bfacc_1_splits_two_touch_chain(detected_0bfacc_1: dict) -> None:
    passes = _pairs(detected_0bfacc_1["passes"], passer_key="passer_tid", receiver_key="receiver_tid")
    assert (2, 4) in passes
    assert (4, 13) in passes
    assert (2, 13) not in passes


def test_0bfacc_1_aerial_pass_18_to_2(detected_0bfacc_1: dict) -> None:
    passes = _pairs(detected_0bfacc_1["passes"], passer_key="passer_tid", receiver_key="receiver_tid")
    assert (18, 2) in passes


def test_0bfacc_1_no_false_18_to_11_turnover(detected_0bfacc_1: dict) -> None:
    turnovers = _pairs(
        detected_0bfacc_1["turnovers"],
        passer_key="passer_tid",
        receiver_key="interceptor_tid",
    )
    assert (18, 11) not in turnovers
