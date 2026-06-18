"""Run the normal pass-network pipeline with a hosted-inference pitch model.

The repo's ``PitchHomography`` does ``from inference import get_model`` to load the
field-keypoint model locally. That heavy package pins ``numpy<2`` / old supervision and
would break this venv. Instead we register a lightweight ``inference`` shim in
``sys.modules`` that proxies keypoint detection to Roboflow's hosted HTTP endpoint
(``detect.roboflow.com``), which returns the same ``predictions[].keypoints[]`` shape the
repo already parses. Everything else (detection cache, homography fitting, caching,
render) runs unchanged.

Usage (from repo root)::

    PYTHONPATH=. python -m world_cup_projects.dev._pitch_hosted_shim -- <pass_network_run args...>
"""

from __future__ import annotations

import base64
import os
import runpy
import sys
import time
import types

import cv2
import requests

_HOSTED_URL = "https://detect.roboflow.com"


class _HostedKeypointModel:
    def __init__(self, model_id: str, api_key: str) -> None:
        self._url = f"{_HOSTED_URL}/{model_id}?api_key={api_key}"
        self._session = requests.Session()

    def infer(self, image, confidence: float = 0.0, **_kwargs):
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            return [{"predictions": []}]
        payload = base64.b64encode(buf.tobytes()).decode("ascii")
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                resp = self._session.post(
                    self._url,
                    data=payload,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=60,
                )
                resp.raise_for_status()
                return [resp.json()]
            except Exception as exc:  # noqa: BLE001 - transient network/HTTP
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Hosted inference failed after retries: {last_exc}")


def _get_model(model_id: str, api_key: str | None = None, **_kwargs):
    key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("ROBOFLOW_API_KEY required for hosted pitch inference.")
    return _HostedKeypointModel(model_id, key)


def _install_shim() -> None:
    shim = types.ModuleType("inference")
    shim.get_model = _get_model  # type: ignore[attr-defined]
    sys.modules["inference"] = shim


def main() -> None:
    _install_shim()
    if "--" in sys.argv:
        run_args = sys.argv[sys.argv.index("--") + 1 :]
    else:
        run_args = sys.argv[1:]
    sys.argv = ["pass_network_run", *run_args]
    runpy.run_module("player_stats.pass_network_run", run_name="__main__")


if __name__ == "__main__":
    main()
