"""Smoke test: synthetic board views -> calibrate -> held-out error is small (no dataset download needed)."""
import numpy as np
import cv2

import common


def synth_views(n=12, seed=1):
    rng = np.random.default_rng(seed)
    obj = np.zeros((54, 3)); obj[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2) * 25.0
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.0]]); D = np.array([-0.2, 0.05, 0, 0, 0])
    pts = {}
    for i in range(n):
        rv = rng.uniform(-0.4, 0.4, 3); tv = np.array([-100, -60, 500]) + rng.uniform(-30, 30, 3)
        p, _ = cv2.projectPoints(obj, rv, tv, K, D)
        pts[f"{i:02d}"] = p.reshape(-1, 2) + rng.normal(0, 0.1, (54, 2))
    return obj, pts


def test_heldout_rms_close_to_noise():
    obj, pts = synth_views()
    tr, te = common.split(pts, 0)
    assert len(tr) + len(te) == len(pts) and not set(tr) & set(te)
    for model in ("pinhole", "fisheye"):
        K, D, *_ = common.calibrate(model, obj, [pts[n] for n in tr], (640, 480))
        assert common.rms_of(common.view_errors(model, K, D, obj, [pts[n] for n in te])) < 0.5
