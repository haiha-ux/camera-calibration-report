"""Shared helpers: datasets, corner detection (cached), splits, held-out error."""
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # tiny problems: BLAS threads only thrash on a busy machine
import cv2
import numpy as np

cv2.setNumThreads(1)  # small problems; avoids thread thrash on busy machines
ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

# pattern = inner corners (cols, rows); square_mm is ASSUMED/taken from the dataset, see README.
DATASETS = {
    "pinhole": dict(pattern=(9, 6), square_mm=25.0, blur_rows=0),
    "fisheye": dict(pattern=(8, 6), square_mm=24.4, blur_rows=330),  # blur_rows: hides a bystander's face in figures only
}
N_SPLITS = 20      # random 70/30 splits used for the mean/std summary
TRAIN_FRAC = 0.7
MODELS = ("pinhole", "pinhole_rational", "fisheye")


def image_pairs(ds):
    d = ROOT / "data" / ds
    if ds == "pinhole":
        return [(f"{i:02d}", d / f"left{i:02d}.jpg", d / f"right{i:02d}.jpg") for i in [*range(1, 10), *range(11, 15)]]
    return [(p.stem, p, d / "right" / p.name) for p in sorted((d / "left").glob("*.jpg"))]


def load_color(path, ds):
    """Color image for figures; top rows blurred for the fisheye set (privacy)."""
    im = cv2.imread(str(path))
    n = DATASETS[ds]["blur_rows"]
    if n:
        im[:n] = cv2.GaussianBlur(im[:n], (0, 0), 25)
    return im


def object_points(ds):
    c, r = DATASETS[ds]["pattern"]
    p = np.zeros((c * r, 3), np.float64)
    p[:, :2] = np.mgrid[:c, :r].T.reshape(-1, 2) * DATASETS[ds]["square_mm"]
    return p


def detect(path, pattern):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    ok, c = cv2.findChessboardCorners(g, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:  # sector-based detector is slower but more robust on dark/blurry views
        ok, c = cv2.findChessboardCornersSB(g, pattern, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not ok:
            return None, g.shape[::-1]
    else:
        c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    return c.reshape(-1, 2).astype(np.float64), g.shape[::-1]


def corners(ds):
    """{name: (left_pts|None, right_pts|None)}, image_size (w,h). Cached in results/corners_<ds>.json."""
    cache = RESULTS / f"corners_{ds}.json"
    if cache.exists():
        j = json.loads(cache.read_text())
        f = lambda v: None if v is None else np.array(v)
        return {k: (f(a), f(b)) for k, (a, b) in j["pts"].items()}, tuple(j["size"])
    pat, out, size = DATASETS[ds]["pattern"], {}, None
    for name, lp, rp in image_pairs(ds):
        (a, size), (b, _) = detect(lp, pat), detect(rp, pat)
        out[name] = (a, b)
    RESULTS.mkdir(exist_ok=True)
    cache.write_text(json.dumps({"size": list(size), "pts": {k: [None if x is None else x.tolist() for x in v] for k, v in out.items()}}))
    return out, size


def split(names, seed):
    """Deterministic 70/30 split of view names."""
    rng = np.random.default_rng(seed)
    n = list(names)
    rng.shuffle(n)
    k = max(3, int(round(TRAIN_FRAC * len(n))))
    return sorted(n[:k]), sorted(n[k:])


def _is_fisheye(model):
    return model == "fisheye"


def calibrate(model, obj, pts, size):
    """-> (K, D, opencv_rms, rvecs, tvecs). Raises cv2.error/ValueError on failure."""
    if _is_fisheye(model):
        o = [obj.reshape(1, -1, 3)] * len(pts)
        i = [p.reshape(1, -1, 2) for p in pts]
        K, D = np.eye(3), np.zeros((4, 1))
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rms, K, D, rv, tv = cv2.fisheye.calibrate(o, i, size, K, D, flags=flags,
                                                  criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9))
    else:
        flags = cv2.CALIB_RATIONAL_MODEL if model == "pinhole_rational" else 0
        rms, K, D, rv, tv = cv2.calibrateCamera([obj.astype(np.float32)] * len(pts), [p.astype(np.float32) for p in pts],
                                                size, None, None, flags=flags)
    if not (np.isfinite(K).all() and np.isfinite(D).all()):
        raise ValueError("non-finite calibration")
    return K, D, float(rms), rv, tv


def view_errors(model, K, D, obj, pts):
    """Per-view RMS px: pose re-estimated from the view's own corners with fixed intrinsics (same for train and held-out)."""
    errs = []
    for p in pts:
        if _is_fisheye(model):
            u = cv2.fisheye.undistortPoints(p.reshape(-1, 1, 2), K, D)
            ok, rv, tv = cv2.solvePnP(obj, u.reshape(-1, 2), np.eye(3), None)
            proj, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rv, tv, K, D)
        else:
            ok, rv, tv = cv2.solvePnP(obj, p, K, D)
            proj, _ = cv2.projectPoints(obj, rv, tv, K, D)
        errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - p) ** 2, axis=1)))))
    return errs


def rms_of(errs):
    return float(np.sqrt(np.mean(np.square(errs))))  # all views have equal point counts


def dump(name, obj):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))
