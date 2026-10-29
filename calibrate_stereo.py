"""Stereo calibration (pinhole + fisheye): extrinsics, baseline, rectification, epipolar error on held-out pairs."""
import common  # noqa: F401  (first: sets BLAS thread env before cv2 loads)
import time

import cv2
import numpy as np

from common import DATASETS, N_SPLITS, RESULTS, calibrate, corners, dump, image_pairs, load_color, object_points, split

STEREO_MODEL = {"pinhole": "pinhole", "fisheye": "fisheye"}
CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)


def aligned_pairs(ds):
    """Pairs detected in both cams. Even x even boards (fisheye 8x6) can come back rotated 180 deg:
    flip the right corners if that makes left/right rows agree better (cameras are near-parallel)."""
    allpts, size = corners(ds)
    out, flipped = {}, []
    for n, (a, b) in allpts.items():
        if a is None or b is None:
            continue
        if np.abs(a[:, 1] - b[::-1, 1]).mean() < np.abs(a[:, 1] - b[:, 1]).mean():
            b, _ = b[::-1].copy(), flipped.append(n)
        out[n] = (a, b)
    return out, size, flipped


def rect_maps_pts(model, K1, D1, K2, D2, size, R, T):
    if model == "fisheye":
        R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(K1, D1, K2, D2, size, R, T, cv2.CALIB_ZERO_DISPARITY, newImageSize=size, balance=0.5, fov_scale=1.0)
    else:
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    return R1, R2, P1, P2


def rectify_pts(model, p, K, D, R, P):
    p = p.reshape(-1, 1, 2)
    f = cv2.fisheye.undistortPoints if model == "fisheye" else cv2.undistortPoints
    return f(p, K, D, R=R, P=P).reshape(-1, 2)


def epipolar(model, pairs, K1, D1, K2, D2, R1, R2, P1, P2):
    """|y_left - y_right| (px) of matching corners after rectification, over all corners of all pairs."""
    d = np.concatenate([np.abs(rectify_pts(model, a, K1, D1, R1, P1)[:, 1] - rectify_pts(model, b, K2, D2, R2, P2)[:, 1]) for a, b in pairs])
    return dict(mean=float(d.mean()), rms=float(np.sqrt((d ** 2).mean())), p95=float(np.percentile(d, 95)), max=float(d.max()))


def stereo_split(ds, model, obj, pairs, size, seed):
    tr, te = split(pairs, seed)
    mono = [calibrate(model, obj, [pairs[n][c] for n in tr], size) for c in (0, 1)]
    (K1, D1, r1, _, _), (K2, D2, r2, _, _) = mono
    if model == "fisheye":
        o = [obj.reshape(1, -1, 3)] * len(tr)
        i1, i2 = [pairs[n][0].reshape(1, -1, 2) for n in tr], [pairs[n][1].reshape(1, -1, 2) for n in tr]
        rms, K1, D1, K2, D2, R, T = cv2.fisheye.stereoCalibrate(o, i1, i2, K1, D1, K2, D2, size, flags=cv2.fisheye.CALIB_FIX_INTRINSIC, criteria=CRIT)[:7]
    else:
        o = [obj.astype(np.float32)] * len(tr)
        i1, i2 = [pairs[n][0].astype(np.float32) for n in tr], [pairs[n][1].astype(np.float32) for n in tr]
        rms, K1, D1, K2, D2, R, T = cv2.stereoCalibrate(o, i1, i2, K1, D1, K2, D2, size, flags=cv2.CALIB_FIX_INTRINSIC, criteria=CRIT)[:7]
    R1, R2, P1, P2 = rect_maps_pts(model, K1, D1, K2, D2, size, R, T)
    ep = lambda names: epipolar(model, [pairs[n] for n in names], K1, D1, K2, D2, R1, R2, P1, P2)
    T = np.asarray(T, float).ravel()
    return dict(train=tr, test=te, stereo_rms=float(rms), baseline_mm=float(np.linalg.norm(T)), T_mm=T.tolist(),
                R=np.asarray(R).tolist(), rotation_deg=float(np.degrees(np.linalg.norm(cv2.Rodrigues(np.asarray(R, float))[0]))),
                K1=np.asarray(K1).tolist(), D1=np.asarray(D1).ravel().tolist(), K2=np.asarray(K2).tolist(), D2=np.asarray(D2).ravel().tolist(),
                R1=R1.tolist(), R2=R2.tolist(), P1=P1.tolist(), P2=P2.tolist(),
                epipolar_train=ep(tr), epipolar_heldout=ep(te))


def save_rectified(ds, model, r, size, pair_name):
    """Side-by-side rectified held-out pair with horizontal guide lines (should cross matching features at equal height)."""
    paths = {n: (a, b) for n, a, b in image_pairs(ds)}
    a, b = (load_color(p, ds) for p in paths[pair_name])
    A = lambda k: np.array(r[k])
    if model == "fisheye":
        m1 = cv2.fisheye.initUndistortRectifyMap(A("K1"), A("D1"), A("R1"), A("P1"), size, cv2.CV_16SC2)
        m2 = cv2.fisheye.initUndistortRectifyMap(A("K2"), A("D2"), A("R2"), A("P2"), size, cv2.CV_16SC2)
    else:
        m1 = cv2.initUndistortRectifyMap(A("K1"), A("D1"), A("R1"), A("P1"), size, cv2.CV_16SC2)
        m2 = cv2.initUndistortRectifyMap(A("K2"), A("D2"), A("R2"), A("P2"), size, cv2.CV_16SC2)
    both = np.hstack([cv2.remap(a, *m1, cv2.INTER_LINEAR), cv2.remap(b, *m2, cv2.INTER_LINEAR)])
    for y in range(40, both.shape[0], 40):
        cv2.line(both, (0, y), (both.shape[1], y), (0, 255, 0), 1)
    scale = 1400 / both.shape[1]
    cv2.imwrite(str(RESULTS / f"rectified_{ds}.png"), cv2.resize(both, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))


def main():
    t0 = time.time()
    for ds, model in STEREO_MODEL.items():
        obj = object_points(ds)
        pairs, size, flipped = aligned_pairs(ds)
        out = dict(dataset=ds, model=model, pairs_used=len(pairs), pairs_total=len(image_pairs(ds)), right_corners_flipped=flipped,
                   square_mm=DATASETS[ds]["square_mm"], primary=None, failures=[], splits=[])
        for seed in range(N_SPLITS):
            try:
                r = stereo_split(ds, model, obj, pairs, size, seed)
            except (cv2.error, ValueError) as e:
                out["failures"].append(dict(seed=seed, error=str(e).strip().splitlines()[-1][:160]))
                continue
            if seed == 0:
                out["primary"] = r
                save_rectified(ds, model, r, size, r["test"][0])
            out["splits"].append(dict(baseline_mm=r["baseline_mm"], stereo_rms=r["stereo_rms"], epi_heldout_mean=r["epipolar_heldout"]["mean"],
                                      epi_heldout_rms=r["epipolar_heldout"]["rms"], epi_train_mean=r["epipolar_train"]["mean"]))
        s = {k: np.array([x[k] for x in out["splits"]]) for k in out["splits"][0]} if out["splits"] else {}
        out["summary"] = {k: dict(mean=float(v.mean()), std=float(v.std()), max=float(v.max())) for k, v in s.items()}
        out["summary"]["n_ok"], out["summary"]["n_failed"] = len(out["splits"]), len(out["failures"])
        dump(f"stereo_{ds}.json", out)
        print(ds, out["pairs_used"], "/", out["pairs_total"], out["summary"])
        print(" primary:", {k: out["primary"][k] for k in ("baseline_mm", "stereo_rms", "epipolar_train", "epipolar_heldout")} if out["primary"] else None)
    print(f"runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
