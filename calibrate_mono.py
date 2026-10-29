"""Single-camera calibration (pinhole, pinhole+rational, fisheye) with held-out reprojection error."""
import common  # noqa: F401  (first: sets BLAS thread env before cv2 loads)
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATASETS, MODELS, N_SPLITS, RESULTS, calibrate, corners, dump, object_points, rms_of, split, view_errors


def run_split(model, obj, pts_by_view, size, seed):
    tr, te = split(pts_by_view, seed)
    K, D, rms_cv, _, _ = calibrate(model, obj, [pts_by_view[n] for n in tr], size)
    e_tr = view_errors(model, K, D, obj, [pts_by_view[n] for n in tr])
    e_te = view_errors(model, K, D, obj, [pts_by_view[n] for n in te])
    return dict(train=tr, test=te, K=K.tolist(), D=D.ravel().tolist(), opencv_rms=rms_cv,
                train_rms=rms_of(e_tr), heldout_rms=rms_of(e_te), train_err=e_tr, test_err=e_te)


def main():
    t0 = time.time()
    for ds in DATASETS:
        obj, (allpts, size) = object_points(ds), (None, None)
        allpts, size = corners(ds)
        out = dict(dataset=ds, image_size=list(size), pattern=DATASETS[ds]["pattern"], cameras={})
        for ci, cam in enumerate(("left", "right")):
            pts = {n: v[ci] for n, v in allpts.items() if v[ci] is not None}
            out["cameras"][cam] = dict(views_detected=len(pts), views_total=len(allpts), models={})
            for model in MODELS:
                row = dict(primary=None, failures=[], splits=[])
                for seed in range(N_SPLITS):
                    try:
                        r = run_split(model, obj, pts, size, seed)
                    except (cv2.error, ValueError) as e:
                        row["failures"].append(dict(seed=seed, error=str(e).strip().splitlines()[-1][:160]))
                        continue
                    if seed == 0:
                        row["primary"] = r
                    row["splits"].append({k: r[k] for k in ("train_rms", "heldout_rms", "opencv_rms")})
                h = [s["heldout_rms"] for s in row["splits"]]
                row["summary"] = dict(n_ok=len(h), n_failed=len(row["failures"]),
                                      heldout_mean=float(np.mean(h)) if h else None, heldout_std=float(np.std(h)) if h else None,
                                      heldout_median=float(np.median(h)) if h else None, heldout_max=float(np.max(h)) if h else None)
                out["cameras"][cam]["models"][model] = row
                print(ds, cam, model, row["summary"])
        dump(f"mono_{ds}.json", out)
        plot_views(ds, out)
    print(f"runtime {time.time() - t0:.1f}s")


def plot_views(ds, out):
    """Per-view error bars (left camera, primary split); solid = train views, hatched = held-out views."""
    models = out["cameras"]["left"]["models"]
    names = sorted({n for m in models.values() if m["primary"] for n in m["primary"]["train"] + m["primary"]["test"]})
    fig, ax = plt.subplots(figsize=(12, 4))
    w = 0.27
    for k, (model, m) in enumerate(models.items()):
        p = m["primary"]
        if not p:
            continue
        err = {n: e for n, e in zip(p["train"] + p["test"], p["train_err"] + p["test_err"])}
        test = set(p["test"])
        for i, n in enumerate(names):
            if n in err:
                ax.bar(i + (k - 1) * w, err[n], w, color=f"C{k}", hatch="//" if n in test else None,
                       edgecolor="k", linewidth=0.3, label=model if i == 0 or n == next(x for x in names if x in err) else None)
    ax.set_xticks(range(len(names)), [n.replace("stereo_pair_", "") for n in names], rotation=90, fontsize=6)
    ax.set_ylabel("per-view RMS reprojection [px]")
    ax.set_title(f"{ds}: left camera, split seed 0 (hatched = held-out views)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / f"mono_per_view_{ds}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
