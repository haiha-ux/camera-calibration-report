"""Download public sample data (not committed). Sources: see README."""
import json, urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/opencv/{repo}/4.x/{path}"
API = "https://api.github.com/repos/opencv/opencv_extra/contents/testdata/cv/cameracalibration/fisheye/calib-3_stereo_from_JY/"


def get(url, dst):
    dst = Path(dst)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(urllib.request.urlopen(url, timeout=60).read())


def main():
    for side in ("left", "right"):  # pinhole stereo: 13 pairs (index 10 does not exist upstream)
        for i in [*range(1, 10), *range(11, 15)]:
            get(RAW.format(repo="opencv", path=f"samples/data/{side}{i:02d}.jpg"), f"data/pinhole/{side}{i:02d}.jpg")
        # fisheye stereo: 34 pairs
        items = json.load(urllib.request.urlopen(API + side, timeout=60))
        for it in items:
            get(it["download_url"], f"data/fisheye/{side}/{it['name']}")


if __name__ == "__main__":
    main()
