import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

from platformdirs import user_cache_dir
from tqdm import tqdm

REPO = "christianhinge/NormalTwinPet"

# Add new weights here. "tag" is the GitHub release tag that hosts the asset.
WEIGHTS = {
    "test-v1": {
        "tag": "weights-test-v1",
        "filename": "ntpet_test_v1.pt",
        "sha256": "6ce940b7d8c7d7f82f62e4c25caa3992d57d42ca8c391d2af17641cda25e4c5d",
    },
}
DEFAULT_WEIGHTS = "test-v1"


def weights_dir() -> Path:
    """Weights location: $NORMALTWINPET_HOME if set, else the user cache dir."""
    home = os.environ.get("NORMALTWINPET_HOME")
    return Path(home) if home else Path(user_cache_dir("normaltwinpet"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_weights(name: str = DEFAULT_WEIGHTS) -> Path:
    """Return the local path to the weights, downloading them on first use."""
    info = WEIGHTS[name]
    path = weights_dir() / info["filename"]
    if path.exists():
        return path

    url = f"https://github.com/{REPO}/releases/download/{info['tag']}/{info['filename']}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"Downloading weights to {path}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        with tqdm(total=int(r.headers.get("Content-Length", 0)), unit="B", unit_scale=True) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    if _sha256(tmp) != info["sha256"]:
        tmp.unlink()
        raise RuntimeError(f"Checksum mismatch for downloaded weights from {url}")
    shutil.move(tmp, path)
    return path
