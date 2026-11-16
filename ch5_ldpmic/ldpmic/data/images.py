"""MNIST and Fashion-MNIST via torchvision.

Pixels are scaled to [0, 1]; every one of the d = 784 coordinates has the
public range [0, 1] (the image format defines it).  Standard 60000/10000
train/test split.  Partition: Dirichlet(alpha) label skew only (there is no
natural group attribute; ``groups_train`` is the label so that a ``group``
partition would at least be well defined).

torchvision verifies the MD5 of each downloaded file against its own
hard-coded list; we additionally record the SHA256 of every file in
``meta["files"]`` and refuse files whose digest differs from
``data/MANIFEST.md`` once recorded (same policy as the tabular loaders).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .common import DATA_DIR, TabularDataset, sha256_file

MNIST_FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
               "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")

# SHA256 of the raw .gz files as downloaded from the mirrors below (recorded in
# data/MANIFEST.md).
SHA256: Dict[str, Dict[str, Optional[str]]] = {
    "mnist": {
        "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    },
    "fmnist": {
        "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
        "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
        "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
        "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
    },
}


def _load(name: str, data_dir: Optional[Path]) -> TabularDataset:
    import torchvision

    cls = {"mnist": torchvision.datasets.MNIST, "fmnist": torchvision.datasets.FashionMNIST}[name]
    root = Path(data_dir) if data_dir is not None else DATA_DIR / name
    train = cls(str(root), train=True, download=True)
    test = cls(str(root), train=False, download=True)
    raw_dir = Path(train.raw_folder)
    files = {}
    for f in MNIST_FILES:
        path = raw_dir / f
        if not path.exists():
            raise RuntimeError(f"expected torchvision to leave {path}")
        digest = sha256_file(path)
        expected = SHA256[name][f]
        if expected is None:
            print(f"WARNING: SHA256 of {name}/{f} not recorded yet; observed {digest}")
        elif digest != expected:
            raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected}, got {digest}")
        files[f] = {"url": cls.mirrors[0] + f, "sha256": digest}
    X_tr = train.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
    X_te = test.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
    y_tr = train.targets.numpy().astype(np.int64)
    y_te = test.targets.numpy().astype(np.int64)
    if X_tr.shape != (60000, 784) or X_te.shape != (10000, 784):
        raise RuntimeError(f"unexpected {name} shapes {X_tr.shape} {X_te.shape}")
    ranges = np.tile([[0.0, 1.0]], (784, 1))
    return TabularDataset(
        name=name, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te, ranges=ranges,
        feature_names=[f"px{i}" for i in range(784)], groups_train=y_tr.astype(str), group_name="label",
        n_classes=10,
        meta={"files": files, "n_train": 60000, "n_test": 10000, "d": 784,
              "classes": list(train.classes), "pixel_scale": "x/255 -> [0,1]; public range [0,1] per pixel"},
    )


def load_mnist(data_dir: Optional[Path] = None) -> TabularDataset:
    return _load("mnist", data_dir)


def load_fmnist(data_dir: Optional[Path] = None) -> TabularDataset:
    return _load("fmnist", data_dir)
