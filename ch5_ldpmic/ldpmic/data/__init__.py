"""Dataset loaders and partitioners.

``load(name)`` returns a ``TabularDataset`` with features in ORIGINAL units
clipped to the PUBLIC ranges documented in data/RANGES.md.
"""

from .common import (DATA_DIR, TabularDataset, client_data, dirichlet_partition, group_partition,
                     make_partition, public_scaler)

LOADERS = {}


def load(name: str, data_dir=None) -> TabularDataset:
    if name == "adult":
        from .adult import load_adult
        return load_adult(data_dir)
    if name == "bank":
        from .bank import load_bank
        return load_bank(data_dir)
    if name == "mnist":
        from .mnist import load_mnist
        return load_mnist(data_dir)
    if name == "fmnist":
        from .fmnist import load_fmnist
        return load_fmnist(data_dir)
    if name == "kdd":
        from .kdd import load_kdd
        return load_kdd(data_dir)
    if name == "cdc":
        from .cdc import load_cdc
        return load_cdc(data_dir)
    raise ValueError(f"unknown dataset {name!r}")


__all__ = ["DATA_DIR", "TabularDataset", "client_data", "dirichlet_partition", "group_partition",
           "make_partition", "public_scaler", "load"]
