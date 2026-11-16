"""Cross-check of src/mic.py against minepy 1.2.6 (the reference C implementation).

Not part of the experiment pipeline (src.run excludes the `crosscheck` marker); skipped when minepy
is not installed. Run `python -m pytest -m crosscheck tests/test_mic_vs_minepy.py`, or
`python -m tests.test_mic_vs_minepy` to also write results/ch4/mic_crosscheck.csv on the
experiment's own score subsamples.
"""
import numpy as np
import pytest

from src import mic as M

minepy = pytest.importorskip("minepy")
pytestmark = pytest.mark.crosscheck


def _ref(x, y, alpha, c, est):
    m = minepy.MINE(alpha=alpha, c=c, est=est)
    m.compute_score(np.ascontiguousarray(x, dtype=float), np.ascontiguousarray(y, dtype=float))
    return m.mic()


CASES = ["linear", "parabola", "sine", "independent", "binary", "ties", "step", "circle"]


def _case(name, rng, n=2000):
    x = rng.uniform(-1, 1, n)
    e = rng.normal(size=n)
    return {"linear": (x, 2 * x + 0.5 * e), "parabola": (x, x ** 2 + 0.2 * e),
            "sine": (x, np.sin(6 * x) + 0.5 * e), "independent": (x, e),
            "binary": (x, (x + e > 0).astype(float)), "ties": (np.round(3 * x), x + e),
            "step": (x, (x > 0).astype(float) + 0.3 * e),
            "circle": (np.cos(np.pi * x) + 0.05 * e, np.sin(np.pi * x) * np.sign(rng.uniform(-1, 1, n)))}[name]


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("est,alpha,c", [("mic_e", 0.55, 5), ("mic_approx", 0.6, 15)])
def test_matches_minepy(name, est, alpha, c):
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    x, y = _case(name, rng)
    assert M.mic(x, y, alpha=alpha, c=c, est=est) == pytest.approx(_ref(x, y, alpha, c, est), abs=1e-9)


def main():
    import pandas as pd
    from src import data as D
    from src.scores import subsample_rows
    cfg = D.load_config()
    rows = []
    for name in D.all_dataset_names(cfg):
        p = D.prepare(name, cfg)
        i = subsample_rows(len(p.y_train), cfg["scores"]["max_rows"], cfg["seeds"]["score_subsample"])
        X, y = p.X_train[i], p.y_train[i]
        a = M.alpha_for_n(len(y))
        for j, f in enumerate(p.features):
            ours = M.mic(X[:, j], y, alpha=a, c=5, est="mic_e")
            ref = _ref(X[:, j], y, a, 5, "mic_e")
            rows.append({"dataset": name, "feature": f, "n": len(y), "alpha": a, "mic_e_own": ours,
                         "mic_e_minepy": ref, "abs_diff": abs(ours - ref)})
    df = pd.DataFrame(rows)
    out = D.ROOT / cfg["outputs"]["dir"] / "mic_crosscheck.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} feature-target pairs; max |own - minepy| = {df.abs_diff.max():.3e}; wrote {out}")


if __name__ == "__main__":
    main()
