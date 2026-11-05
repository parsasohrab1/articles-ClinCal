"""Cross-site study on UCI Heart Disease: train on Cleveland, test on the three other sites.

usage: python research/run_study.py /path/to/heart+disease
"""

import json
import sys
from pathlib import Path

import numpy as np

from clincal.calib import fit_model, logit, metrics, recalibrate_intercept, recalibrate_platt
from clincal.data import load_site

K_LIST = (20, 50, 100)
REPS = 200


def main(root: str):
    Xs, ys = load_site(root, "cleveland")
    model = fit_model(Xs, ys)
    out = {
        "note": "UCI Heart Disease, real patients; train = Cleveland, target = other sites; "
        "binary target = angiographic disease (num > 0)",
        "source": {"n": len(ys), **metrics(ys, model.predict_proba(Xs)[:, 1])},
        "targets": {},
    }
    # in-sample source metrics are optimistic; add a 5-fold CV view
    from sklearn.model_selection import cross_val_predict

    pcv = cross_val_predict(model, Xs, ys, cv=5, method="predict_proba")[:, 1]
    out["source_cv5"] = metrics(ys, pcv)
    rng = np.random.default_rng(0)
    for site in ("hungarian", "switzerland", "va"):
        Xt, yt = load_site(root, site)
        p = model.predict_proba(Xt)[:, 1]
        z = logit(p)
        res = {"n": len(yt), "no_recalibration": metrics(yt, p), "recalibrated": {}}
        for k in K_LIST:
            agg = {"intercept_only": [], "platt": [], "none": []}
            for _ in range(REPS):
                idx = rng.permutation(len(yt))
                c, t = idx[:k], idx[k:]
                if yt[c].min() == yt[c].max() or len(t) < 30:
                    continue
                agg["none"].append(metrics(yt[t], p[t]))
                agg["intercept_only"].append(
                    metrics(yt[t], recalibrate_intercept(z[c], yt[c])(z[t]))
                )
                agg["platt"].append(metrics(yt[t], recalibrate_platt(z[c], yt[c])(z[t])))
            res["recalibrated"][str(k)] = {
                name: {m: float(np.mean([r[m] for r in lst])) for m in ("brier", "ece", "auc")}
                for name, lst in agg.items()
                if lst
            } | {"n_reps": len(agg["none"])}
        out["targets"][site] = res
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
