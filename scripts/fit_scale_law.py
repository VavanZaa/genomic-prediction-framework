"""Turn the Mizzou learning curve into an answer to "how many animals do we need?"

Fits the standard genomic-prediction scaling law to the measured curve:

    r(n) = sqrt( n h2 / (n h2 + Me) )

Me, the effective number of independent chromosome segments, is the free parameter; h2 is scanned
over a plausible range and the pair with the lowest residual error is kept. The fit is then inverted
to give the training size required for a target accuracy.

This is an extrapolation from 12 measured points and inherits every limitation of the curve it fits:
it assumes the relationship keeps its shape beyond the largest training set actually run (6,000),
which is exactly where the evidence stops.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/mizzou_scale"
H2_GRID = (0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40)
TARGETS = (0.35, 0.40, 0.50, 0.60, 0.70)


def main():
    data = json.loads((OUT / "scale_curve.json").read_text(encoding="utf-8"))
    out = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source": "output/mizzou_scale/scale_curve.json",
           "model": "r(n) = sqrt(n*h2 / (n*h2 + Me))", "splits": {}}
    for split in ("random", "event_held_out"):
        pts = [(r["n_train"], r["gblup_rho"]) for r in data["curve"] if r["split"] == split]
        n = np.array([p[0] for p in pts], dtype=float)
        r = np.array([p[1] for p in pts], dtype=float)
        best = None
        for h2 in H2_GRID:
            for me in np.logspace(2.0, 6.0, 2000):
                pred = np.sqrt(n * h2 / (n * h2 + me))
                sse = float(np.sum((pred - r) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, h2, float(me))
        sse, h2, me = best
        needed = {}
        for t in TARGETS:
            req = me * t ** 2 / (h2 * (1 - t ** 2))
            needed[f"{t:.2f}"] = int(round(req)) if req < 5e6 else "beyond a practical population"
        fitted = np.sqrt(n * h2 / (n * h2 + me))
        out["splits"][split] = {
            "points_used": len(pts), "largest_training_set_measured": int(n.max()),
            "fitted_h2": h2, "fitted_Me": int(round(me)),
            "rmse": round(float(np.sqrt(sse / len(pts))), 4),
            "measured_vs_fitted": [{"n": int(a), "measured": round(float(b), 3),
                                    "fitted": round(float(c), 3)} for a, b, c in zip(n, r, fitted)],
            "animals_needed_for_accuracy": needed,
            "accuracy_at_10000": round(float(np.sqrt(10000 * h2 / (10000 * h2 + me))), 3),
            "accuracy_at_25000": round(float(np.sqrt(25000 * h2 / (25000 * h2 + me))), 3),
            "asymptote": 1.0,
        }
        s = out["splits"][split]
        print(f"\n{split}:  h2 {h2}, Me {s['fitted_Me']:,}, fit rmse {s['rmse']}")
        for row in s["measured_vs_fitted"]:
            print(f"    n={row['n']:>5}  measured {row['measured']:+.3f}   fitted {row['fitted']:+.3f}")
        print(f"    projected accuracy: {s['accuracy_at_10000']} at 10,000 animals, "
              f"{s['accuracy_at_25000']} at 25,000")
        print(f"    animals needed: {needed}")

    out["caveats"] = [
        "extrapolation beyond 6,000 training animals is model-based, not measured",
        "the fitted h2 is a curve-fitting parameter, not an independent heritability estimate",
        "Mizzou's trait is a 1-5 shedding score; Florida's traits and population differ, and in "
        "Florida an ancestry baseline is a genuine competitor whereas here it is not",
        "the random split shares relatives between train and test; the event-held-out split is the "
        "one to quote for a new herd",
    ]
    (OUT / "scale_law_fit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {(OUT / 'scale_law_fit.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
