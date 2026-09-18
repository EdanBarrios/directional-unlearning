"""Phase 3: full probe of a checkpoint plus the C1 to C6 control table.

Runs the probe with every alternative (all 149) and writes one results file per
checkpoint. The control table is computed here rather than in a notebook so the
pass/fail judgements live in version control and are identical for every run.

    python -m src.evaluate --checkpoint checkpoints/npo_gd_u_fwd_seed0 --out results/phase3/npo_gd_u_fwd_seed0.json
    python -m src.evaluate --checkpoint checkpoints/M1 --baseline --out results/phase3/M1.json
"""

import argparse
import json
from pathlib import Path

from src.probe import DIRECTIONS, print_summary, run
from src.util import SMOKE

# A control is a pass/fail judgement with a stated threshold, so that "did this run
# survive" is never decided after looking at the interesting numbers.
THRESHOLDS = {
    "c1_forget_fwd_accuracy_max": 0.05,   # forward direction must actually be gone
    "c2_retain_accuracy_min": 0.80,       # retain set must still work, both directions
    "c2_ppl_ratio_max": 1.50,             # perplexity may not blow up vs the baseline
    "c5_side_accuracy_min": 0.80,         # the entity must survive losing the relation
}


def controls(metrics, baseline=None):
    """C1, C2, C5 as explicit pass/fail. C3, C4, C6 are design-level (they are whole
    conditions or the metric's definition), so they are not per-run checks."""
    ppl, base_ppl = metrics.get("ppl"), (baseline or {}).get("ppl")
    ratio = ppl / base_ppl if (ppl and base_ppl) else None
    c = {
        "c1_unlearning_validity": {
            "forget_fwd_accuracy": metrics["d_fwd"]["forget"]["accuracy"],
            "threshold": THRESHOLDS["c1_forget_fwd_accuracy_max"],
            "pass": metrics["d_fwd"]["forget"]["accuracy"] <= THRESHOLDS["c1_forget_fwd_accuracy_max"],
        },
        "c2_utility_retention": {
            "retain_fwd_accuracy": metrics["d_fwd"]["retain"]["accuracy"],
            "retain_rev_accuracy": metrics["d_rev"]["retain"]["accuracy"],
            "ppl": ppl,
            "baseline_ppl": base_ppl,
            "ppl_ratio": ratio,
            "pass": (
                min(metrics["d_fwd"]["retain"]["accuracy"], metrics["d_rev"]["retain"]["accuracy"]) >= THRESHOLDS["c2_retain_accuracy_min"]
                and (ratio is None or ratio <= THRESHOLDS["c2_ppl_ratio_max"])
            ),
        },
        "c5_entity_vs_relation": {
            "forget_side_accuracy": metrics["s_fwd"]["forget"]["accuracy"],
            "threshold": THRESHOLDS["c5_side_accuracy_min"],
            "pass": metrics["s_fwd"]["forget"]["accuracy"] >= THRESHOLDS["c5_side_accuracy_min"],
        },
    }
    # The Q0 reading, only meaningful when the controls hold.
    c["q0_reverse_survival"] = {
        "forget_rev_accuracy": metrics["d_rev"]["forget"]["accuracy"],
        "forget_rev_margin_corr": metrics["d_rev"]["forget"].get("margin_corr"),
        "interpretable": c["c1_unlearning_validity"]["pass"] and c["c2_utility_retention"]["pass"] and c["c5_entity_vs_relation"]["pass"],
    }
    return c


def print_controls(c):
    for k in ("c1_unlearning_validity", "c2_utility_retention", "c5_entity_vs_relation"):
        v = c[k]
        detail = " ".join(f"{kk}={vv:.3f}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in v.items() if kk not in ("pass", "threshold"))
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k:24} {detail}")
    q = c["q0_reverse_survival"]
    mark = "" if q["interpretable"] else "  (NOT INTERPRETABLE: a control failed)"
    corr = f" corr={q['forget_rev_margin_corr']:+.3f}" if q["forget_rev_margin_corr"] is not None else ""
    print(f"  Q0    reverse accuracy on the forget set = {q['forget_rev_accuracy']:.3f}{corr}{mark}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--facts", default="data/facts.json")
    p.add_argument("--out", required=True)
    p.add_argument("--baseline", help="a Phase 3 results json (usually M1) to compare perplexity against")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()

    base_metrics = json.load(open(a.baseline))["metrics"] if a.baseline else None
    res = run(a.checkpoint, a.facts, a.out, k_alts=None, seed=a.seed, batch_size=a.batch_size, ppl=True)
    print(f"wrote {a.out}")
    print_summary(res["metrics"])

    c = controls(res["metrics"], base_metrics)
    print("controls:")
    print_controls(c)
    # Re-open and add the control table, so the probe stays a pure measurement.
    d = json.load(open(a.out))
    d["controls"] = c
    d["config"]["baseline"] = a.baseline
    json.dump(d, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
