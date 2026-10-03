"""Phase 3: full probe of a checkpoint plus the C1 to C6 control table.

Runs the probe with every alternative (all 149) and writes one results file per
checkpoint. The control table is computed here rather than in a notebook so the
pass/fail judgements live in version control and are identical for every run.

    python -m src.evaluate --checkpoint checkpoints/M1 --out results/phase3/M1.json
    python -m src.evaluate --checkpoint checkpoints/npo_gd_u_fwd_seed0 --baseline results/phase3/M1.json --out results/phase3/npo_gd_u_fwd_seed0.json
    python -m src.evaluate --checkpoint checkpoints/npo_gd_u_rev_seed0 --baseline results/phase3/M1.json --target d_rev --out ...
"""

import argparse
import json
from pathlib import Path

from src.probe import DIRECTIONS, GATE_CORPUS, print_summary, run
from src.util import SMOKE

# A control is a pass/fail judgement with a stated threshold, so that "did this run
# survive" is never decided after looking at the interesting numbers. All accuracies are
# candidate accuracy (rank 1 of 150, decision 21). C2 and C5 are ratios against M1, as
# the perplexity check already was: an absolute 0.80 would test M1's ceiling, not damage.
THRESHOLDS = {
    "c1_forget_fwd_cand_accuracy_max": 0.05,  # forward must be gone; chance is 1/150
    "c2_retain_ratio_min": 0.80,              # retain candidate accuracy vs M1, both directions
    "c2_ppl_ratio_max": 1.50,                 # on the gate corpus, which replay never touches
    "c5_side_ratio_min": 0.80,                # forget-set side attribute vs M1
}
ACC = "cand_accuracy"


def controls(metrics, baseline, target="d_fwd"):
    """C1, C2, C5 as explicit pass/fail against M1's metrics. C3, C4, C6 are design-level
    (they are whole conditions or the metric's definition), so they are not per-run checks.

    `target` is the direction this condition unlearned (d_rev for u_rev); C1 checks it and
    Q0 reads the other one."""
    other = "d_rev" if target == "d_fwd" else "d_fwd"
    ratio = lambda d, s: metrics[d][s][ACC] / baseline[d][s][ACC] if baseline[d][s][ACC] else None
    ppl_ratio = metrics["ppl"] / baseline["ppl"]
    r_fwd, r_rev, r_side = ratio("d_fwd", "retain"), ratio("d_rev", "retain"), ratio("s_fwd", "forget")
    c = {
        "c1_unlearning_validity": {
            "direction": target,
            "forget_cand_accuracy": metrics[target]["forget"][ACC],
            "threshold": THRESHOLDS["c1_forget_fwd_cand_accuracy_max"],
            "pass": metrics[target]["forget"][ACC] <= THRESHOLDS["c1_forget_fwd_cand_accuracy_max"],
        },
        "c2_utility_retention": {
            "retain_fwd_ratio": r_fwd,
            "retain_rev_ratio": r_rev,
            "ppl": metrics["ppl"],
            "baseline_ppl": baseline["ppl"],
            "ppl_ratio": ppl_ratio,
            "pass": None not in (r_fwd, r_rev) and min(r_fwd, r_rev) >= THRESHOLDS["c2_retain_ratio_min"] and ppl_ratio <= THRESHOLDS["c2_ppl_ratio_max"],
        },
        "c5_entity_vs_relation": {
            "forget_side_ratio": r_side,
            "threshold": THRESHOLDS["c5_side_ratio_min"],
            "pass": r_side is not None and r_side >= THRESHOLDS["c5_side_ratio_min"],
        },
    }
    # The Q0 reading, only meaningful when the controls hold. Retention is graded: the
    # surviving direction's corrected margin on the forget set as a fraction of the same
    # direction's on the retain set, so "survived" is a number, not just "above chance".
    fm, rm = metrics[other]["forget"].get("margin_corr"), metrics[other]["retain"].get("margin_corr")
    c["q0_reverse_survival"] = {
        "direction": other,
        "forget_cand_accuracy": metrics[other]["forget"][ACC],
        "forget_greedy_accuracy": metrics[other]["forget"].get("accuracy"),
        "forget_margin_corr": fm,
        "retention": fm / rm if (fm is not None and rm) else None,
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
    corr = f" margin_corr={q['forget_margin_corr']:+.3f}" if q["forget_margin_corr"] is not None else ""
    ret = f" retention={q['retention']:.2f}" if q["retention"] is not None else ""
    print(f"  Q0    {q['direction']} on the forget set: cand={q['forget_cand_accuracy']:.3f}{corr}{ret}{mark}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--facts", default="data/facts.json")
    p.add_argument("--out", required=True)
    p.add_argument("--baseline", help="M1's Phase 3 results json. Required for controls; without it this is a baseline run")
    p.add_argument("--target", default="d_fwd", choices=["d_fwd", "d_rev"], help="the direction this condition unlearned (d_rev for u_rev)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()

    base_metrics = json.load(open(a.baseline))["metrics"] if a.baseline else None
    res = run(a.checkpoint, a.facts, a.out, k_alts=None, seed=a.seed, batch_size=a.batch_size, ppl=True)
    print(f"wrote {a.out}")
    print_summary(res["metrics"])

    if base_metrics is None:
        print("no --baseline: this is a baseline run (M1), so no controls are computed")
        return
    c = controls(res["metrics"], base_metrics, a.target)
    print("controls:")
    print_controls(c)
    # Re-open and add the control table, so the probe stays a pure measurement.
    d = json.load(open(a.out))
    d["controls"] = c
    d["config"]["baseline"] = a.baseline
    json.dump(d, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
