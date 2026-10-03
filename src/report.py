"""Tables for the writeup, regenerated from results files. Nothing here is computed by hand.

Takes any results file that stores per-prompt rows (probe, evaluate, the Phase 1 full
probe) and prints the set table and the per-template table: candidate accuracy, its
prior-corrected version, greedy accuracy and corrected margin. Older files that predate
candidate accuracy (decision 21) are re-aggregated from their stored candidate scores.

    python -m src.report results/phase1/lr1e-05_rw1_kl1_seed0_M1_probe.json
    python -m src.report results/phase1/*.json --markdown
"""

import argparse
import json

from src.probe import DIRECTIONS, aggregate

SETS = ("forget", "dose", "retain")


def metrics_from(path):
    """Set and per-template metrics, recomputed from per-prompt rows."""
    d = json.load(open(path))
    rows = d["per_prompt"]
    for r in rows:
        r.setdefault("cand_hit", r["cand_mean"][0] > max(r["cand_mean"][1:]))
        r.setdefault("gen", None)
    m, _ = aggregate(rows)
    m["ppl_by_corpus"] = d["metrics"].get("ppl_by_corpus")
    m["n_candidates"] = len(rows[0]["cand_mean"])
    return d, m


def fmt(v, spec=".2f"):
    return "-" if v is None else format(v, spec)


def set_table(m):
    out = ["| direction | set | cand | cand corr | greedy | margin corr |", "|---|---|---|---|---|---|"]
    for dname in DIRECTIONS:
        for s in SETS:
            x = m[dname][s]
            out.append(f"| {dname} | {s} | {fmt(x['cand_accuracy'])} | {fmt(x.get('cand_accuracy_corr'))} | {fmt(x.get('accuracy'))} | {fmt(x.get('margin_corr'), '+.2f')} |")
    return out


def template_table(m):
    out = ["| direction | template | cand | greedy | margin corr |", "|---|---|---|---|---|"]
    for dname in DIRECTIONS:
        for t, x in sorted(m[dname]["per_template"].items(), key=lambda kv: int(kv[0])):
            out.append(f"| {dname} | t{t} | {fmt(x['cand_accuracy'])} | {fmt(x.get('accuracy'))} | {fmt(x.get('margin_corr'), '+.2f')} |")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--markdown", action="store_true", help="markdown tables (default is the same, printed plainly)")
    a = p.parse_args()
    for path in a.files:
        d, m = metrics_from(path)
        print(f"\n## {path}\ncommit {d.get('git_commit')}, {m['n_candidates']} candidates, ppl {m['ppl_by_corpus']}\n")
        print("\n".join(set_table(m)))
        print()
        print("\n".join(template_table(m)))


if __name__ == "__main__":
    main()
