"""Phase 4: relearn the forward direction from an unlearned model and time the recovery.

This is the headline measurement (Q1). The relearning attack from the literature is
used here as an instrument: how fast a direction comes back, from starting points that
differ only in what else was unlearned, is evidence about how the fact was stored.

    U-fwd       reverse intact
    U-both      reverse also unlearned
    U-fwd+dose  reverse intact, equal total unlearning dose (control C6)

Faster recovery in U-fwd than U-both, with U-fwd+dose looking like U-fwd, is evidence
that the directions share structure. All three the same is evidence they are independent
entries. U-fwd+dose looking like U-both means the effect is general damage, not direction.

Transfer (decision 22): every eval also probes the reverse direction. If relearning
forward in U-both brings reverse back with it, the two directions share structure, a
more direct test than relearning speed alone. Curves are stored per fact, so conditions
can be compared fact by fact (all start from the same M1).

Every parameter here is pre-registered (HANDOFF 7.4 / decision 17) because the outcome
is a comparison of curves, and a threshold chosen after seeing them would choose itself.
The target is target_fraction of M1's own forget-set forward candidate accuracy, read
from M1's Phase 3 results file (decision 21).

    python -m src.relearn --checkpoint checkpoints/npo_gd_u_fwd_seed0 --m1-results results/phase3/M1.json --seed 0
"""

import argparse
import json
from pathlib import Path

import yaml

from src import data as D
from src.losses import lm_loss
from src.model import load
from src.probe import aggregate, probe
from src.seed import rng, set_seed
from src.train_loop import train
from src.util import SMOKE, SMOKE_STEPS, file_sha, git_commit


def relearn_records(data, n_templates, seed):
    """Forget set, forward direction, a fixed seeded subset of the training templates.

    A subset rather than all 25, because the question is how quickly the fact returns
    from a small nudge; handing back the full training set would measure retraining.
    """
    recs = D.prompts(data, "d_fwd", "train", sets=("forget",))
    keep = set(rng(seed + 31).sample(sorted({r["template"] for r in recs}), n_templates))
    return [r for r in recs if r["template"] in keep]


def forget_eval(model, tok, data, k_alts, seed):
    """Recovery probe: forget set, held-out prompts, both D directions, per fact."""
    rows = probe(model, tok, data, directions=("d_fwd", "d_rev"), sets=("forget",), k_alts=k_alts, seed=seed, quiet=True)
    metrics, per_fact = aggregate(rows)
    return {**metrics, "per_fact": per_fact}


def recovery(history, target, direction="d_fwd", key="cand_accuracy"):
    """Steps to reach the target accuracy, and normalized area under the recovery curve.

    AUC is trapezoidal over (step, accuracy) divided by the step span, so it is a mean
    accuracy over the run in [0, 1] and is comparable across runs of the same cap. It
    separates "recovered late but fully" from "recovered early", which steps-to-threshold
    alone cannot.
    """
    val = lambda h: h["eval"][direction]["forget"][key]
    pts = [(0, val(history[0]))] if history[0]["step"] != 0 else []
    pts += [(h["step"], val(h)) for h in history]
    hit = next((s for s, a in pts if a >= target), None)
    area = sum((pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1]) / 2 for i in range(1, len(pts)))
    span = pts[-1][0] - pts[0][0]
    return {"steps_to_threshold": hit, "reached_threshold": hit is not None, "auc": area / span if span else None, "curve": pts}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/relearn.yaml")
    p.add_argument("--checkpoint", required=True, help="an unlearned model, or M1 for the sanity check")
    p.add_argument("--m1-results", required=True, help="M1's Phase 3 results json; the recovery target is derived from it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--facts", default="data/facts.json")
    p.add_argument("--out")
    p.add_argument("--save")
    p.add_argument("--no-save", action="store_true", default=True)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if SMOKE:
        cfg.update(max_steps=SMOKE_STEPS, batch_size=4, eval_every=1, n_templates=1)
    m1 = json.load(open(a.m1_results))["metrics"]
    target = cfg["target_fraction"] * m1["d_fwd"]["forget"]["cand_accuracy"]
    tag = Path(a.checkpoint).name + f"_seed{a.seed}" + ("_smoke" if SMOKE else "")
    out = Path(a.out or f"results/phase4/{tag}.json")
    assert not out.exists(), f"{out} exists; new run, new file"

    commit, facts_file = git_commit(), D.facts_path(a.facts)
    facts_sha = file_sha(facts_file)
    set_seed(a.seed)
    data = D.load_facts(a.facts)
    model, tok = load(a.checkpoint)
    recs = relearn_records(data, cfg["n_templates"], a.seed)
    print(f"relearn {a.checkpoint}: {len(recs)} records ({cfg['n_templates']} templates x {len(recs)//cfg['n_templates']} facts), lr={cfg['lr']:g}, target acc={target:.2f}")

    start = forget_eval(model, tok, data, cfg["eval"]["k_alts"], a.seed)
    for d in ("d_fwd", "d_rev"):
        m = start[d]["forget"]
        print(f"  start {d}: cand={m['cand_accuracy']:.2f} greedy={m['accuracy']:.2f} margin={m['margin']:+.3f}")

    history = train(
        model,
        tok,
        recs,
        lm_loss,
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        seed=a.seed,
        max_steps=cfg["max_steps"],
        eval_fn=lambda m, step, epoch: forget_eval(m, tok, data, cfg["eval"]["k_alts"], a.seed),
        eval_every_steps=cfg["eval_every"],
        grad_clip=cfg.get("grad_clip", 1.0),
    )
    history = [{"step": 0, "epoch": 0, "loss": None, "eval": start, "time_s": 0.0}] + history
    rec = recovery(history, target)
    # Transfer: the reverse direction's curve while only forward is trained. No threshold
    # is pre-registered for it; it is reported as a curve and AUC.
    transfer = recovery(history, target, direction="d_rev")
    print(f"  d_fwd steps_to_threshold={rec['steps_to_threshold']} auc={rec['auc']:.3f} final cand={history[-1]['eval']['d_fwd']['forget']['cand_accuracy']:.2f}")
    print(f"  d_rev (transfer) auc={transfer['auc']:.3f} start cand={start['d_rev']['forget']['cand_accuracy']:.2f} final cand={history[-1]['eval']['d_rev']['forget']['cand_accuracy']:.2f}")

    if a.save and not a.no_save:
        Path(a.save).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(a.save)
        tok.save_pretrained(a.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "config": {**cfg, "start_checkpoint": a.checkpoint, "m1_results": a.m1_results, "target_accuracy": target, "facts": facts_file, "facts_sha": facts_sha, "smoke": SMOKE},
            "git_commit": commit,
            "seed": a.seed,
            "metrics": {**rec, "transfer_d_rev": transfer, "start": start, "final": history[-1]["eval"]},
            "history": history,
        },
        open(out, "w"),
        indent=1,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
