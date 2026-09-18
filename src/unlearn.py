"""Phase 2: unlearn from M1 under one method and one condition.

Conditions differ only in which fact-directions are pushed against. The retain
regularizer is the retain set in every condition, so the conditions are comparable.

    condition    | ascended                                   | groups
    u_fwd        | forget D forward                           | 50
    u_both       | forget D forward + D reverse               | 100
    u_fwd_dose   | forget D forward + dose-set D forward      | 100

u_fwd_dose matches u_both's dose, so a difference between them is about direction
rather than about how much unlearning happened (control C6).

Stop rule (C1): probe the forget set's held-out forward prompts every `eval_every`
steps and stop when raw margin <= 0 and accuracy <= 0.05. If the step cap is reached
first, the run is marked failed and must be excluded from analysis, not tuned until
it passes.

    python -m src.unlearn --config configs/npo_gd.yaml --condition u_fwd --seed 0
"""

import argparse
import copy
import json
from pathlib import Path

import yaml

from src import data as D
from src import losses
from src.model import load
from src.probe import aggregate, probe
from src.seed import set_seed
from src.train_loop import train
from src.util import SMOKE, SMOKE_STEPS, file_sha, git_commit

# condition -> (direction, set) pairs to push against
CONDITIONS = {
    "u_fwd": [("d_fwd", "forget")],
    "u_both": [("d_fwd", "forget"), ("d_rev", "forget")],
    "u_fwd_dose": [("d_fwd", "forget"), ("d_fwd", "dose")],
}


def ascend_records(data, condition):
    """Training-template records for the fact-directions this condition unlearns."""
    out = []
    for direction, s in CONDITIONS[condition]:
        out += D.prompts(data, direction, "train", sets=(s,))
    return out


def retain_records(data):
    """Retain set, every trained direction. Identical across conditions."""
    return D.train_records(data, sets=("retain",))


def forget_eval(model, tok, data, k_alts, seed):
    """Stop-rule probe: forget set, held-out forward prompts only."""
    rows = probe(model, tok, data, directions=("d_fwd",), sets=("forget",), k_alts=k_alts, seed=seed, quiet=True)
    metrics, _ = aggregate(rows)
    return metrics["d_fwd"]["forget"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default="checkpoints/M1", help="M1, the Phase 1 model")
    p.add_argument("--facts", default="data/facts.json")
    p.add_argument("--lr", type=float)
    p.add_argument("--out")
    p.add_argument("--save")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.lr:
        cfg["lr"] = a.lr
    if SMOKE:
        cfg.update(max_steps=SMOKE_STEPS, batch_size=4, eval_every=SMOKE_STEPS)
    method = cfg["method"]
    tag = f"{method}_{a.condition}_seed{a.seed}" + ("_smoke" if SMOKE else "")
    out = Path(a.out or f"results/phase2/{tag}.json")
    assert not out.exists(), f"{out} exists; new run, new file"
    save = None if a.no_save else Path(a.save or f"checkpoints/{tag}")

    commit, facts_file = git_commit(), D.facts_path(a.facts)
    facts_sha = file_sha(facts_file)
    set_seed(a.seed)
    data = D.load_facts(a.facts)
    model, tok = load(a.checkpoint)

    ref = None
    if method == "npo_gd":
        ref, _ = load(a.checkpoint)  # frozen copy of M1; NPO measures drift against it
        ref.requires_grad_(False)
        ref.eval()

    loss_fn = losses.build(method, ref_model=ref, beta=cfg.get("beta", 0.1), retain_weight=cfg.get("retain_weight", 1.0))
    ascend, retain = ascend_records(data, a.condition), retain_records(data)
    groups = len({(r["fact_id"], r["direction"]) for r in ascend})
    print(f"{method} / {a.condition} / seed {a.seed}: {len(ascend)} ascend records ({groups} fact-direction groups), {len(retain)} retain records")

    baseline = forget_eval(model, tok, data, cfg["eval"]["k_alts"], a.seed)
    print(f"  M1 baseline forget d_fwd: acc={baseline['accuracy']:.2f} margin={baseline['margin']:+.3f}")

    def stop(ev):
        return ev["margin"] <= cfg["stop"]["margin"] and ev["accuracy"] <= cfg["stop"]["accuracy"]

    history = train(
        model,
        tok,
        ascend,
        loss_fn,
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        seed=a.seed,
        aux_records=retain,
        aux_batch_size=cfg.get("retain_batch_size"),
        max_steps=cfg["max_steps"],
        eval_fn=lambda m, step, epoch: forget_eval(m, tok, data, cfg["eval"]["k_alts"], a.seed),
        eval_every_steps=cfg["eval_every"],
        stop_fn=stop,
        grad_clip=cfg.get("grad_clip", 1.0),
    )

    final = history[-1]["eval"]
    unlearned = stop(final)
    steps = history[-1]["step"]
    print(f"  {'UNLEARNED' if unlearned else 'FAILED C1 (hit step cap)'} at step {steps}: acc={final['accuracy']:.2f} margin={final['margin']:+.3f}")

    if save and unlearned:
        save.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save)
        tok.save_pretrained(save)
    elif save:
        print("  not saving a checkpoint for a failed run")

    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "config": {
                **cfg,
                "condition": a.condition,
                "start_checkpoint": a.checkpoint,
                "facts": facts_file,
                "facts_sha": facts_sha,
                "smoke": SMOKE,
                "checkpoint": str(save) if (save and unlearned) else None,
            },
            "git_commit": commit,
            "seed": a.seed,
            "metrics": {
                "unlearned": unlearned,  # False means failed C1; exclude from analysis
                "steps": steps,
                "groups_ascended": groups,
                "baseline": baseline,
                "final": final,
            },
            "history": history,
        },
        open(out, "w"),
        indent=1,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
