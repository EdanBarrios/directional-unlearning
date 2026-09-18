"""Phase 1: bidirectional finetune of the base model on all facts -> M1.

Trains on the training templates of D forward, D reverse, and S forward for every
entity. Evaluates held-out accuracy and margin each epoch on a fixed fact subset,
then on all facts at the end. Success: held-out accuracy >= 0.9 in both D
directions on both the forget and retain sets.

    python -m src.finetune --config configs/phase1.yaml --seed 0
    python -m src.finetune --config configs/phase1.yaml --seed 0 --lr 3e-5 --epochs 10
"""

import argparse
import json
from pathlib import Path

import yaml

from src import data as D
from src.losses import lm_loss
from src.model import load
from src.probe import aggregate, perplexity, probe
from src.seed import rng, set_seed
from src.train_loop import train
from src.util import SMOKE, SMOKE_MODEL, SMOKE_STEPS, file_sha, git_commit


def quick_eval(model, tok, data, facts_per_set, k_alts, seed, ppl_lines=0):
    """Held-out probe on a fixed subset of facts per set (all facts if facts_per_set is None).

    With ppl_lines set, also measures general-text perplexity, so the accuracy-versus-utility
    trade-off is visible during the run rather than discovered at the end.
    """
    if facts_per_set is None:
        fact_ids = None
    else:
        r = rng(seed + 7)
        fact_ids = []
        for s in D.SETS:
            ids = sorted(f["id"] for f in data["facts"] if f["set"] == s)
            fact_ids += r.sample(ids, min(facts_per_set, len(ids)))
    rows = probe(model, tok, data, k_alts=k_alts, seed=seed, quiet=True, fact_ids=fact_ids)
    metrics, _ = aggregate(rows)
    if ppl_lines:
        metrics["ppl"] = perplexity(model, tok, n_lines=ppl_lines)
    return metrics


def success(ev, threshold, max_ppl=None):
    """Phase 1 criteria: D accuracy >= threshold both directions on forget and retain,
    and general-text perplexity within the allowed bound. Both must hold: a model that
    knows every fact but cannot write English is not a usable starting point."""
    acc_ok = all(ev[d][s]["accuracy"] >= threshold for d in ("d_fwd", "d_rev") for s in ("forget", "retain"))
    ppl_ok = max_ppl is None or ev.get("ppl") is None or ev["ppl"] <= max_ppl
    return acc_ok and ppl_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/phase1.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--facts", default="data/facts.json")
    p.add_argument("--out", help="results json; default results/phase1/lr{lr}_seed{seed}.json")
    p.add_argument("--save", help="checkpoint dir; default checkpoints/phase1_lr{lr}_seed{seed}")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.lr:
        cfg["lr"] = a.lr
    if a.epochs:
        cfg["epochs"] = a.epochs
    if SMOKE:
        cfg.update(model=SMOKE_MODEL, epochs=1, max_steps=SMOKE_STEPS, batch_size=4)
        cfg["eval"]["facts_per_set"] = None
    tag = f"lr{cfg['lr']:g}_seed{a.seed}" + ("_smoke" if SMOKE else "")
    out = Path(a.out or f"results/phase1/{tag}.json")
    assert not out.exists(), f"{out} exists; new run, new file"
    save = None if a.no_save else Path(a.save or f"checkpoints/phase1_{tag}")

    # Provenance is captured at launch, not at write time, so edits made while a run is
    # in flight cannot restamp its results file with inputs it never used.
    commit, facts_file = git_commit(), D.facts_path(a.facts)
    facts_sha = file_sha(facts_file)
    set_seed(a.seed)
    data = D.load_facts(a.facts)
    model, tok = load(cfg["model"])
    records = D.train_records(data)

    # Rehearsal: general text trained alongside the facts, sized in tokens relative to
    # the fact set. Without it Phase 1 destroys the model (HANDOFF decision 18).
    replay = None
    if cfg.get("replay_ratio", 0):
        fact_tokens = D.count_tokens(tok, records)
        replay = D.replay_records(tok, int(fact_tokens * cfg["replay_ratio"]), cfg.get("replay_max_len", 64), a.seed)
        print(f"replay: {len(replay)} sequences, {D.count_tokens(tok, replay)} tokens vs {fact_tokens} fact tokens (ratio {cfg['replay_ratio']})")
    print(f"{len(records)} training sequences, lr={cfg['lr']:g}, epochs={cfg['epochs']}, batch={cfg['batch_size']}")

    ev_cfg = cfg["eval"]
    max_ppl = cfg.get("max_ppl")
    loss_fn = lambda m, b, aux=None: lm_loss(m, b, aux, replay_weight=cfg.get("replay_weight", 1.0))
    history = train(
        model,
        tok,
        records,
        loss_fn,
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        seed=a.seed,
        aux_records=replay,
        aux_batch_size=cfg.get("replay_batch_size"),
        epochs=cfg["epochs"],
        max_steps=cfg.get("max_steps"),
        eval_fn=lambda m, step, epoch: quick_eval(m, tok, data, ev_cfg["facts_per_set"], ev_cfg["k_alts"], a.seed, ev_cfg.get("ppl_lines", 0)),
        eval_every_epoch=True,
        stop_fn=(lambda ev: success(ev, cfg["success_accuracy"], max_ppl)) if cfg.get("stop_on_success") else None,
        grad_clip=cfg.get("grad_clip", 1.0),
    )

    print("final eval on all facts")
    final = quick_eval(model, tok, data, None, ev_cfg["k_alts"], a.seed, ppl_lines=20 if SMOKE else 1000)
    ok = success(final, cfg["success_accuracy"], max_ppl)
    print(f"success={ok}  ppl={final.get('ppl', float('nan')):.2f} (max {max_ppl})  " + " ".join(f"{d}/{s}={final[d][s]['accuracy']:.2f}" for d in ("d_fwd", "d_rev", "s_fwd") for s in D.SETS))

    if save:
        save.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save)
        tok.save_pretrained(save)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "config": {**cfg, "facts": facts_file, "facts_sha": facts_sha, "smoke": SMOKE, "checkpoint": str(save) if save else None},
            "git_commit": commit,
            "seed": a.seed,
            "metrics": {"final": final, "success": ok, "steps": history[-1]["step"], "epochs": history[-1]["epoch"]},
            "history": history,
        },
        open(out, "w"),
        indent=1,
    )
    print(f"wrote {out}" + (f", checkpoint {save}" if save else ""))


if __name__ == "__main__":
    main()
