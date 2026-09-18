"""The one training loop. Takes a loss function and optional eval / stop hooks.

Runs for `epochs` full passes or `max_steps` optimizer steps, whichever comes first.
Shuffle order is the only thing the seed controls; the dataset is fixed.
"""

import time

import torch

from src.data import SETS, encode
from src.seed import rng


def train(
    model,
    tok,
    records,
    loss_fn,
    *,
    lr,
    batch_size,
    seed,
    epochs=None,
    max_steps=None,
    eval_fn=None,
    eval_every_steps=None,
    eval_every_epoch=False,
    stop_fn=None,
    grad_clip=1.0,
    log=print,
):
    """Train in place. Returns a history of {step, epoch, loss, eval} dicts.

    eval_fn(model, step, epoch) -> dict is called at eval points; stop_fn(eval) -> bool
    ends training early. Loss in each history entry is the running mean since the last one.
    """
    assert epochs or max_steps, "need epochs or max_steps"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    r = rng(seed)
    history, step, epoch = [], 0, 0
    losses, t0 = [], time.time()

    def do_eval():
        model.eval()
        ev = eval_fn(model, step, epoch) if eval_fn else None
        model.train()
        entry = {"step": step, "epoch": epoch, "loss": sum(losses) / max(len(losses), 1), "eval": ev, "time_s": round(time.time() - t0, 1)}
        history.append(entry)
        losses.clear()
        log(f"  step {step:5d} epoch {epoch:3d} loss {entry['loss']:.4f} " + (fmt_eval(ev) if ev else ""))
        return ev

    model.train()
    done = False
    while not done:
        order = list(range(len(records)))
        r.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = encode(tok, [records[j] for j in order[i : i + batch_size]], model.device)
            loss = loss_fn(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            step += 1
            losses.append(loss.item())
            if eval_every_steps and step % eval_every_steps == 0:
                ev = do_eval()
                if stop_fn and ev and stop_fn(ev):
                    done = True
                    break
            if max_steps and step >= max_steps:
                done = True
                break
        epoch += 1
        if not done and eval_every_epoch:
            ev = do_eval()
            if stop_fn and ev and stop_fn(ev):
                done = True
        if epochs and epoch >= epochs:
            done = True
    if not history or history[-1]["step"] != step:
        do_eval()
    model.eval()
    return history


def fmt_eval(ev):
    """One-line summary of an eval dict produced by finetune.quick_eval.

    Only the per-set entries are shown; metrics[direction] also carries 'all' and
    'per_template', which are not set summaries.
    """
    parts = []
    for d in ("d_fwd", "d_rev", "s_fwd"):
        if d not in ev:
            continue
        cells = [f"{s[:3]}={ev[d][s]['accuracy']:.2f}/{ev[d][s].get('margin_corr', ev[d][s]['margin']):+.2f}" for s in SETS if s in ev[d]]
        parts.append(d + " " + " ".join(cells))
    return " | ".join(parts)
