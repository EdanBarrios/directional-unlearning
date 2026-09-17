"""Log-prob margin and greedy accuracy, both directions, plus held-out perplexity.

Margin for one prompt: mean per-token log-prob of the correct answer under teacher
forcing, minus the mean of that quantity over alternative answers (other entities'
answers for the same direction). Chance is zero. Raw token sums are reported
alongside the per-token means. Prompt tokens are masked with -100.

    python -m src.probe --model EleutherAI/pythia-160m --facts data/facts.json --out results/phase0/base.json
"""

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from src import data as D
from src.model import load
from src.seed import rng, set_seed
from src.util import SMOKE, SMOKE_MODEL, file_sha, git_commit

DIRECTIONS = ("d_fwd", "d_rev", "s_fwd")


@torch.no_grad()
def answer_logprobs(model, tok, prompt, answers, ans_ids, batch_size=64):
    """Per-token mean and raw sum log-prob of each answer following the prompt.

    Every answer starts with a space, so joint tokenization equals prompt ids followed
    by answer ids. That boundary property is asserted on the first answer per prompt.
    """
    dev = model.device
    p_ids = tok(prompt)["input_ids"]
    assert answers[0].startswith(" "), answers[0]
    assert tok(prompt + answers[0])["input_ids"] == p_ids + ans_ids[0], (prompt, answers[0])
    means, sums = [], []
    for i in range(0, len(answers), batch_size):
        chunk = ans_ids[i : i + batch_size]
        L = len(p_ids) + max(len(a) for a in chunk)
        input_ids = torch.full((len(chunk), L), tok.pad_token_id)
        attn = torch.zeros((len(chunk), L), dtype=torch.long)
        labels = torch.full((len(chunk), L), -100)
        for j, a in enumerate(chunk):
            seq = p_ids + a
            input_ids[j, : len(seq)] = torch.tensor(seq)
            attn[j, : len(seq)] = 1
            labels[j, len(p_ids) : len(seq)] = torch.tensor(a)
        logits = model(input_ids=input_ids.to(dev), attention_mask=attn.to(dev)).logits[:, :-1].float()
        tgt = labels[:, 1:].to(dev)
        mask = tgt != -100
        tok_lp = logits.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)
        tok_lp = tok_lp * mask
        s, n = tok_lp.sum(-1), mask.sum(-1)
        sums += s.tolist()
        means += (s / n).tolist()
    return means, sums


@torch.no_grad()
def greedy_hits(model, tok, records, batch_size=32):
    """Generate len(answer) tokens greedily; hit if decoded text starts with the answer."""
    tok.padding_side = "left"
    hits = []
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        enc = tok([r["prompt"] for r in chunk], return_tensors="pt", padding=True).to(model.device)
        n_new = max(len(tok(r["answer"])["input_ids"]) for r in chunk)
        out = model.generate(**enc, max_new_tokens=n_new, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        hits += [g.startswith(r["answer"]) for g, r in zip(gen, chunk)]
    tok.padding_side = "right"
    return hits


def alternatives(all_answers, fact_id, k, seed):
    """Other entities' answers. With k set, a fixed seeded subset per fact, the same every call."""
    others = sorted(i for i in all_answers if i != fact_id)
    if k is not None and k < len(others):
        others = rng(seed * 100003 + fact_id).sample(others, k)
    return [all_answers[i] for i in others]


def probe(model, tok, data, directions=DIRECTIONS, split="heldout", sets=None, k_alts=None, seed=0, batch_size=64, accuracy=True):
    """One row per (direction, fact, template) with margin, raw-sum margin, and greedy hit."""
    rows = []
    for d in directions:
        recs = D.prompts(data, d, split, sets)
        hits = greedy_hits(model, tok, recs, batch_size) if accuracy else [None] * len(recs)
        all_ans = D.answers(data, d)
        ids = {a: tok(a)["input_ids"] for a in all_ans.values()}
        for r, hit in zip(recs, hits):
            cand = [r["answer"]] + alternatives(all_ans, r["fact_id"], k_alts, seed)
            means, sums = answer_logprobs(model, tok, r["prompt"], cand, [ids[c] for c in cand], batch_size)
            rows.append(
                dict(
                    r,
                    correct_mean=means[0],
                    correct_sum=sums[0],
                    margin=means[0] - statistics.mean(means[1:]),
                    margin_sum=sums[0] - statistics.mean(sums[1:]),
                    n_alts=len(cand) - 1,
                    hit=hit,
                )
            )
    return rows


def aggregate(rows):
    """metrics[direction][set] and per_fact[fact_id][direction], averaged over templates."""
    groups = defaultdict(list)
    per_fact_rows = defaultdict(list)
    for r in rows:
        groups[(r["direction"], r["set"])].append(r)
        groups[(r["direction"], "all")].append(r)
        per_fact_rows[(r["fact_id"], r["direction"])].append(r)

    def summ(rs):
        out = {
            "margin": statistics.mean(x["margin"] for x in rs),
            "margin_sum": statistics.mean(x["margin_sum"] for x in rs),
            "correct_mean": statistics.mean(x["correct_mean"] for x in rs),
            "n": len(rs),
        }
        if rs[0]["hit"] is not None:
            out["accuracy"] = statistics.mean(float(x["hit"]) for x in rs)
        if len(rs) > 1:
            out["margin_sd"] = statistics.stdev(x["margin"] for x in rs)
        return out

    metrics = defaultdict(dict)
    for (d, s), rs in groups.items():
        metrics[d][s] = summ(rs)
    per_fact = defaultdict(dict)
    for (fid, d), rs in per_fact_rows.items():
        per_fact[fid]["set"] = rs[0]["set"]
        per_fact[fid][d] = summ(rs)
    return dict(metrics), {str(k): v for k, v in sorted(per_fact.items())}


@torch.no_grad()
def perplexity(model, tok, n_lines=1000, max_len=256, batch_size=16):
    """Perplexity on the first n_lines prose lines of WikiText-2 test (headings skipped)."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    lines = [t for t in ds["text"] if t.strip() and not t.strip().startswith("=")][:n_lines]
    nll, n = 0.0, 0
    for i in range(0, len(lines), batch_size):
        enc = tok(lines[i : i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        loss = model(**enc, labels=labels).loss
        cnt = int((labels[:, 1:] != -100).sum())
        nll += loss.item() * cnt
        n += cnt
    return math.exp(nll / n)


def run(model_name, facts, out, split="heldout", k_alts=None, seed=0, batch_size=64, ppl=True, extra_config=None):
    """Full probe of one model, written to a new results file. Refuses to overwrite."""
    out = Path(out)
    assert not out.exists(), f"{out} exists; new run, new file"
    set_seed(seed)
    t0 = time.time()
    data = D.load_facts(facts)
    model, tok = load(model_name)
    rows = probe(model, tok, data, split=split, k_alts=k_alts, seed=seed, batch_size=batch_size)
    metrics, per_fact = aggregate(rows)
    if ppl:
        metrics["ppl"] = perplexity(model, tok, n_lines=20 if SMOKE else 1000)
    metrics["time_s"] = round(time.time() - t0, 1)
    result = {
        "config": {
            "model": model_name,
            "facts": D.facts_path(facts),
            "facts_sha": file_sha(D.facts_path(facts)),
            "split": split,
            "k_alts": k_alts,
            "smoke": SMOKE,
            "device": str(model.device),
            **(extra_config or {}),
        },
        "git_commit": git_commit(),
        "seed": seed,
        "metrics": metrics,
        "per_fact": per_fact,
        "per_prompt": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out, "w"), indent=1)
    return result


def print_summary(metrics):
    for d in DIRECTIONS:
        if d not in metrics:
            continue
        for s in ("forget", "dose", "retain", "all"):
            m = metrics[d].get(s)
            if m:
                acc = f"acc={m['accuracy']:.2f}" if "accuracy" in m else ""
                print(f"  {d} {s:6} margin={m['margin']:+.3f} sum={m['margin_sum']:+.2f} {acc} n={m['n']}")
    if "ppl" in metrics:
        print(f"  ppl={metrics['ppl']:.2f}")
    print(f"  time={metrics['time_s']}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-160m", help="HF name or checkpoint dir")
    p.add_argument("--facts", default="data/facts.json")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="heldout", choices=["heldout", "train"])
    p.add_argument("--alternatives", default="all", help="'all' or an integer subset size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-ppl", action="store_true")
    a = p.parse_args()
    if SMOKE and a.model == "EleutherAI/pythia-160m":
        a.model = SMOKE_MODEL
    k = None if a.alternatives == "all" else int(a.alternatives)
    res = run(a.model, a.facts, a.out, a.split, k, a.seed, a.batch_size, not a.no_ppl)
    print(f"wrote {a.out}")
    print_summary(res["metrics"])


if __name__ == "__main__":
    main()
