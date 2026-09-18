"""Facts and prompts. Templates are filled here. Answers always carry a leading space."""

import json

from src.util import SMOKE

# direction -> (template key, answer field)
DIRECTIONS = {
    "d_fwd": ("d_forward", "desc"),
    "d_rev": ("d_reverse", "name"),
    "s_fwd": ("s_forward", "town"),
}
SETS = ("forget", "dose", "retain")


def facts_path(path: str) -> str:
    """Under SMOKE the default facts file is swapped for the smoke one."""
    if SMOKE and path == "data/facts.json":
        return "data/facts_smoke.json"
    return path


def load_facts(path: str) -> dict:
    return json.load(open(facts_path(path)))


def fill(template: str, fact: dict) -> str:
    """{name}, {desc}, and {Desc} (sentence-initial capital)."""
    desc = fact["desc"]
    return template.format(name=fact["name"], desc=desc, Desc=desc[0].upper() + desc[1:])


def templates(data: dict, direction: str, split: str) -> list[tuple[int, str]]:
    """(index, template) pairs for one direction and split ('train' or 'heldout')."""
    t = data["templates"][DIRECTIONS[direction][0]]
    n = t["train"]
    idx = range(n) if split == "train" else range(n, len(t["templates"]))
    return [(i, t["templates"][i]) for i in idx]


def prompts(data: dict, direction: str, split: str, sets=None, fact_ids=None) -> list[dict]:
    """One record per (fact, template): prompt, answer, fact id, set."""
    ans_key = DIRECTIONS[direction][1]
    out = []
    for f in data["facts"]:
        if sets and f["set"] not in sets:
            continue
        if fact_ids is not None and f["id"] not in fact_ids:
            continue
        for ti, tpl in templates(data, direction, split):
            p = fill(tpl, f)
            assert not p.endswith(" "), p  # answer supplies the space; prompt must not
            out.append(
                {
                    "fact_id": f["id"],
                    "set": f["set"],
                    "direction": direction,
                    "template": ti,
                    "prompt": p,
                    "answer": " " + f[ans_key],
                }
            )
    return out


def train_records(data: dict, directions=("d_fwd", "d_rev", "s_fwd"), sets=None) -> list[dict]:
    """All training-template records across the given directions and sets."""
    out = []
    for d in directions:
        out += prompts(data, d, "train", sets)
    return out


def replay_records(tok, n_tokens: int, max_len: int = 64, seed: int = 0) -> list[dict]:
    """General-text sequences for rehearsal during Phase 1.

    Drawn from WikiText-2 *train*. The C2 perplexity metric uses the *test* split, so
    the model is never trained on the text it is later measured on.
    """
    from datasets import load_dataset

    from src.seed import rng

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    lines = [t.strip() for t in ds["text"] if t.strip() and not t.strip().startswith("=")]
    rng(seed + 991).shuffle(lines)
    out, total = [], 0
    for line in lines:
        ids = tok(line, truncation=True, max_length=max_len)["input_ids"]
        if len(ids) < 8:
            continue
        out.append({"kind": "replay", "ids": ids})
        total += len(ids)
        if total >= n_tokens:
            break
    return out


def count_tokens(tok, records: list[dict]) -> int:
    """Total answer-plus-prompt tokens across records, for sizing the replay set."""
    return sum(len(r["ids"]) if r.get("kind") == "replay" else len(tok(r["prompt"])["input_ids"]) + len(tok(r["answer"])["input_ids"]) for r in records)


def encode(tok, records: list[dict], device) -> dict:
    """Right-padded batch. labels are -100 on prompt and pad, so loss covers answer tokens
    only. Replay records carry raw ids and take loss on every token."""
    import torch

    seqs, labs = [], []
    for r in records:
        if r.get("kind") == "replay":
            seqs.append(r["ids"])
            labs.append(r["ids"])
            continue
        p = tok(r["prompt"])["input_ids"]
        a = tok(r["answer"])["input_ids"]
        seqs.append(p + a)
        labs.append([-100] * len(p) + a)
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    labels = torch.full((len(seqs), L), -100)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1
        labels[i, : len(l)] = torch.tensor(l)
    return {"input_ids": ids.to(device), "attention_mask": attn.to(device), "labels": labels.to(device)}


def answers(data: dict, direction: str) -> dict[int, str]:
    """Every fact's answer for one direction, keyed by fact id. Margin alternatives come from here."""
    ans_key = DIRECTIONS[direction][1]
    return {f["id"]: " " + f[ans_key] for f in data["facts"]}
