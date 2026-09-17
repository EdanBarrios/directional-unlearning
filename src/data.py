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


def prompts(data: dict, direction: str, split: str, sets=None) -> list[dict]:
    """One record per (fact, template): prompt, answer, fact id, set."""
    ans_key = DIRECTIONS[direction][1]
    out = []
    for f in data["facts"]:
        if sets and f["set"] not in sets:
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


def answers(data: dict, direction: str) -> dict[int, str]:
    """Every fact's answer for one direction, keyed by fact id. Margin alternatives come from here."""
    ans_key = DIRECTIONS[direction][1]
    return {f["id"]: " " + f[ans_key] for f in data["facts"]}
