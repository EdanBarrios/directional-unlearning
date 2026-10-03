"""Add extra training templates to an existing facts file (HANDOFF decision 23).

Facts, sets, spares and held-out templates are copied unchanged; each direction gains an
`extra_train` list. Deterministic, so the output is reproducible from committed inputs.

    python -m data.augment --facts data/facts.json --extra data/templates_extra.json --out data/facts_aug.json
"""

import argparse
import copy
import json

from src.util import file_sha

KEYS = ("d_forward", "d_reverse", "s_forward")


def augment(facts: dict, extra: dict) -> dict:
    """A copy of `facts` with extra training templates, refusing any that duplicate an
    existing template (train or held-out)."""
    out = copy.deepcopy(facts)
    for k in KEYS:
        existing = set(out["templates"][k]["templates"])
        dup = [x for x in extra[k] if x in existing]
        assert not dup, f"{k}: extra templates duplicate existing ones: {dup}"
        out["templates"][k]["extra_train"] = list(extra[k])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--facts", default="data/facts.json")
    p.add_argument("--extra", default="data/templates_extra.json")
    p.add_argument("--out", default="data/facts_aug.json")
    a = p.parse_args()
    out = augment(json.load(open(a.facts)), json.load(open(a.extra)))
    out["config"]["augmented_from"] = {"facts": a.facts, "facts_sha": file_sha(a.facts), "extra": a.extra, "extra_sha": file_sha(a.extra)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: " + ", ".join(f"{k} {out['templates'][k]['train']}+{len(out['templates'][k]['extra_train'])} train" for k in KEYS))


if __name__ == "__main__":
    main()
