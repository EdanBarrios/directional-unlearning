"""Build the synthetic fact set from committed word lists.

Every entity gets a unique first name, last name, title, and town. Full names are
filtered to a token-length band so reverse-direction answers are comparable.
Entities are split into forget / dose / retain by seeded shuffle, with a balance
check on token lengths and role counts. A reserve of spares is written alongside
so a Phase 0 rejection can be replaced by the next draw without regenerating.

The dataset seed is fixed (default 0) and independent of training seeds.

    python -m data.gen_facts --n 150 --out data/facts.json
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from src.seed import rng
from src.util import SMOKE, SMOKE_FACTS, SMOKE_HELDOUT_TEMPLATES, SMOKE_MODEL, SMOKE_TRAIN_TEMPLATES, file_sha, git_commit

SETS = ("forget", "dose", "retain")


def n_tokens(tok, text: str) -> int:
    """Token count of an answer span, which always carries a leading space."""
    assert text.startswith(" "), text
    return len(tok(text)["input_ids"])


def draw_names(r, first, last, tok, n, lo, hi):
    """Unique (first, last) pairs whose token count is in [lo, hi].

    Each last name tries unused first names in shuffled order until one passes,
    so a rejected pair costs one first name, not one last name.
    """
    first, last = first[:], last[:]
    r.shuffle(first)
    r.shuffle(last)
    kept = []
    for l in last:
        for i, f in enumerate(first):
            if lo <= n_tokens(tok, f" {f} {l}") <= hi:
                kept.append((f, l))
                first.pop(i)
                break
        if len(kept) == n:
            return kept
    raise RuntimeError(f"only {len(kept)} names passed the [{lo},{hi}] token filter, need {n}")


def draw_unique_pairs(r, a, b, n, cap_a, cap_b):
    """n distinct (a, b) pairs from the cross product, each word reused at most cap times."""
    pairs = [(x, y) for x in a for y in b]
    r.shuffle(pairs)
    used_a, used_b, kept = Counter(), Counter(), []
    for x, y in pairs:
        if used_a[x] < cap_a and used_b[y] < cap_b:
            kept.append((x, y))
            used_a[x] += 1
            used_b[y] += 1
        if len(kept) == n:
            return kept
    raise RuntimeError(f"only {len(kept)} pairs under caps ({cap_a}, {cap_b}), need {n}")


def build_entities(r, w, tok, n, lo, hi):
    """n entities with name, role, title, description, and town. No set assignment yet."""
    names = draw_names(r, w["first_names"], w["last_names"], tok, n, lo, hi)
    # Titles: a word may appear in at most 2 titles. Towns: suffixes are common by nature
    # (-moor, -wick), so only prefixes are capped.
    titles = draw_unique_pairs(r, w["title_adjectives"], w["title_nouns"], n, cap_a=2, cap_b=2)
    towns = draw_unique_pairs(r, w["town_prefixes"], w["town_suffixes"], n, cap_a=4, cap_b=n)
    roles = w["roles"]
    out = []
    for i in range(n):
        f, l = names[i]
        adj, noun = titles[i]
        role = roles[i % len(roles)]  # even role coverage, order randomized by the name shuffle
        desc = f"the {role} of {adj} {noun}"
        e = {
            "id": i,
            "name": f"{f} {l}",
            "role": role,
            "title": f"{adj} {noun}",
            "desc": desc,
            "town": towns[i][0] + towns[i][1],
        }
        e["name_tokens"] = n_tokens(tok, " " + e["name"])
        e["desc_tokens"] = n_tokens(tok, " " + e["desc"])
        e["town_tokens"] = n_tokens(tok, " " + e["town"])
        out.append(e)
    return out


def balanced(ents, tol_tokens=0.4, tol_role=3):
    """True if the three sets have similar token lengths and no role is lopsided."""
    by_set = {s: [e for e in ents if e["set"] == s] for s in SETS}
    for key in ("name_tokens", "desc_tokens", "town_tokens"):
        means = [statistics.mean(e[key] for e in v) for v in by_set.values()]
        if max(means) - min(means) > tol_tokens:
            return False
    roles = {s: Counter(e["role"] for e in v) for s, v in by_set.items()}
    for role in {e["role"] for e in ents}:
        counts = [roles[s][role] for s in SETS]
        if max(counts) - min(counts) > tol_role:
            return False
    return True


def assign_sets(r, ents, tries=200):
    """Shuffle entities into equal forget / dose / retain sets until the balance check passes."""
    per = len(ents) // 3
    for _ in range(tries):
        r.shuffle(ents)
        for i, e in enumerate(ents):
            e["set"] = SETS[min(i // per, 2)]
        if len(ents) < 30 or balanced(ents):  # balance is meaningless for smoke-sized sets
            return sorted(ents, key=lambda e: e["id"])
    raise RuntimeError("could not find a balanced split; loosen tolerances or change seed")


def smoke_templates(t):
    """Cut every template list to the smoke sizes, keeping the train / held-out structure."""
    out = {"_notes": t["_notes"]}
    for k, v in t.items():
        if k.startswith("_"):
            continue
        tr = v["templates"][: SMOKE_TRAIN_TEMPLATES]
        ho = v["templates"][v["train"] : v["train"] + SMOKE_HELDOUT_TEMPLATES]
        out[k] = {"train": len(tr), "templates": tr + ho}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--spares", type=int, default=30)
    p.add_argument("--reject", type=int, nargs="*", default=[], help="entity ids to replace with spares (Phase 0 rejects)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-tokens", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=4)
    p.add_argument("--model", default="EleutherAI/pythia-160m", help="tokenizer source")
    p.add_argument("--wordlists", default="data/wordlists.json")
    p.add_argument("--templates", default="data/templates.json")
    p.add_argument("--out", default="data/facts.json")
    a = p.parse_args()

    if SMOKE:
        a.n, a.spares, a.model = SMOKE_FACTS, 2, SMOKE_MODEL
        if a.out == "data/facts.json":
            a.out = "data/facts_smoke.json"

    assert a.n % 3 == 0 or SMOKE, "--n must divide into three equal sets"
    w = json.load(open(a.wordlists))
    t = json.load(open(a.templates))
    if SMOKE:
        t = smoke_templates(t)
    tok = AutoTokenizer.from_pretrained(a.model)
    r = rng(a.seed)

    ents = build_entities(r, w, tok, a.n + a.spares, a.min_tokens, a.max_tokens)
    active, spares = ents[: a.n], ents[a.n :]
    active = assign_sets(r, active)
    for s in spares:
        s["set"] = None
    rejected = []
    for rid in a.reject:  # Phase 0 reject: next spare takes the rejected entity's id and set
        i = next(k for k, e in enumerate(active) if e["id"] == rid)
        old, new = active[i], spares.pop(0)
        rejected.append(dict(old, replaced_by=new["name"]))
        new["id"], new["set"] = old["id"], old["set"]
        active[i] = new

    out = {
        "config": {
            **{k: v for k, v in vars(a).items()},
            "git_commit": git_commit(),
            "wordlists_sha": file_sha(a.wordlists),
            "templates_sha": file_sha(a.templates),
            "smoke": SMOKE,
        },
        "templates": t,
        "facts": active,
        "spares": spares,
        "rejected": rejected,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    by_set = Counter(e["set"] for e in active)
    print(f"wrote {a.out}: {len(active)} facts {dict(by_set)}, {len(spares)} spares")
    for key in ("name_tokens", "desc_tokens", "town_tokens"):
        means = {s: round(statistics.mean(e[key] for e in active if e["set"] == s), 2) for s in SETS}
        print(f"  mean {key:12} {means}")
    for e in active[:3]:
        print(f"  {e['set']:6} | {e['name']} | {e['desc']} | {e['town']}")


if __name__ == "__main__":
    main()
