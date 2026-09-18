"""Phase 5: probe a model under NF4 4-bit quantization. Answers Q2.

Post-training quantization is known to restore knowledge that unlearning removed
(Zhang et al., arXiv 2410.16454), because rounding to 4 bits erases small weight
changes. NPO keeps changes small, so that is the method where the effect exists.
The open question here is whether the two directions come back by the same amount.

Quantization degrades any model somewhat, so the comparison needs M1's own
degradation as the reference. Restoration is read per direction as

    (unlearned quantized) - (unlearned fp32)

and judged against

    (M1 quantized) - (M1 fp32)

which is what quantization does to a model that was never unlearned. CUDA only:
bitsandbytes has no MPS backend, so this phase runs on Kaggle.

    python -m src.quantize --checkpoint checkpoints/npo_gd_u_both_seed0 \
        --fp32-results results/phase3/npo_gd_u_both_seed0.json --out results/phase5/npo_gd_u_both_seed0_nf4.json
"""

import argparse
import json
from pathlib import Path

import torch

from src import data as D
from src.probe import DIRECTIONS, aggregate, perplexity, print_summary, probe
from src.seed import set_seed
from src.util import file_sha, git_commit


def load_nf4(name_or_path):
    """Load in NF4 4-bit. Weights are quantized on load; nothing is trained afterwards."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    assert torch.cuda.is_available(), "bitsandbytes NF4 needs CUDA; run this phase on Kaggle"
    tok = AutoTokenizer.from_pretrained(name_or_path)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
    )
    model.eval()
    return model, tok


def deltas(quantized, fp32):
    """Per-direction change on the forget set caused by quantizing. Positive means the
    quantized model scores the fact higher, which is restoration."""
    out = {}
    for d in DIRECTIONS:
        q, f = quantized[d]["forget"], fp32[d]["forget"]
        out[d] = {
            "fp32_accuracy": f["accuracy"],
            "nf4_accuracy": q["accuracy"],
            "d_accuracy": q["accuracy"] - f["accuracy"],
            "fp32_margin_corr": f.get("margin_corr"),
            "nf4_margin_corr": q.get("margin_corr"),
            "d_margin_corr": (q.get("margin_corr") - f.get("margin_corr")) if q.get("margin_corr") is not None and f.get("margin_corr") is not None else None,
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fp32-results", required=True, help="the Phase 3 results file for this same checkpoint")
    p.add_argument("--reference", help="M1's quantization deltas, from this script run on M1; the baseline for Q2")
    p.add_argument("--facts", default="data/facts.json")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()

    out = Path(a.out)
    assert not out.exists(), f"{out} exists; new run, new file"
    commit, facts_file = git_commit(), D.facts_path(a.facts)
    facts_sha = file_sha(facts_file)
    set_seed(a.seed)

    fp32 = json.load(open(a.fp32_results))
    assert fp32["config"]["facts_sha"] == facts_sha, "fp32 results used a different dataset"
    data = D.load_facts(a.facts)
    model, tok = load_nf4(a.checkpoint)
    rows = probe(model, tok, data, k_alts=None, seed=a.seed, batch_size=a.batch_size)
    metrics, per_fact = aggregate(rows)
    metrics["ppl"] = perplexity(model, tok)
    print_summary(metrics)

    dl = deltas(metrics, fp32["metrics"])
    print("\nquantization delta on the forget set (positive = restored by quantizing):")
    for d, v in dl.items():
        print(f"  {d}  acc {v['fp32_accuracy']:.3f} -> {v['nf4_accuracy']:.3f} ({v['d_accuracy']:+.3f})   margin_corr {v['d_margin_corr']:+.3f}" if v["d_margin_corr"] is not None else f"  {d}  acc {v['d_accuracy']:+.3f}")
    asym = None
    if dl["d_fwd"]["d_margin_corr"] is not None:
        asym = dl["d_fwd"]["d_margin_corr"] - dl["d_rev"]["d_margin_corr"]
        print(f"  forward minus reverse restoration: {asym:+.3f} nats")
        if a.reference:
            ref = json.load(open(a.reference))["quantization"]["asymmetry"]
            print(f"  M1's own asymmetry under quantization: {ref:+.3f} nats (the Q2 baseline)")

    json.dump(
        {
            "config": {
                "checkpoint": a.checkpoint,
                "quantization": "nf4-double",
                "fp32_results": a.fp32_results,
                "reference": a.reference,
                "facts": facts_file,
                "facts_sha": facts_sha,
                "device": str(model.device),
            },
            "git_commit": commit,
            "seed": a.seed,
            "metrics": metrics,
            "quantization": {"deltas": dl, "asymmetry": asym},
            "per_fact": per_fact,
            "per_prompt": rows,
        },
        open(out, "w"),
        indent=1,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
