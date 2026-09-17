# Directional structure of LLM unlearning

Does unlearning a fact in one direction ("A is B") remove it in the other ("B is A"), and are the two directions stored independently in the weights?

Prior work has shown that unlearning or editing the forward direction of a fact leaves the reverse direction largely intact:

- BAKE: Ma et al., "Untying the Reversal Curse via Bidirectional Language Model Editing" ([arXiv 2310.10322](https://arxiv.org/abs/2310.10322))
- UGBench: Wang et al., "Erasing Without Remembering: Implicit Knowledge Forgetting in Large Language Models" ([arXiv 2502.19982](https://arxiv.org/abs/2502.19982))
- GONE: Dahal et al., "Structural Knowledge Unlearning via Neighborhood-Expanded Distribution Shaping" ([arXiv 2603.12275](https://arxiv.org/abs/2603.12275))
- "The Illusion of Latent Generalization: Bi-directionality and the Reversal Curse" ([arXiv 2604.04943](https://arxiv.org/abs/2604.04943))

This project first replicates that result in a controlled setting (synthetic facts, both directions trained explicitly, Pythia-160m), then asks the question prior work could not: after unlearning the forward direction, does an intact reverse direction make the forward direction faster to relearn? A secondary question is whether 4-bit quantization restores the two directions symmetrically.

Design, rationale, and locked decisions are in [docs/HANDOFF.md](docs/HANDOFF.md).

## Status

Design locked 2026-09-16. Code in progress.

## Setup

```
uv sync
SMOKE=1 uv run python -m src.probe --model EleutherAI/pythia-70m --facts data/facts.json --out results/phase0/smoke.json
```

Full runs happen on Kaggle. See `kaggle_launch.ipynb`.

## License

MIT
