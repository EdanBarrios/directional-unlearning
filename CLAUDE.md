# CLAUDE.md

Research repo: does LLM unlearning remove a fact in both directions, and are the two directions stored independently? Design, rationale, and locked decisions live in `docs/HANDOFF.md`. Read it before touching anything experimental.

## Layout

```
data/wordlists.json    raw material, LLM-written once, committed
data/templates.json    30+30+10 sentence frames, split by position
data/gen_facts.py      word lists → data/facts.json (150 facts + 30 spares). SMOKE writes facts_smoke.json
src/seed.py            all randomness goes through here
src/util.py            SMOKE flag and sizes, git hash, file hash
src/model.py           load model + tokenizer (pad token set here, nowhere else)
src/data.py            facts.json → prompt/answer records per direction and split
src/probe.py           log-prob margin + accuracy, both directions, perplexity
src/train_loop.py      the only training loop; takes a loss fn
src/losses.py          lm, gradient ascent, NPO
src/finetune.py        Phase 1     src/unlearn.py   Phase 2
src/relearn.py         Phase 4     src/quantize.py  Phase 5
src/evaluate.py        Phase 3 → results/*.json
configs/*.yaml         one per phase and method
kaggle_launch.ipynb    clone, pip install, run. Nothing else lives in notebooks.
```

## Commands

```
python -m data.gen_facts --n 150 --out data/facts.json
python -m src.probe --model EleutherAI/pythia-160m --facts data/facts.json --out results/phase0/base.json
python -m src.finetune --config configs/phase1.yaml --seed 0
python -m src.unlearn  --config configs/npo_gd.yaml --condition u_fwd --seed 0
python -m src.evaluate --checkpoint checkpoints/npo_gd_u_fwd_seed0 --baseline results/phase3/M1.json --out results/phase3/npo_gd_u_fwd_seed0.json
python -m src.relearn  --config configs/relearn.yaml --checkpoint checkpoints/npo_gd_u_fwd_seed0 --seed 0
python -m src.quantize --checkpoint checkpoints/npo_gd_u_both_seed0 --fp32-results results/phase3/npo_gd_u_both_seed0.json --out results/phase5/...
```

Phase 5 needs CUDA: `uv sync --extra cuda` on a CUDA box, never locally.

`SMOKE=1` in front of any command switches to Pythia-70m, 5 facts, 2 templates, 3 steps. Every script must honor it. Run smoke before any GPU run.

## Conventions

- Pythia tokenizer has no pad token. `pad_token = eos_token`, set in `src/model.py` only.
- Prompts are completions ("<Name> is"), never questions. Pythia is a base model.
- Answers are tokenized with a leading space. `probe.py` asserts this.
- Answer log-prob: teacher forcing, prompt tokens masked with `-100`, mean over answer tokens. Raw sum reported alongside.
- fp32, full finetune, no LoRA, no HF `Trainer`, no mixed precision. T4 has no bf16.
- Test prompts use held-out templates only. Training templates never appear in evaluation.
- Results: one JSON per run at `results/<phase>/<method>_<condition>_seed<N>.json` with `config`, `git_commit`, `seed`, `metrics`. New run, new file. Never overwrite.
- All randomness goes through `src/seed.py`. Dataset is fixed; seed controls shuffle order.

## Rules

- Do not change a condition, seed, threshold, or metric definition silently. State it in the response and update `docs/HANDOFF.md` section 6 or 7.
- Do not commit checkpoints, HF cache, `.env`, tokens, or notebook outputs. `.gitignore` covers these; keep it that way.
- Local machine is an Apple M4, 16 GB, MPS available. Smoke tests and Phase 0/1 iteration can run locally. The full seed grid and anything needing CUDA (Phase 5, bitsandbytes) run on Kaggle.
- Run long jobs as `caffeinate -i uv run python -u -m ...`. Without `-u` Python block-buffers stdout when not attached to a terminal, so a running job looks hung until it exits.
- Prefix every local training run with `caffeinate -i`. A sleeping laptop suspends the job, and wall-clock `time_s` in the results file counts the sleep as compute. Never run a second GPU job while one is training; both slow down and the timings become meaningless.
- Reference timings, M4 fp32 Pythia-160m: 618 ms per step at batch 32, 3.2 min per Phase 1 epoch, 22 s per in-loop eval. Grad clip is 42% of the step because MPS has no `foreach` path; on CUDA it is not.
- Python env is managed by `uv`. Prefix commands with `uv run` locally. On Kaggle, `pip install` from `pyproject.toml`.
- Small pure functions. No state that exists only in a notebook cell.
- Short docstrings. No em-dashes in docs, comments, or commit messages.
- If a run fails the C1 check (forward accuracy not near zero after unlearning), mark it failed in the results file and stop. Do not tune until it passes.
