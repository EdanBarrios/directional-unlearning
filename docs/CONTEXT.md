# Working context

Session notes that are not derivable from the code or git history. Paste this into a new
chat alongside `docs/HANDOFF.md` and `CLAUDE.md` when picking the project back up.

## Machine and accounts

- Laptop as of 2026-09-20: Apple M4, 16 GB, MPS available, no CUDA. (The old work laptop
  was returned; HANDOFF v2 originally described an Intel Air, which was wrong.)
- Kaggle username `edanbarriosidrugo`. The current CLI reads a `KGAT_` token from
  `~/.kaggle/access_token`, **not** `kaggle.json`. Phone verification is done, so GPU
  sessions work. The token from the old machine should be regenerated.
- Git identity: `Edan Barrios <110215725+EdanBarrios@users.noreply.github.com>`.
- Repo is public, MIT. `uv sync` rebuilds the environment; `uv sync --extra cuda` adds
  bitsandbytes for Phase 5, and only works on a CUDA box.

## New machine, from scratch

```
brew install uv gh && gh auth login
git clone https://github.com/EdanBarrios/directional-unlearning.git
cd directional-unlearning && uv sync
SMOKE=1 uv run python -u -m src.finetune --config configs/phase1.yaml --seed 0 --no-save
```

Nothing under `checkpoints/` is committed, by design: every checkpoint is reproducible
from committed code plus its seed, and each results file records the commit it ran from.
Losing a checkpoint costs GPU time, never information.

## How to work with me on this

Edan is a production ML engineer and a first-time researcher. Explain the why behind a
step (what a control rules out, why a threshold is pre-registered) alongside doing it,
define research terms on first use, plain language first and then depth. Comprehension
checks are welcome; label them clearly as checks rather than as decisions, and say
explicitly when something is a decision only Edan can make. Short writing, no filler,
no em-dashes (HANDOFF section 1).

## Things learned the hard way

Each of these cost real time and is recorded so it is not rediscovered.

- **Wall clock is not compute.** A run reported 5.2 hours; its per-epoch times were
  `4.3 4.2 4.3 201.8 83.0 3.1 3.1 3.1 3.2 3.5` minutes. The laptop slept and a second GPU
  job was competing. Real compute was ~35 minutes. Run long jobs as
  `caffeinate -i uv run python -u -m ...` and never run a second GPU job while training.
- **Silence is buffering, not a hang.** Without `python -u`, a running job prints nothing
  until it exits.
- **Capture provenance at launch.** Results files record the git commit and dataset hash
  when the run *starts*. Reading them at write time silently mislabels a run whose code
  was edited while it was in flight.
- **Check utility at every phase that touches weights.** The design expected utility
  collapse from unlearning (Phase 2, control C2), but it happened in Phase 1 from ordinary
  finetuning: perplexity 48.75 -> 9033 while every accuracy metric looked perfect. See
  HANDOFF decision 18.
- **A measurement can be precise and still answer the wrong question.** One held-out
  template scored 0.07 because it ended in a bare `{name} ... is`, which 25 other
  templates had taught meant "give the description." See HANDOFF decision 15.
- **Reference timings, M4 fp32 Pythia-160m:** 618 ms/step at batch 32, ~3.2 min/epoch,
  22 s per in-loop eval. Grad clip is 42% of the step because MPS has no `foreach` path;
  on CUDA it is not.

## Companion artifacts

- Field guide to every concept and decision in the project, with self-checks:
  https://claude.ai/artifact/KMmSm6uBdXHnMXorKPG1rU
