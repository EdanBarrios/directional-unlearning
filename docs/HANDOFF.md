# Handoff v2: Directional Structure of LLM Unlearning

Status: design locked 2026-09-16. No code written. Supersedes v1.

## 0. How to use this document

Paste it into any new chat session. Claude Code reads `CLAUDE.md` automatically; this file supplies the design context that `CLAUDE.md` does not. Decisions get written back here, never left in chat. Sections 6 and 7 are the live parts.

---

## 1. Who is running this

- Edan Barrios. B.S. EECS, UC Berkeley, Summer 2026. AI Engineer at Paradigm Study.
- Applied to MATS Winter 2027, Empirical track. Submitted Sep 6, 2026. Decisions early-to-mid November. Stream selection at Stage 2, so streams are not fixed.
- Background: production ML engineering (RAG systems, CV pipelines, C++ systems). Strong on systems and architecture. Heavy user of CLI AI coding tools; unassisted PyTorch fluency not recently measured. Assume implementation help is welcome. Do not assume fluency with training loops or HuggingFace internals.
- Has read closely, comprehension verified: Chinchilla, Deep Double Descent, Pre-LN vs Post-LN, Reward Model Overoptimization, Tensor Programs V, The Reversal Curse.
- Has run zero experiments. This is the first. "Something finished" outranks novelty.

### Hardware and budget

- Local: Apple M4 MacBook, 16 GB unified memory, MPS available. (Corrected 2026-09-16; v2 originally said Intel.) Current torch (2.14) and transformers (5.17) via `uv`. A Pythia-160m full finetune in fp32 needs about 2.5 GB, so Phase 0 and Phase 1 iteration can happen locally. Not available locally: CUDA, so bitsandbytes NF4 (Phase 5) is Kaggle only.
- Compute: free tiers only. Kaggle is primary (about 30 GPU-hours/week, T4 or P100, 12-hour sessions, headless execution via Save Version). Colab and Lightning AI are backups.
- Everything is sized for Pythia-70m to 410m.

### Learning preferences

- Plain-language explanation first, then a comprehension check, then depth. Define research terms as they come up (glossary in section 11).
- Quiz distractors must be length-matched to the correct answer.
- Short writing, no filler, no em-dashes.

---

## 2. What changed from v1

1. The prior-work search ran on 2026-09-16. The v1 headline claim, that unlearning the forward direction of a fact leaves the reverse direction intact, has been measured at least three times (section 4b). It is now the expected replication result, not the finding.
2. New headline question: are the two directions independent entries in the weights? Tested through relearning speed (section 3, Q1).
3. Secondary question: does 4-bit quantization restore the two directions symmetrically (Q2).
4. Platform decided: Kaggle. Repo is public. Local machine downgraded to Intel.
5. Data decided: synthetic, one-to-one relations only, not TOFU.

---

## 3. Research questions

**Q0 (replication, required).** In a small model with verified bidirectional knowledge of synthetic facts, does unlearning the forward direction leave the reverse direction recoverable above chance?

Expected answer: yes. This is what BAKE, UGBench, and GONE found on larger models with pretrained facts. Q0 exists to prove the pipeline works and to establish the baseline the other questions need. If Q0 comes out "no," that is itself interesting (section 3, H0) and the project pivots to explaining why the controlled setting differs from prior work.

**Q1 (headline).** Are the forward and reverse directions of a fact stored as independent entries, or do they share structure?

Test: after unlearning the forward direction, finetune the forward direction back and measure how fast it recovers. Compare three starting points:

- reverse direction intact (U-fwd)
- reverse direction also unlearned (U-both)
- reverse intact, but an equal volume of unrelated facts also unlearned (U-fwd+dose), so total unlearning dose matches U-both

Hypotheses:

- **H-indep:** recovery speed is the same across the three. The reverse entry contributes nothing to forward relearning. Consistent with arXiv 2604.04943's claim that bidirectional training produces two distinct entries.
- **H-shared:** recovery is faster in U-fwd than in U-both, and U-fwd+dose looks like U-fwd, not U-both. The intact reverse entry scaffolds forward relearning, so the directions share structure.
- **H-dose:** U-fwd+dose looks like U-both. Then slower recovery is a general-damage effect, not a directional one, and the experiment says nothing about shared structure.

All three are interpretable. Pre-register the recovery threshold before running (section 7, item 4).

**Q2 (secondary).** After unlearning both directions, does 4-bit quantization restore the forward and reverse directions by the same amount?

Known: post-training quantization can restore unlearned knowledge, mainly for methods that keep weight changes small (NPO-style), because quantization rounds those small changes away. Unknown: whether the two directions restore symmetrically. If unlearning damage is asymmetric in magnitude by direction, quantization recovery will be asymmetric too.

**Q3 (optional, only if time).** Do different unlearning methods attack the relation or the entity? Gradient ascent and NPO push down output likelihood; RMU scrambles internal activations on forget-set inputs. Prediction: NPO leaves the reverse alive, RMU kills both directions and also damages unrelated attributes of the same entity. Requires implementing RMU. Skip unless Q0 through Q2 are done.

**H0 for Q0.** Unlearning generalizes across directions because the method damages the shared entity representation rather than a directional pathway. C5 (side-attribute survival) distinguishes this from directional removal.

---

## 4. Prior work

### 4a. The Reversal Curse

Berglund, Tong, Kaufmann, Balesni, Cooper Stickland, Korbak, Evans (2023), arXiv 2309.12288, ICLR 2024.

Finetune a model on "A is B" and it learns nothing measurable about "B is A." The log-probability of the correct reversed answer is statistically indistinguishable from a random name. Method: fictitious celebrities to avoid pretraining contamination; three finetuning subsets (NameToDescription, DescriptionToName, Both); test the unseen direction. Validated on real celebrity parent pairs with GPT-4 (about 79% forward, 33% reverse). Does not improve with scale. Not fixed by paraphrase augmentation. Does not appear in in-context learning: the fact in the prompt reverses fine. This is a claim about weight-encoded storage, not reasoning.

Mechanism: the autoregressive objective learns conditional probabilities, which are directional. Training on "A is B" puts gradient on the pathway from A's representation toward producing B. B's representation receives no gradient linking back to A. Complementary view: feed-forward layers act as key-value memories (Geva et al., arXiv 2012.14913) and reading a value does not trigger its key.

Independent confirmation via influence functions: Grosse et al. (2023), arXiv 2308.03296.

Related failures: two-hop curse (Balesni et al., 2024); Physics of Language Models (Allen-Zhu and Li), models cannot do reverse search without generating reasoning tokens; Negation Neglect (Truthful AI, 2026); binding-problem reframing (arXiv 2504.01928).

### 4b. Directly overlapping work (found 2026-09-16)

**BAKE / BIRD.** Ma, Gu, Ling, Liu, Liu. "Untying the Reversal Curse via Bidirectional Language Model Editing." arXiv 2310.10322, ICML 2024. Model editing, not unlearning. Builds a benchmark of counterfactual edits on Wikidata relations with reverse-direction probes. Finding: FT, MEND, KN, ROME, MEMIT all recall the edited fact in the edit direction and fail in reverse (ROME on LLaMA-2: 99.70% forward, 0.26% reverse QA). For this project the key finding is section 6.5: after editing, the probability of the *original* answer in the reverse direction is almost unchanged for every method except KN. The old fact survives in reverse. Restricts reverse QA to one-to-one and one-to-many relations because many-to-one relations have no unique reverse answer.

**UGBench / PerMU.** Wang, Jing, Sun, Wang, Wang, Liao, Tao. "Erasing Without Remembering: Implicit Knowledge Forgetting in Large Language Models." arXiv 2502.19982, Feb 2025. Defines an "unlearning scope" that includes rephrased, subject-replaced, one-hop, and relation-reversed samples. Evaluates 13 methods (GA, DPO, NPO, task vectors, WHP, ULD, ICL, plus GD and KL regularized variants) on Phi-1.3B and LLaMA2-7B. On the ZsRE reversed-relation split, unlearned models score F1 81 to 96 where the retain-only model scores 50. Also finds the correct answer token re-emerges in the unlearned model's middle layers. Does not cite Berglund. Frames the result as a generalisation failure. Does not verify that the reverse direction was known before unlearning; the reverse came from pretraining.

**GONE / NEDS.** Dahal, Balasubramaniam, Xiong. "GONE: Structural Knowledge Unlearning via Neighborhood-Expanded Distribution Shaping." arXiv 2603.12275, Mar 2026. Knowledge-graph benchmark on Wikidata and ConceptNet with an explicit inverse probe. Filters to probes the base model answers correctly before unlearning, so the reverse was known. NPO on Llama-3-8B: direct UE 0.635, inverse UE 0.209. GA on Mistral-7B: 0.985 direct, 0.322 inverse. Frames it as leakage. Does not cite Berglund.

**"The Illusion of Latent Generalization: Bi-directionality and the Reversal Curse."** arXiv 2604.04943, Apr 2026. Bidirectional supervision (MLM or masking-based decoder training) fixes the behavioral curse, but representational analysis shows the two directions are not coupled in representation space more than unrelated facts are. Conclusion: two distinct entries per fact, one per direction. This predicts Q0 = yes and is the claim Q1 tests causally.

**What none of them do.** Connect to the Reversal Curse mechanism. Control the provenance of the reverse direction (train both directions together, then unlearn one). Include an entity-vs-relation control. Use Berglund's log-prob-vs-random-name metric. Test relearning speed by condition. Test quantization by direction.

### 4c. Unlearning background

Machine unlearning removes the influence of specific data while preserving the rest. Used for privacy compliance and hazardous-capability removal.

Methods used here:

- **Gradient ascent (GA):** maximize the loss on forget examples. Simple, damages utility badly.
- **Gradient difference (GA+GD):** GA on forget examples plus normal training on retain examples. The "GD" is the retain-set regularizer.
- **NPO (Negative Preference Optimization):** Zhang, Lin, Bai, Mei, arXiv 2404.05868. Treats forget examples as dispreferred in a DPO-style objective against a frozen reference model. Bounded loss, so it avoids GA's collapse. β controls how far the model may drift from the reference; 0.1 is the common default. Used with a retain regularizer as NPO+GD.
- **RMU:** Li et al., WMDP, arXiv 2403.03218. Pushes internal activations on forget inputs toward a random direction. Only relevant to Q3.

Benchmarks: TOFU (arXiv 2401.06121, fictitious authors), MUSE (2407.06460), WMDP (2403.03218), PISTOL (2406.16810, synthetic contract graphs), RWKU.

Unlearned models are broadly not robust to recovery. The standard attack suite, following Lucki et al. (arXiv 2409.18025): logit lens, finetuning, orthogonalization, enhanced GCG, pruning. None is directional; every one probes the direction the fact was stated.

Other known fragilities:

- **Quantization:** Zhang et al., arXiv 2410.16454, ICLR 2025. 4-bit post-training quantization can restore up to 83% of forgotten knowledge, mainly for methods with utility constraints that keep weight changes small. GA without constraint makes large changes and quantization does not recover them. This is why Q2 uses NPO.
- **Relearning:** benign finetuning on a small amount of related data restores forgotten knowledge quickly (Hu et al., "Jogging the memory of unlearned LLMs via benign relearning," 2024). Q1 turns this attack into a measurement.
- **Continual unlearning:** knowledge erosion and "forgetting reversal" (arXiv 2604.19108). There, "reversal" means temporal recurrence across unlearning phases, not directional reversal. Do not conflate.
- The literature increasingly describes unlearning as obfuscation rather than deletion under realistic threat models.

Related evaluation work: Wu et al., "Evaluating Deep Unlearning" (2410.15153), whether unlearning a fact also removes facts that logically imply it. Wei et al., "Do LLMs Really Forget?" (2506.05735), residual knowledge via correlation with confidence.

### 4d. Knowledge localization

ROME (2202.05262) and MEMIT (2210.07229): factual associations live in middle MLP layers, found by causal tracing. Geva et al.: feed-forward layers as key-value memories. Only needed if the project extends into where surviving knowledge lives.

---

## 5. Experimental design

### 5a. Model

Pythia-160m (EleutherAI, arXiv 2304.01373) as primary. Open training data, published checkpoints, a family from 70m to 12b for a cheap scaling extension. Pythia-70m for smoke tests. Pythia-410m as the scaling extension after everything works on 160m.

Pythia is a base model: prompts are completions, not questions. No chat template.

### 5b. Data

150 fictitious entities, split 50 forget / 50 dose / 50 retain. Each entity has:

- a **name**: fictitious first + last name, checked against the base model (Phase 0) for zero prior knowledge
- a **target description D**: unique, one-to-one (e.g. "the composer of Abyssal Melodies"), the relation that gets unlearned
- a **side attribute S**: a second, unrelated fact about the entity (e.g. a fictional town of residence), never unlearned. Survival of S after unlearning D is the entity-vs-relation control (C5).

Relation type is one-to-one only. Reason: the reverse direction must have a unique correct answer, or reverse accuracy is undefined. BAKE restricts reverse QA to one-to-one and one-to-many relations for this reason. Symmetric relations (sibling-of) and many-to-one relations (author-of) are excluded from v1.

Templates:

- D forward: 30 templates ("<Name> is <D>", "Known as <D>, <Name>...", etc.). 25 for training, 5 held out for testing.
- D reverse: 30 templates ("<D> is <Name>"). Same 25/5 split.
- S forward: 10 templates, 8/2 split. Only the forward direction is trained and tested; S exists to measure entity survival, not directionality.

All test prompts use held-out templates so accuracy measures the fact, not the string.

Generation: names and descriptions generated by an LLM offline, then filtered for uniqueness and for non-collision with real entities. Method of generation is an open item (section 7, item 8). Dataset is fixed across seeds; seeds vary training order (section 5f).

### 5c. Phases

**Phase 0, contamination check.** Base Pythia-160m on all facts, both directions. Log-prob margin (section 5e) should be about zero everywhere and greedy generation should produce nothing from the dataset. Any entity with nonzero prior knowledge is replaced.

**Phase 1, bidirectional training.** Full finetune M0 → M1 on all 150 entities, both directions of D, forward of S. Success criteria: held-out accuracy ≥ 90% in both directions on both the forget and retain sets, margin well above zero. This is the critical departure from Berglund, who trained one direction. The experiment requires verified bidirectional knowledge. Save M1 as a Kaggle Dataset so every later phase starts from it without retraining. This is the first real milestone.

**Phase 2, unlearning.** From M1, per method (GA+GD, NPO+GD), per seed, three conditions:

| condition | forget-set D forward | forget-set D reverse | dose-set D forward | example groups ascended |
|---|---|---|---|---|
| U-fwd | yes | no | no | 50 |
| U-both | yes | yes | no | 100 |
| U-fwd+dose | yes | no | yes | 100 |

The retain regularizer uses the retain set only, in all conditions. The dose set is untouched in U-fwd and U-both.

Stop rule: evaluate every 10 steps on the held-out forward prompts of the forget set; stop when margin ≤ 0 and accuracy ≤ 5%, with a hard cap on steps. Record steps taken. If the cap is hit before the rule, the run is marked "unlearning failed" and excluded (C1).

**Phase 3, bidirectional probing.** Every unlearned model, both directions, all three sets, side attribute, held-out perplexity. Answers Q0 and produces the C1 to C6 tables.

**Phase 4, relearning.** From each unlearned model, finetune on the forget set's forward D using a small subset of training templates (5 per fact) at a low learning rate. Evaluate held-out forward margin and accuracy every 5 steps up to a cap. Report steps-to-threshold and area under the recovery curve. Also run from M1 as a sanity check (should start at ceiling). Answers Q1.

**Phase 5, quantization.** NF4 4-bit via bitsandbytes on M1, U-fwd, and U-both (NPO+GD variants). Probe both directions. Compare recovery of forward vs reverse in U-both against M1's own degradation under quantization. Answers Q2. CUDA only.

### 5d. Controls, all required

- **C1, unlearning validity.** Forward accuracy on the forget set near zero after Phase 2. If not, nothing downstream is interpretable.
- **C2, utility retention.** Retain-set accuracy both directions, and held-out perplexity on a small general-text slice. A surviving reverse direction in a wrecked model means nothing.
- **C3, full-removal upper bound.** U-both. Establishes what removing both directions looks like and costs.
- **C4, chance baseline.** Log-prob margin against alternatives, section 5e. Chance is zero.
- **C5, entity vs relation.** Side attribute S survival after unlearning D. If S dies, the entity representation was damaged and reverse-direction failure says nothing about directionality.
- **C6, dose match.** U-fwd+dose. Separates "more unlearning updates" from "reverse direction unlearned" in Q1.

### 5e. Metrics

- **Log-prob margin (primary).** For each test prompt, mean per-token log-probability of the correct answer under teacher forcing, minus the mean of the same quantity over the other 149 entities' answers as alternatives. Per-token normalization because answers differ in length. Chance = 0 on average. This is Berglund's metric with length normalization; it separates "learned nothing" from "learned weakly," which accuracy cannot.
- **Corrected margin (added 2026-09-17 after Phase 0).** Phase 0 showed per-fact raw margins on the base model spread from -5.8 to +3.1, and a prompt-swap diagnostic showed the spread is entirely an answer-string prior (e.g. " Winterfell" scores +3 after anyone's name). The probe already scores every answer under every prompt, so each answer's prior is its mean row-relative score under other entities' prompts for the same template. Corrected margin = raw margin - prior. Chance = 0 per fact. Reported alongside raw, never instead: raw falling while corrected holds means the answer string was suppressed, not the entity link. The C1 stop rule stays on raw margin and accuracy as pre-registered.
- **Accuracy.** Greedy decoding, exact match on the answer span. Secondary.
- **Retain accuracy and margin**, both directions.
- **Side-attribute accuracy and margin.**
- **Held-out perplexity** on about 1k sequences of general text.
- **Phase 2 steps to stop rule.**
- **Phase 4 steps-to-threshold and recovery AUC.**

### 5f. Seeds and statistics

5 seeds per condition per method. The dataset is fixed; a seed controls data shuffling order in Phases 1, 2, and 4. Report mean and standard deviation across seeds. Q1 comparisons are paired across seeds. Q0 per-fact analysis uses the 50 forget facts as units. State every test in advance; do not add tests after seeing results.

### 5g. Compute estimate

150 entities × (25 + 25 + 8) training templates ≈ 8.7k sequences of about 20 tokens. Phase 1 at 10 epochs on a T4: minutes. All Phase 2 conditions for one seed and one method: under 30 minutes. The whole v1 grid is under 5 GPU-hours. Leaves quota for 410m.

---

## 6. Decisions locked (2026-09-16)

| # | decision | rationale |
|---|---|---|
| 1 | Pythia-160m primary, 70m smoke, 410m later | open data, cheap scaling |
| 2 | Synthetic facts, not TOFU | TOFU QA pairs do not decompose into forward/reverse |
| 3 | One-to-one relations only | reverse must have a unique answer (BAKE) |
| 4 | 150 entities: 50 forget / 50 dose / 50 retain, 30+30+10 templates | power vs compute; dose set needed for C6 |
| 5 | GA+GD and NPO+GD | one simple, one standard, so a null cannot be blamed on method choice |
| 6 | 5 seeds per cell | single-seed results are a red flag |
| 7 | Headline Q1 relearning asymmetry; secondary Q2 quantization | the gaps prior work leaves open |
| 8 | Full finetune, fp32, hand-written loop, no LoRA, no HF Trainer | LoRA confounds storage; custom losses fight Trainer |
| 9 | Kaggle for the seed grid and Phase 5; local M4 for smoke tests and Phase 0/1 iteration (amended 2026-09-16) | headless, quota-friendly; local MPS is free |
| 10 | Public repo, MIT license | nothing to protect; evidence of running experiments |
| 11 | Entities built from committed word lists by a seeded combinator (`gen_facts.py`), not free-form LLM generation (locked 2026-09-16) | uniqueness by construction; deterministic regeneration; a Phase 0 reject is replaced by the next draw. Word lists themselves are LLM-written once and committed |
| 17 | Phase 4 protocol pre-registered 2026-09-18, before any unlearned checkpoint exists, in `configs/relearn.yaml`: lr 1e-5 (same as Phase 1, so it is a benign finetune and not a stronger push), 5 of the 25 training templates chosen by a fixed seed, batch 32, cap 200 steps, probe every 5 steps, recovery threshold 0.90 (90% of M1's forget-set forward accuracy of 1.00). Reported: steps-to-threshold and normalized recovery AUC | Q1's answer is which curve rises first, so a threshold picked after seeing the curves would pick itself. AUC is reported alongside because it separates "recovered late but fully" from "recovered early", which steps-to-threshold cannot |
| 16 | Control thresholds pre-registered 2026-09-18, before any Phase 2 run exists, in `src/evaluate.py::THRESHOLDS`: C1 forget-set forward accuracy <= 0.05; C2 retain accuracy >= 0.80 in both directions and perplexity ratio <= 1.50 against M1; C5 forget-set side-attribute accuracy >= 0.80. A run where any of these fails is marked not interpretable for Q0 and is not tuned until it passes | the pass/fail judgement has to exist before the interesting numbers do, or it gets chosen to fit them. C3, C4, C6 are design-level (whole conditions, or the metric's definition) so they are not per-run checks |
| 15 | s_forward held-out template 9 changed from "The place {name} lives is" to "You will find {name} in" (2026-09-18) | Per-template audit on the lr1e-5 checkpoint: that one template scored 0.07 while the other 11 held-out templates across all three directions scored 0.97 to 1.00. The prompt ends in a bare "{name} ... is", which 25 d_forward templates teach is followed by a description, so the model answered with the description. C5 is the entity-vs-relation control and needs a real ceiling; a 52% baseline cannot detect the drop it exists to detect. Phase 1 rerun after the change |
| 14 | Training recipe (2026-09-17): loss on answer tokens only (prompt masked -100, same span as the probe and as Phase 2 losses); AdamW, weight decay 0, constant LR, batch 32, grad clip 1.0; per-epoch quick eval on 10 facts per set with 20 alternatives, end-of-run eval on all facts with 20, full 149-alternative probe only on the chosen M1 | answer-only loss puts gradient on the association under study; everything else is the plainest defensible default |
| 13 | Corrected margin reported alongside raw margin (2026-09-17, see 5e). Results files store the full candidate score vector per prompt so priors are recomputable | Phase 0 showed the raw per-fact margin carries an answer-string prior of several nats that is unrelated to the entity |
| 12 | Probe details (locked 2026-09-17): margin alternatives are all 149 other answers in Phases 0, 3, 5 and a fixed seeded subset of 20 per fact for in-loop checks in Phases 2, 4. Accuracy is greedy generation of len(answer) tokens, decoded text must start with the answer. Perplexity on the first 1,000 prose lines of WikiText-2 test, max 256 tokens each | in-loop probes run dozens of times per run; 20 alternatives is stable and comparable step to step. Decoded-text match asks "did it write the answer," not "did it pick our tokenization." WikiText-2 because the Pile is no longer distributed |

---

## 7. Open decisions

1. Phase 1 learning rate and epochs. Swept 2026-09-17/18, seed 0, 10 epochs, batch 32. lr 1e-5: D forward 1.00/1.00/1.00, D reverse 1.00/0.99/0.99. lr 3e-5: forward 0.90/0.88/0.92 and a visibly unstable curve. lr 1e-4: see `results/phase1/`. Pick 1e-5, which is both the best and the smallest, as pre-registered. Confirm on the post-template-fix rerun before locking.
2. NPO β (default 0.1) and the retain-regularizer weight (default 1.0). Use defaults unless C2 fails.
3. Phase 2 stop-rule thresholds and step cap. Proposed: margin ≤ 0 and accuracy ≤ 5%, cap 500 steps. Confirm after the first GA run shows the curve.
4. Resolved 2026-09-18 as decision 17, pre-registered in `configs/relearn.yaml` before any unlearned checkpoint existed.
5. Quantization tool: bitsandbytes NF4 vs GPTQ. And whether Pythia-160m survives NF4 at all; M1-quantized decides this.
6. Whether to run Q3 (RMU).
7. Whether to run 410m.
8. Resolved 2026-09-16, see decision 11. Remaining sub-questions: word list contents, name tokenization, description shape. See section 5b once settled.
9. Whether seeds should also regenerate the dataset. Default no, for interpretability.
10. Phase 0 rejection threshold. Phase 0 ran 2026-09-17 (`results/phase0/base.json`, commit 27dff65): set-level margins +0.004 / +0.015 / -0.020, 0 of 1800 greedy hits, ppl 48.75. Prompt-swap diagnostic on all outliers: swapped margin equals own margin, so no entity is known; the per-fact spread is answer-string prior. Rule (2026-09-17): reject on any greedy hit, corrected margin > 0.5 in any direction, or a name-level collision with a famous real or fictional entity. First two: none. Third: " Winterfell" (entity 47, retain) swapped for spare "Victor Gibson" via `gen_facts.py --reject 47`. Rerun `base_v2.json` (stamped `dcd6e2d-dirty`: Phase 1 files were edited while it ran; probe numerics regression-checked identical, dataset is commit dcd6e2d's) showed corrected per-fact SD 0.09 to 0.17 and one entity over the 0.5 cutoff: id 148, d_rev corrected +0.62. Rule applied as written: 148 swapped for the next spare (`--reject 47 148`). Final Phase 0 on the final dataset is `results/phase0/base_v3.json`. Rejections are recorded in `facts.json["rejected"]`.

---

## 8. Threats to validity

- **Utility collapse.** Aggressive unlearning wrecks the model. C2. Do not skip.
- **Entity deletion.** Unlearning removes the entity wholesale; both directions fail for the wrong reason. C5.
- **Dose confound in Q1.** U-both has had twice as many examples ascended as U-fwd, so slower relearning could be general damage. C6 matches the dose. Also report retain-set utility at the start of Phase 4 and condition on it.
- **Synthetic-data artifact.** Facts learned by short finetuning may be stored differently from pretraining facts. Real limitation; state it. Prior work on pretrained facts (GONE, UGBench) partially covers this direction.
- **Contamination.** Phase 0.
- **Weak unlearning.** If the forward direction never really left, everything downstream is noise. C1 and the stop rule.
- **Quantization destroys the small model.** NF4 on 160m may degrade everything. M1-quantized is the baseline for Q2.
- **Length-normalization artifacts** in the margin metric. Report raw sums as a secondary check.
- **Multiple comparisons.** Pre-register tests; report all of them.

---

## 9. Workflow

Repo layout:

```
data/gen_facts.py        synthetic entities and templates → data/facts.json
src/model.py             load model + tokenizer, pad token, device
src/probe.py             log-prob margin and accuracy, both directions
src/train_loop.py        the one training loop, takes a loss fn
src/losses.py            lm loss, GA, NPO
src/finetune.py          Phase 1
src/unlearn.py           Phase 2, takes condition + method, writes checkpoint + steps
src/relearn.py           Phase 4
src/quantize.py          Phase 5
src/evaluate.py          Phase 3 tables from checkpoints → results/*.json
configs/*.yaml           one per phase and method
kaggle_launch.ipynb      clone, install, run
docs/HANDOFF.md          this file
CLAUDE.md                conventions for Claude Code
```

Rules:

- Every script supports `SMOKE=1`: Pythia-70m, 5 facts, 2 templates, 3 steps. Run it locally before any Kaggle run.
- Every results JSON records config, git commit hash, seed, and metrics. New run, new file. Never overwrite.
- Commit before every Kaggle run so results trace to a commit.
- Never commit tokens, checkpoints, or the HF cache.
- Chat sessions for reading and design, seeded with this file. Claude Code for code. Decisions written back here.

---

## 10. Immediate next steps

1. Create the repo: this file, `CLAUDE.md`, `.gitignore`, MIT license, README stub that states the project as replication plus extension and cites BAKE, UGBench, GONE, 2604.04943.
2. `gen_facts.py` and `probe.py`. Run Phase 0 on Kaggle. First real number.
3. `finetune.py`. Run Phase 1 until success criteria hold. Save M1 as a Kaggle Dataset. First real milestone.
4. `unlearn.py` with GA+GD. Verify C1 and C2 on one seed. Then NPO+GD.
5. Phase 3 across the grid. Q0 table.
6. `relearn.py`. Pre-register item 7.4. Phase 4. Q1.
7. `quantize.py`. Phase 5. Q2.
8. Write up. Workshop-paper shape: controlled replication, then the question prior work could not ask.
9. Read arXiv 2604.04943 this week; it is the mechanistic claim Q1 tests.

---

## 11. Glossary

- **Log-prob margin:** log-probability the model assigns to the correct answer minus the average it assigns to wrong alternatives. Zero means the model cannot tell them apart.
- **Teacher forcing:** feed the model the prompt plus the gold answer and read off the probability it assigned to each answer token, rather than letting it generate.
- **Held-out:** data the model never trained on, used only for testing.
- **Paraphrase / template:** a different surface form of the same fact. Training on many and testing on unseen ones checks the fact was learned, not the sentence.
- **Forget set / retain set:** facts to remove / facts that must survive.
- **Dose:** here, the number of examples the unlearning method pushed against. Matched across conditions so damage volume is not a confound.
- **Gradient ascent:** training in reverse, increasing the loss on chosen examples.
- **Gradient difference:** gradient ascent on the forget set plus normal training on the retain set.
- **NPO:** a bounded unlearning loss that treats forget examples as dispreferred relative to a frozen copy of the original model.
- **Reference model:** the frozen copy of the pre-unlearning model that NPO measures drift against.
- **RMU:** unlearning by scrambling internal activations on forget inputs rather than by changing output likelihoods.
- **NF4:** 4-bit normal-float, the standard post-training quantization format in bitsandbytes.
- **Post-training quantization (PTQ):** compressing weights to fewer bits after training, without retraining.
- **Relearning attack:** briefly finetuning an unlearned model on related data to bring the forgotten knowledge back.
- **Logit lens:** reading the model's next-token predictions from intermediate layers.
- **Causal tracing:** corrupting and restoring internal activations to find which ones carry a fact.
- **Construct validity:** whether a measurement measures the thing it claims to.
- **Seed:** the number that fixes random choices (shuffle order, initialization) so a run can be repeated.
- **One-to-one relation:** each subject maps to exactly one object and vice versa, so both directions have a unique answer.
- **Perplexity:** exponentiated average loss on text; lower means the model predicts the text better. Used here as a general-utility check.

---

## 12. Reading list

Verified IDs:

- Berglund et al. 2023, The Reversal Curse, 2309.12288
- Grosse et al. 2023, influence functions, 2308.03296
- Ma et al. 2023, BAKE/BIRD, 2310.10322
- Wang et al. 2025, UGBench/PerMU, 2502.19982
- Dahal et al. 2026, GONE/NEDS, 2603.12275
- The Illusion of Latent Generalization, 2604.04943
- Reversal curse as binding problem, 2504.01928
- Zhang et al. 2024, quantization recovers unlearning, 2410.16454
- Zhang et al. 2024, NPO, 2404.05868
- Fan et al. 2024, SimNPO, 2410.07163
- Maini et al. 2024, TOFU, 2401.06121
- Shi et al. 2024, MUSE, 2407.06460
- Li et al. 2024, WMDP and RMU, 2403.03218
- Qiu et al. 2024, PISTOL, 2406.16810
- Lucki et al. 2024, adversarial perspective on unlearning, 2409.18025
- Wu et al. 2024, deep unlearning, 2410.15153
- Wei et al. 2025, Do LLMs Really Forget, 2506.05735
- Park et al. 2026, continual unlearning, 2604.19108
- Meng et al. 2022, ROME, 2202.05262
- Meng et al. 2023, MEMIT, 2210.07229
- Geva et al. 2021, FF layers as key-value memories, 2012.14913
- Biderman et al. 2023, Pythia, 2304.01373

Verify before citing (recalled, not checked this session):

- Balesni et al. 2024, The Two-Hop Curse
- Hu et al. 2024, Jogging the memory of unlearned LLMs via benign relearning
- Mayne et al. 2026, Negation Neglect (Truthful AI)
