# Experiment log

*Training a Next-Concept Model on a single 12 GB GPU* — Gianluca Meola. NCM paper: arXiv 2610.14162.
All experiments of 2026-11-16 to 2026-11-20, in chronological order. Data: `results/runs/<run>/`, `results/runs_index.csv`, cycle tables in `results/runs/_cicli/`.

## How the experiments were run

- One RTX 3080 Ti (12 GB); TinyStories, 128 BPE tokens per story; generator of 88.6M parameters for the standard 12-level configuration (other level counts change the parameter count).
- Comparisons only at the same number of stories seen (standard: 51,200 = 400 steps × 128), same data order, one variable changed.
- Gen. PPL = generative perplexity with GPT-2 large on 1,280 generated samples = 256 × 5 sampling seeds; "±" is the standard deviation over those 5 seeds. Lower is better.
- Entropy of the generated text is a guard: not more than 0.1 below the reference, otherwise a low Gen. PPL may just be repetitive text.
- Thresholds, from the seed noise measured at 51,200 stories (2.4% per run): 6.7% between single runs, 4.7% between means of two seeds. The first cycles used earlier thresholds, given in their sections.
- "Seed" = model initialisation/training seed (42, 43); the data order is fixed.
- Decision rules were written before looking at the data. Gen. PPL decides; validation loss and reconstruction are reported only, and losses are not comparable across tokenizers.
- All trainings are short: as reported in the paper (Tables 5-6), the official tokenizer saw about 123 million stories and the official generator 102 million; my longest runs saw 9.8 million and 409,600.

## Glossary

| Label | Meaning |
|---|---|
| tokuff, tokuff64 | Official tokenizer; the same with codes computed in 64-bit precision |
| mytok | My tokenizer |
| level 0 | The coarsest level, one code per story; at generation time it is sampled from stored frequencies, the generator predicts levels 1-11 |
| `std`, `lin` | Level spacing as in the official tokenizer; linear |
| R1 (`-dati`) | Codebooks initialised from data (k-means) |
| R2 (`-rv`) | Restart of nearly unused codes (usage statistic below 0.1) up to a fixed step: the first 307,200 stories (step 1,200 at batch 256) |
| R12 | Both |
| `-nonorm` | My tokenizer without level-0 normalisation |
| paper recipe (`lin75-25_lr3e-4`) | My reconstruction: linear corruption 0.75→0.25, lr 3e-4 |
| best recipe (`c50_lr1e-3_flex`) | Constant corruption 0.5 + FlexAttention + lr 1e-3 |
| `cNN`, `lin25-75`, `unif0-100`, trailing `l0` | Constant corruption 0.NN; linear increasing; one rate per story; level 0 corrupted too |
| standard generator | 88.6M parameters, 51,200 stories |
| `-pNNNN`, `-ts43` | Generator on a tokenizer snapshot at step NNNN; on a tokenizer trained with seed 43 |
| codes in use | Distinct codes chosen on 4,096 validation stories, out of 4,096 per level (`codici_usati`) |
| live codes | Codes whose usage statistic in the codebook is at least 0.1 |
| perfect stories | Perfectly reconstructed validation stories (`sequenze_perfette`) |
| cycle | Runs compared in one table (`configs/cicli/`); `_interrotta`, `_annullata` = interrupted, cancelled, never used |
| Italian words in file names | `ciclo` cycle, `catena` chain of cycles, `conferme` confirmations, `tenuta` hold-up, `coda` tail, `ricetta` recipe, `passi` steps, `rimedi` remedies, `calibrazione` calibration, `livelli` levels, `generatore-lungo` long generator, `norma` normalisation, `ufficiale` official |

**Index:** [1 Reference](#1-reference-measurements-on-the-official-checkpoints) · [2 Resume](#2-pause-and-resume) · [3 First run](#3-first-generator-run-and-gen-ppl-protocol-check) · [4 Cycles 1-2](#4-cycles-1-and-2-input-corruption-and-flexattention) · [5 Cycle 3](#5-cycle-3-budget-low-corruption-rates-learning-rate) · [6 Batch](#6-tokenizer-batch-size-judged-on-reconstruction) · [7 First tokenizer](#7-my-first-tokenizer-against-the-official-one) · [8 Fix](#8-is-my-tokenizer-fix-really-a-fix) · [9 Remedies](#9-remedies-against-code-collapse) · [10 Levels](#10-number-of-levels) · [11 Doubling](#11-tokenizer-training-doubled) · [12 Batch on text](#12-tokenizer-batch-size-judged-on-generated-text) · [13 Test C](#13-test-c-official-tokenizer-with-exact-codes) · [14 Test A](#14-test-a-generator-four-times-longer) · [15 Test B](#15-test-b-two-extremes-on-the-tokenizer) · [16 Fragility](#16-where-are-my-codes-fragile)

---

## 1. Reference measurements on the official checkpoints

*2026-11-16*

- **Question.** How well does the official tokenizer reconstruct, what validation loss does the official generator reach, and is my reconstruction of the generator's input format right?
- **Setup.** No training. `evaluation/reference_official.py`, first 4,096 validation stories, teacher forcing with a one-token EOS prefix, targets = codes of levels 1-11 (640 positions); seed 0, FP32.
- **Decision rule.** None recorded.
- **Results.**

| First k levels | 1 | 4 | 8 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| Tokens correct (no padding) | 9.6% | 35.9% | 94.1% | 98.2% | 99.0% | 99.98% |
| Perfect stories | 0 | 0 | 0.4% | 17.7% | 35.0% | 93.3% |

Generator loss per code (nats): official 1.360; input shuffled between stories 16.32; random weights 8.17 (ln 4,096 = 8.32). Official per level: L1 6.64 · L2 5.10 · L6 1.24 · L7 0.47 · L11 0.27. Level 0: 1,632 distinct codes on the validation stories; the frequency table stored in the checkpoint has 2,558 non-zero entries, minimum exactly 1/65,536.

- **Conclusion.** Input format confirmed. The weights show an embedding-only encoder and a one-block decoder, as in the paper. The tokenizer is nearly lossless with 12 levels; coarse levels alone do not reconstruct the text. Unverified: that the level-0 table is a count over 65,536 sequences. A first version of the script computed the partial-level reconstructions wrongly (an in-place accumulation made every partial level equal to the last); the table is from the corrected script, and the generator numbers were never affected.

## 2. Pause and resume

*2026-11-16*

- **Question.** Does a training paused and resumed in a new process give exactly the same numbers as one never interrupted?
- **Setup.** `tests/test_resume.py`: VQ-VAE with fresh weights, 12 steps, micro-batch 16, accumulation 2, BF16, deterministic mode. A1, A2: continuous. B: paused at step 6, resumed in a new process.
- **Decision rule.** None recorded; the comparison is exact equality.
- **Results.**

| Difference | Loss per step | Weights | Buffers | Optimizer | Data state |
|---|---|---|---|---|---|
| A1 vs A2 | 0 | 0 | 0 | 0 | 0 |
| A1 vs B (resumed) | 0 | 0 | 0 | 0 | 0 |

Step-12 loss 8.2682213783 in all three. Stop-file pause on a 400-step run: stopped at step 36, checkpoint (362 MB) written in 1.5 s.

- **Conclusion.** Resume is bit-identical in deterministic mode, and the stop-file pause works. Not tested automatically: pause by keyboard interrupt (same code path). Not measured: without deterministic mode continuous runs may already differ slightly, so bit equality is not expected there.

## 3. First generator run and Gen PPL protocol check

*2026-11-16*

- **Question.** Does my stage-2 script learn? Does my Gen. PPL evaluation reproduce the paper's value on the official checkpoint?
- **Setup.** Run `gen-tokuff_38400_300x128_lin75-25_lr3e-4_s42`: official tokenizer, 300 steps × 128 = 38,400 stories (0.4% of a reduced run I had planned, 10,000 × 1,024 = 10.24M stories; 0.04% of the official generator's budget), lr 3e-4 → 3e-5, warmup 30, level 0 not corrupted. Evaluation `evaluation/gen_ppl_run.py`: sampling seeds 42, 0, 1, 2, 3, temperature 1, no top-k/top-p.
- **Decision rule.** For the official checkpoint: Gen. PPL between 15.0 and 16.6 and entropy between 3.9 and 4.05; outside, stop.
- **Results.** Validation loss 8.17 (random weights) → 6.47 → 5.36 → 4.99 at steps 100, 200, 300 (official model 1.36). At step 300: L1 6.82, L2 7.44, L11 2.51 (official 6.64, 5.10, 0.27).

| Model | Gen. PPL | ± | Entropy |
|---|---|---|---|
| Official, my measurement | 16.58 | 0.47 | 3.969 |
| Official, as reported in the paper | 15.41 | — | 3.98 |
| 300-step run | 214.07 | 5.24 | 3.890 |

- **Conclusion.** The script learns; fine levels learn much faster than coarse ones (L1, L2 almost flat). The acceptance test passed at its upper edge. The +1.2 from the paper is unexplained (unverified guesses: library version, sampling differences); all later comparisons refer to my 16.58. Sampling-seed noise is about 3% of the value. Nothing can be extrapolated from 300 steps (the learning rate had already reached its minimum). Unverified: with an unweighted per-code loss level 1 is 8/640 of the loss, so coarse levels get little signal.

## 4. Cycles 1 and 2: input corruption and FlexAttention

*2026-11-16 to 2026-11-17*

- **Question.** All else equal, which input-corruption scheme gives the best text, and is FlexAttention usable? Do the gains repeat with a second seed and add up?
- **Setup.** `configs/cicli/2026-11-16_ciclo-corruzione.yaml` (6 runs), then `2026-11-17_ciclo2-corruzione.yaml` (6 more). Official tokenizer, 51,200 stories (0.05% of the official generator's budget), lr 3e-4 → 3e-5, warmup 20.
- **Decision rule.** A variant wins only if it improves the base by more than twice the noise without lowering entropy by more than 0.1. Noise from the first seed pair: 2.50, so 5 points.
- **Results.**

| Variant | Seed | Gen. PPL | Entropy | Val loss |
|---|---|---|---|---|
| constant 0.35 | 42 | 131.31 | 3.923 | 3.886 |
| constant 0.5 + Flex | 42 | 131.48 | 3.888 | 4.383 |
| constant 0.5 | 42 / 43 | 134.72 / 140.92 | 3.871 / 3.895 | 4.402 / 4.417 |
| Flex | 42 / 43 | 142.64 / 143.01 | 3.899 / 3.892 | 4.397 / 4.413 |
| linear reversed 0.25→0.75 | 42 | 144.71 | 3.889 | 4.422 |
| base: linear 0.75→0.25 | 42 / 43 | 149.42 / 151.92 | 3.897 / 3.913 | 4.444 / 4.445 |
| level 0 corrupted too | 42 | 150.65 | 3.906 | 4.563 |
| one rate per story, U(0, 1) | 42 | 169.47 | 3.904 | 4.189 |
| constant 0.65 | 42 | 170.56 | 3.850 | 5.139 |

Paired difference from the base: constant 0.5 −12.9 (mean of 2 seeds), Flex −7.8 (2 seeds). Flex: 113 vs 85 seq/s, peak 7.03 vs 5.83 GB. Seed noise from 3 pairs: 2.73 points per run, so 7.7 points for two single runs at 2 sigma. Correlation with Gen. PPL over the 12 runs: final training loss +0.82, validation loss +0.54.

- **Conclusion.** At 51,200 stories: constant 0.5 beats the linear schedule on two seeds; FlexAttention gives +33% speed with a probable, weaker quality gain. High rates hurt. Constant 0.35 and 0.5 are not distinguishable (3.4 points at the same seed); I had first written that 0.35 "is the best", which is not shown, and the 5-point threshold was too low. The two gains add partly (−21.5 expected, −17.9 observed, within noise). A slope helps in neither direction. The per-story rate has the second-best validation loss (4.189; only constant 0.35 is lower, 3.886) and nearly the worst text: validation loss on codes does not predict text quality. Corrupting level 0 has no effect on the text (+0.8%). Level-1 loss is the same in every run (6.77-6.79) except the one with level 0 corrupted (7.06), and already near the official 6.64. Limits: hold-up with more training not verified here (the constant-rate advantage could be an early-steps effect); Flex has no attention dropout, and short runs favour less regularisation; most variants have one seed; the training-loss correlation comes from runs where almost only the corruption varies.

## 5. Cycle 3: budget, low corruption rates, learning rate

*2026-11-17*

- **Question.** Do the two advantages hold at 200 and 800 steps? Where is the minimum between rates 0 and 0.35? What does the learning rate do?
- **Setup.** `configs/cicli/2026-11-17_c3a-tenuta800.yaml`, `_c3b-tenuta200`, `_c3c-coda-corruzione`, `_c3d-learning-rate`, `_c3e-conferme`: 17 runs, official tokenizer, 25,600 / 51,200 / 102,400 stories, warmup 5%. Learning-rate sweep on constant 0.5 + Flex, minimum lr = peak / 10. Two follow-up cycles were stopped early: `_c4a-ricetta800` after two runs (its 102,400 stories are not comparable with the 51,200 standard) and `_c4c-800passi-51k` (800 steps × 64) after one.
- **Decision rule.** Thresholds 5% (single runs) and 3.5% (two-seed means), recomputed per budget from the seed pairs before reading. E = (variant − base) / base at the same seed; at 800 steps E ≤ −threshold = holds, −threshold < E < 0 = undecided, E ≥ 0 = vanished. FlexAttention is adopted for speed if quality is not worse beyond the threshold. Corruption: rates within the threshold of the minimum are equivalent, take the centre; a rate lowering entropy by more than 0.1 is excluded. Learning rate: compare with 3e-4 at 5%; if 1e-3 is best, refine at 6e-4 and 1.5e-3 with two seeds; a winner must pass the hold-up test before adoption.
- **Results.** Recomputed noise per run: 0.7% (200 steps, 1 pair), 2.4% (400, 5 pairs), 2.1% (800, 2 pairs); thresholds 6.7% / 4.7% at 400 and 5.9% / 4.2% at 800.

| Gen. PPL, seed 42 / 43 | 200 steps | 400 steps | 800 steps |
|---|---|---|---|
| base linear 0.75→0.25 | 353.78 / 350.15 | 149.42 / 151.92 | 73.52 / 76.11 |
| constant 0.5 | 318.07 / — | 134.72 / 140.92 | 70.02 / 71.67 |
| FlexAttention | 343.81 / — | 142.64 / 143.01 | 74.51 / — |
| E, constant 0.5 | −10.1% | −8.5% | −5.3% |
| E, FlexAttention | −2.8% | −5.2% | +1.3% |

E is from one seed at 200 steps and two at 400; at 800 steps, two seeds for constant 0.5 and one for FlexAttention. Constant rate at 400 steps: 0 → 224.80; 0.10 → 173.86; 0.20 → 144.61; 0.35 → 131.31 / 138.28; 0.50 → 134.72 / 140.92; 0.65 → 170.56. Learning rate at 400 steps: 1e-4 → 358.09 (+172%); 3e-4 → 131.48 / 134.74; 1e-3 → 65.89 (−49.9%); 3e-3 → 76.87 (−41.5%). Follow-ups: best recipe at 102,400 stories 43.35 / 44.33; paper recipe at 800 × 64 83.16 ± 2.60 (149.42 at 400 × 128, same stories and seed).

- **Conclusion.** Constant 0.5 holds at 800 steps, shrinking with budget (−10.1% → −8.5% → −5.3%). FlexAttention is quality-neutral and was adopted for speed, with the reserve on attention dropout. Rates 0.35-0.50 are equivalent (2.2% apart on two-seed means); the centre (about 0.4) was never run and I kept 0.5. Learning rate 1e-3 halves Gen. PPL at 400 steps (one seed here; seed 43 in section 7: 66.62). The refinement required by the rule was not run, and the lr 3e-4 counterpart at 800 steps was cancelled, so the hold-up of 1e-3 is known only against the paper recipe: −56% at 51,200 stories, −41% at 102,400 (two seeds). Short runs favour high learning rates, and a 4× budget range guarantees nothing at the paper's budget, about 1,000× larger. The single 800 × 64 run was not followed up: no conclusion. Some runs were up to 8% slower than usual (cause not found). Nothing here is comparable with the official model (16.58).

## 6. Tokenizer batch size judged on reconstruction

*2026-11-17*

- **Question.** At equal stories seen (614,400), which batch gives the best tokenizer in a short training: 4,096 (as in the paper), 1,024 or 256?
- **Setup.** Runs `mytok-12std_614400_150x4096_s42`, `_600x1024_s42`, `_2400x256_s42`: 12 levels, 4,096 codes per level, level 0 normalised, usage statistics initialised at 1, no code restart, nested dropout 0.5, lr 5e-4 → 5e-5, seed 42, same data order. No generator trained.
- **Decision rule.** None recorded for the batch choice. Entry criterion for a usable tokenizer: at least 99.9% of tokens reconstructed (no padding), stable encoding, bounded codebook norms.
- **Results.**

| | 4,096 × 150 | 1,024 × 600 | 256 × 2,400 | Official |
|---|---|---|---|---|
| Tokens correct (no padding) | 76.05% | 96.77% | 99.27% | 99.98% |
| Perfect stories | 0.0% | 10.9% | 59.6% | 93.3% |
| Codes in use, mean over levels | 6.7% | 29.2% | 37.0% | — |
| Codes in use at level 0 | 0.17% | 0.61% | 0.34% (14) | — |
| Largest codebook norm | 22.3 | 19.6 | 19.8 | ~1.2 × 10¹¹ |
| Codes changing between encoding batch 256 and 16 | 0.024% | 0.033% | 0.017% | 8.6% |

- **Conclusion.** In the same 12 minutes and on the same stories, more updates with a small batch reconstruct far better. Norms are bounded and encoding is stable. Batch 256 is near the official tokenizer on tokens but far on perfect stories and on middle levels (first 8 levels: 73% vs 94%); the entry criterion was not met. Level 0 is almost unused. I chose batch 256, looking only at reconstruction: section 12 repeats the choice on generated text. Open at this point: whether my normalisation causes the level-0 collapse (section 8: not at 600 steps; section 15: without it level 0 uses more codes, but the text is worse).

## 7. My first tokenizer against the official one

*2026-11-17*

- **Question.** With the same generator, recipe, stories and seed, is the text on my tokenizer better or worse than on the official one?
- **Setup.** `configs/cicli/2026-11-17_t1-mytok-vs-tokuff.yaml`; `mytok-12std_614400_2400x256_s42` (about 200 times fewer stories than the official one); standard generator, both recipes, seeds 42 and 43.
- **Decision rule.** None specific; standard thresholds.
- **Results.**

| Recipe | Seed | Official | Mine | Difference |
|---|---|---|---|---|
| Paper | 42 | 149.42 | 474.54 | +218% |
| Paper | 43 | 151.92 | 463.36 | +205% |
| Best | 42 | 65.89 | 219.24 | +233% |
| Best | 43 | 66.62 | 226.31 | +240% |

Entropy: mine 3.833-3.869, official 3.897-3.939 (within the guard). Best against paper recipe: −52% on mine (two-seed means), −56% on the official one.

- **Conclusion.** My 12-minute tokenizer gives text about 3 times worse, with both recipes and seeds; the recipe gain transfers. Unverified: that the disadvantage sits in the middle and fine levels.

## 8. Is my tokenizer fix really a fix

*2026-11-18*

- **Question.** Is the official tokenizer's encoding instability what I thought, do my two changes (level-0 normalisation, usage statistics initialised at 1) prevent it, and do they cause the nearly empty level 0?
- **Setup.** No runs; scripts and raw results in `benchmarks/2026-11-18/`. Without training: official tokenizer and `mytok-12std_614400_2400x256_s42`, 1,024 validation stories. With training: 600 steps × 256, seed 42, same data. V0 = as the original, V1 = normalisation only, V2 = statistics at 1 only, V3 = both.
- **Decision rule.** Written for a planned 2,400-step comparison: if the level-0 norm grows without stopping in V0 and not in V1/V3, the normalisation is justified; if level 0 is empty only with normalisation, try a lighter one. The check ran at 600 steps; the longer V0 run was not done.
- **Results.**

| | Official | Mine |
|---|---|---|
| Codes changing between batch 256 and 16, 32-bit | 8.58% | 0.017% |
| Same, 64-bit | 0% | 0% |
| 32-bit codes different from the exact 64-bit ones | 13.6% | 0.02% |

Rescaling level 0 equivalently (input norm 1.2e11 → 19.6) leaves the instability (8.7%). Along the level-0 upsampling path the official values fall from about 1e9 to 4,150 and are cancelled down to 5.5, with an absolute 32-bit error of 0.75; in mine they stay between 8 and 14, error 0.003.

| After 600 steps | Tokens reconstructed | Level-0 input norm, step 100 → 600 | Distinct codes level 0 / 1 | Unstable codes |
|---|---|---|---|---|
| V0 | 90.9% | 36 → 21 | 3 / 99 | 0.04% |
| V1 | 92.4% | 19.6 fixed | 2 / 177 | 0.05% |
| V2 | 92.1% | 29 → 15 | 4 / 91 | 0.04% |
| V3 | 93.1% | 19.6 fixed | 8 / 164 | 0.04% |

In all four, level 0 uses a single code at step 100 and level 1 one or two. On 4,096 validation stories: official 1,632 / 3,063 distinct codes at levels 0 / 1, my 2,400-step tokenizer 14 / 427.

- **Conclusion.** The official defect is real: 32-bit precision loss on an ill-conditioned level-0 path, not a logic error. My tokenizers do not have it. It is **not** shown that my changes prevent it: in a short training the unmodified variant is healthy too, and the problem may appear only in long trainings. The normalisation is a protection by construction, not a demonstrated fix. I had named the 1e11 norm as the cause; it is the symptom. The real problem of my tokenizer is code collapse at the coarsest levels, with or without the changes. Effect of the defect on text: section 13.

## 9. Remedies against code collapse

*2026-11-18*

- **Question.** Do codebooks initialised from data (R1) and an early restart of unused codes (R2) improve the generated text?
- **Setup.** Quick test: 600 steps, on top of V3 (`benchmarks/2026-11-18/`). Full test, `configs/cicli/2026-11-18_f1d-rimedi.yaml`: `mytok-12std_614400_2400x256_s42` (random init), `mytok-12std-dati_…` (R1: k-means per level on 8,192 stories), `mytok-12std-dati-rv_…` (R12: also restart of codes with usage statistic below 0.1, up to step 1,200); standard generator, both recipes, generator seeds 42 and 43.
- **Decision rule.** Carry forward the variant best in Gen. PPL beyond the threshold (6.7% single runs, 4.7% two-seed means); if tied, the more stable one.
- **Results.** Quick test, distinct level-0 codes after 600 steps: V3 8, R1 10, R2 70, R12 55; tokens reconstructed 93.1%, 95.1%, 94.1%, 95.0%.

| Tokenizer | Paper recipe, s42 / s43 / mean | Best recipe, s42 / s43 / mean | Ratio to official |
|---|---|---|---|
| Random init | 474.54 / 463.36 / 468.95 | 219.24 / 226.31 / 222.78 | 3.11× / 3.36× |
| R1 | 241.51 / 232.26 / 236.88 | 116.97 / 113.78 / 115.38 | 1.57× / 1.74× |
| R12 | 225.86 / 213.34 / 219.60 | 106.54 / 109.80 / 108.17 | 1.46× / 1.63× |
| Official | 149.42 / 151.92 / 150.67 | 65.89 / 66.62 / 66.26 | 1× |

| Tokenizer | Tokens / perfect stories | Codes in use, mean / level 0 | Code entropy (nats) | Learned by the generator, paper / best |
|---|---|---|---|---|
| Random init | 99.27% / 59.6% | 37.0% / 14 | 6.07 | 12% / 24% |
| R1 | 99.03% / 51.8% | 49.0% / 17 | 7.17 | 32% / 46% |
| R12 | 98.57% / 39.9% | 49.4% / 22 | 7.23 | 33% / 46% |
| Official | 99.62% / 93.3% | — / 1,632 | 7.48 | 41% / 56% |

"Learned" = what the generator learns beyond code frequencies (`evaluation/entropia_codici.py`).

- **Conclusion.** R1 halves Gen. PPL: −49.5% (paper) and −48.2% (best) on two-seed means. R12 against R1: −7.3% and −6.2%, same sign in all four paired comparisons, above the 4.7% threshold. Reconstruction got worse while the text improved: reconstruction does not rank tokenizers. The gain comes from the middle and fine levels (levels 4-10: from about 1,300-1,700 to 1,850-3,000 codes in use; level 7: from 2,855 to 3,241-3,312), not from level 0. No remedy prevents the initial collapse (R1 starts from 456 distinct level-0 codes and has one left at step 100). Instability stays at 0.02%. I carried R12 forward. Limit: only the generator seed was varied; each tokenizer was trained once. For R1 this is unlikely to matter; for the restart (−6/−7%) it could, so "the restart helps" needs a second tokenizer seed. Unverified: the nearly empty levels 0-3 as a cause of the remaining 1.5-1.6× gap.

Early-only dead-code restarts were also proposed in [upstream PR #27](https://github.com/HazyResearch/ncm/pull/27); [issue #16](https://github.com/HazyResearch/ncm/issues/16) discusses restart behaviour and reconstruction versus generation. The contribution here is the measured downstream effect at the tested budgets.

## 10. Number of levels

*2026-11-17 to 2026-11-18*

- **Question.** With the same tokenizer and generator training, which number of levels gives the best text?
- **Setup.** Tokenizers: 614,400 stories, 2,400 × 256, seed 42; 12 levels with official spacing, the others linear. Standard generator, seed 42, both recipes (micro-batch reduced above 12 levels; control at 12 levels: 470.38 vs 474.54). (a) Random-init codebooks: `configs/cicli/2026-11-17_t3-livelli-12-26-40.yaml`, `2026-11-18_t4-livelli-8.yaml`, `2026-11-18_t5-livelli-16.yaml`. (b) R12, after section 9: `2026-11-18_f1e-livelli-rimedi.yaml`.
- **Decision rule.** (a) With more levels a tokenizer is less mature at equal training, so a win of 26/40 would be a strong signal and a loss is not conclusive; threshold 6.7%. (b) For 26 levels, best recipe: the help of the remedies is "progressive" if the gain exceeds the 66% of 16 levels (Gen. PPL below about 153).
- **Results.**

| Levels | Codes per story | (a) tokens / perfect stories / codes in use | (a) Gen. PPL paper / best | (b) Gen. PPL paper / best | (b) entropy paper / best |
|---|---|---|---|---|---|
| 8 | 516 | 99.27% / 59.8% / 44.8% | 647.82 / 281.73 | 335.06 / 170.65 | 3.914 / 3.937 |
| 12 | 641 | 99.27% / 59.6% / 37.0% | 474.54 / 219.24 | 225.86 / 106.54 | 3.859 / 3.873 |
| 16 | 1,032 | 99.43% / 67.1% / 33.0% | 644.75 / 342.72 | 307.24 / 115.54 | 3.777 / 3.841 |
| 26 | 1,677 | 99.55% / 71.9% / 25.9% | 794.77 / 448.91 | 418.63 / 144.73 | 3.710 / 3.780 |
| 40 | 2,580 | 99.63% / 75.4% / 20.9% | 1,401.24 / 896.88 | — | — |

Gain from the remedies (paper / best): −48% / −39%, −52% / −51%, −52% / −66%, −47% / −68% at 8, 12, 16, 26 levels. 12 levels with R12, seed 43: 213.34 / 109.80. Level 0 stays nearly empty with R12 (30, 22, 20 codes at 8, 12, 16 levels).

- **Conclusion.** (a) With random init 12 is best and the text worsens steadily above it while reconstruction improves; this was dominated by code collapse, and the 26- and 40-level results are not valid for judging those level counts. (b) With R12 the order changes: under the best recipe 16 is close to 12 (+5-8% against its two seeds, around the threshold) and 8 is clearly last (+60%); under the paper recipe 12 is clearly ahead. I had first written "12 wins with both recipes", which is not shown for the best recipe. The gain from the remedies grows up to 16 levels, then flattens (rule met narrowly). Entropy falls with the number of levels: 26 levels with the paper recipe is 0.15 below 12, beyond the guard, so its Gen. PPL is optimistic. The generator is 1.6 times slower at 16 levels and 2.5 times at 26. I kept 12 at this budget. Limits: one seed for 8, 16, 26, 40; level count and spacing change together; the order may change with longer training (in the paper's ablation, as reported there, more levels do better).

## 11. Tokenizer training doubled

*2026-11-18 to 2026-11-19*

- **Question.** How much training does my R12 tokenizer need before the text stops improving, and how close does it get to the official one?
- **Setup.** `configs/cicli/2026-11-18_f2-calibrazione.yaml`: `mytok-12std-dati-rv_614400_2400x256_s42` and `mytok-12std-dati-rv_1228800_4800x256_s42`, each trained from scratch; best recipe, seed 42, generator at 51,200 and then 102,400 stories. One 102,400-story run was interrupted at step 490 without a checkpoint and re-run from scratch.
- **Decision rule.** Keep doubling while the gain exceeds 6.7%. When it flattens, compare the last two with the generator at 102,400 stories: the longer tokenizer pulls ahead if it is lower by more than 6.7%.
- **Results.**

| Tokenizer | Tokens / perfect stories | Codes in use, mean / level 0 | Gen. PPL at 51,200 (entropy) | Gen. PPL at 102,400 (entropy) |
|---|---|---|---|---|
| Mine, 2,400 steps | 98.57% / 39.9% | 49.4% / 22 | 106.54 (3.873); seed 43: 109.80 | 65.68 (3.918) |
| Mine, 4,800 steps | 99.27% / 59.9% | 56.5% / 91 | 103.61 (3.910) | 67.59 (3.939) |
| Official | 99.62% / 93.3% | — / 1,632 | 65.89 / 66.62 (3.939) | 43.35 / 44.33 (3.951 / 3.954) |

- **Conclusion.** Doubling gives −2.8% (−4.2% against the two-seed mean), below the threshold, and +2.9% with the longer generator: the rule is not met and the doubling stopped. The tokenizer's own numbers improve a lot and the text entropy rises, which Gen. PPL does not show. Distance from the official tokenizer: 1.6× at 51,200 stories, 1.5× at 102,400. One tokenizer seed. One doubling says little about much longer training: section 15 goes to 38,400 steps and finds a gain.

## 12. Tokenizer batch size judged on generated text

*2026-11-19*

- **Question.** Does the gap depend on the tokenizer batch (mine 256, official 4,096)? Do the coarse levels fill up with a larger batch?
- **Setup.** `configs/cicli/2026-11-19_f2b-batch.yaml`: R12 tokenizers at the same 1,228,800 stories, 256 × 4,800 (section 11), `mytok-12std-dati-rv_1228800_1200x1024_s42`, `mytok-12std-dati-rv_1228800_300x4096_s42`; restart on the first 307,200 stories, seed 42. Standard generator, best recipe, seed 42.
- **Decision rule.** A batch wins if it is below 103.61 by more than 6.7% (below about 96.7).
- **Results.**

| Batch × steps | Tokens / perfect stories | Gen. PPL | Entropy | Live codes at levels 0 / 1 / 2 / 3 | at levels 8 / 9 / 10 |
|---|---|---|---|---|---|
| 256 × 4,800 | 99.27% / 59.9% | 103.61 | 3.910 | 86 / 339 / 868 / 1,336 | 2,820 / 2,919 / 2,874 |
| 1,024 × 1,200 | 98.18% / 30.9% | 98.29 (−5.1%) | 3.909 | 49 / 286 / 737 / 1,341 | 3,314 / 3,539 / 3,542 |
| 4,096 × 300 | 93.86% / 1.2% | 100.36 (−3.1%) | 3.863 | 81 / 303 / 342 / 1,379 | 3,303 / 3,478 / 3,413 |
| Official (4,096 × 30,000) | 99.62% / 93.3% | 65.89 | 3.939 | 2,377 / 3,101 / 3,947 / 4,055 | 4,074 / 4,069 / 4,085 |

- **Conclusion.** No batch passes the threshold: the three give equivalent text. "Coarse levels are empty because of batch 256" is refuted: with batch 4,096 levels 0-2 stay nearly empty while the fine levels fill up. Reconstruction again does not rank tokenizers (1.2% against 59.9% perfect stories, same text). Batch 4,096 lowers the entropy. The choice of 256 was neither wrong nor justified by the text. One seed. Left by exclusion: the 100× longer training of the official tokenizer, the nearly empty coarse levels, the official precision defect (section 13).

## 13. Test C: official tokenizer with exact codes

*2026-11-19*

- **Question.** The official tokenizer computes 13.3-13.6% of its codes differently from the exact ones. Does that help or hurt generation, and is it part of its advantage over mine?
- **Setup.** `configs/cicli/2026-11-19_c-ufficiale-64bit.yaml`, runs `gen-tokuff64_51200_400x128_c50_lr1e-3_flex_s42` / `_s43`: same tokenizer and standard generator, best recipe; only the codes change (32-bit pre-computed, or 64-bit on the fly); generator input and decoding stay at 32 bits. Reconstruction check: `benchmarks/2026-11-19/ricostruzione_fp64.py`.
- **Decision rule.** A difference above 4.7% on the two-seed mean = the defect matters; below = no effect on the text.
- **Results.**

| Codes | Seed 42 | Seed 43 | Mean | Entropy s42 / s43 |
|---|---|---|---|---|
| 32-bit | 65.89 | 66.62 | 66.26 | 3.939 / 3.927 |
| 64-bit (exact) | 66.42 (+0.8%) | 66.63 (+0.0%) | 66.53 (+0.4%) | 3.950 / 3.925 |

Reconstruction is the same in all combinations of 32/64-bit codes and decoding: 99.616-99.619% of tokens, 93.38-93.60% perfect stories. Codes differing between 64 and 32 bits: 13.3%. Distinct codes at levels 0-3: 1,632 / 3,059 / 3,874 / 3,956 (32-bit), 1,633 / 3,077 / 3,872 / 3,956 (64-bit).

- **Conclusion.** The defect is real but has no measurable effect on reconstruction or on text (two seeds): the "wrong" codes are near-equivalent choices. The official advantage does not come from it; with exact codes its first levels are just as full. It is a numerical curiosity (encoding that depends on the batch size), not a cause of worse quality.

## 14. Test A: generator four times longer

*2026-11-19 to 2026-11-20*

- **Question.** Does the 1.5-1.6× gap between my R12 tokenizer and the official one come from the short generator training, or from my codes and their decoding?
- **Setup.** `configs/cicli/2026-11-19_a-generatore-lungo.yaml` (seed 43 via `2026-11-20_conferme-catena.cmd`). Frozen tokenizers: official (32-bit pre-computed codes) and `mytok-12std-dati-rv_614400_2400x256_s42` (it already had the points at 51,200 and 102,400 stories). Best recipe, 409,600 stories = 3,200 × 128. Runs `gen-tokuff_409600_3200x128_c50_lr1e-3_flex_s42` / `_s43` and `gen-mytok12std-dati-rv-2400x256_409600_…`. Interrupted by a power cut; resumed from the last checkpoint.
- **Decision rule.** Ratio mine / official at 409,600 stories, with the entropy guard: ≤ 1.25 = the short generator was a large part of the gap; ≥ 1.40 = they stay apart, a stronger indication that my codes or their decoding are a limit with this recipe (not a proof that level 0 is the cause); in between = not decisive. Confirmation with seed 43: ratio ≥ 1.40 in that seed and in the mean.
- **Results.**

| Stories seen | Official | Mine | Ratio | Entropy off. / mine | Val loss off. / mine |
|---|---|---|---|---|---|
| 51,200, seed 42 | 65.89 | 106.54 | 1.62 | | |
| 102,400, seed 42 | 43.35 | 65.68 | 1.52 | | |
| 409,600, seed 42 | 26.04 ± 0.48 | 42.76 ± 1.05 | 1.64 | 3.952 / 3.965 | 1.920 / 1.946 |
| 409,600, seed 43 | 25.71 | 41.24 | 1.60 | 3.946 / 3.961 | 1.915 / 1.950 |
| 409,600, mean | 25.88 | 42.00 | 1.62 | | |

Validation loss per level at 409,600 stories (seed 42), official / mine: L1 6.52 / 5.13 · L5 2.73 / 3.05 · L11 0.39 / 0.50.

- **Conclusion.** They stay apart, on two seeds. The ratios range from about 1.50 to 1.64 across the reported budgets and seed comparisons; the 102,400-story point is lower, but the gap persists at 409,600. Increasing the generator budget did not eliminate the gap for this frozen 2,400-step tokenizer and recipe; other generator settings remain untested. This does not prove that level 0 is the cause. Validation losses are numerically close (the +1.4% comparison is for seed 42), but their target code distributions differ. These losses do not establish equal prediction quality or isolate a causal defect in the codes. In practice my tokenizer at 409,600 stories equals the official one at 102,400. For this pair the ratio at 51,200 stories already matched the one at 409,600; this holds for a large gap, not for tokenizers that are close to each other (section 11). With two-seed means where available the ratios are 1.63 at 51,200 (108.17 / 66.26) and 1.50 at 102,400 (65.68, one seed / 43.84).

## 15. Test B: two extremes on the tokenizer

*2026-11-19 to 2026-11-20*

- **Question.** Does my level-0 normalisation hinder learning? Does the tokenizer need much more training?
- **Setup.** `configs/cicli/2026-11-19_b-catena.cmd` (`_b1-norma-l0.yaml`, `_b2-training-lungo.yaml`). Base for both pairs: `mytok-12std-dati-rv_1228800_4800x256_s42` (103.61). Pair 1 changes only the normalisation (`mytok-12std-dati-rv-nonorm_1228800_4800x256_s42`). Pair 2 changes only the length (`mytok-12std-dati-rv_9830400_38400x256_s42`: same peak and final lr, cosine over the whole run, warmup 5%, restart up to step 1,200 in both). Its snapshots at steps 9,600 and 19,200 are mid-run points with the learning rate still high, not full trainings of that length. Judge: standard generator, best recipe, seed 42. Confirmation: `…_38400x256_s43` (only the training seed changes). One generator run was interrupted by a power cut and re-run from scratch.
- **Decision rule.** Against 103.61, entropy not more than 0.1 below 3.910: the new arm wins below 96.7, loses above 110.6, no measurable effect in between; it closes the gap at 82.4 or less (1.25× the official 65.89). A win between 82.4 and 96.7 needs a second tokenizer seed. In pair 2 only the end point decides. Confirmation: seed 43 below 96.7 and the two-seed mean below the base by more than 4.7%.
- **Results.**

| Tokenizer | Gen. PPL | Entropy | vs base | Ratio to official | Perfect stories | Codes in use, levels 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| 4,800 steps (base) | 103.61 ± 2.25 | 3.910 | — | 1.57 | 59.9% | 91 / 551 / 1,103 / 1,588 |
| 4,800 steps, no normalisation | 115.28 ± 2.34 | 3.913 | +11.3% | | 59.8% | 305 / 562 / 1,373 / 1,953 |
| snapshot at step 9,600 | 95.15 ± 1.76 | 3.902 | −8.2% | 1.44 | 71.9% | 186 / 943 / 1,707 / 2,259 |
| snapshot at step 19,200 | 90.27 ± 2.73 | 3.925 | −12.9% | 1.37 | 80.6% | 361 / 1,254 / 2,092 / 2,671 |
| 38,400 steps, seed 42 | 91.46 ± 1.66 | 3.934 | −11.7% | 1.39 | 82.7% | 500 / 1,447 / 2,389 / 2,942 |
| 38,400 steps, seed 43 | 93.23 ± 2.46 | 3.941 | | | 82.0% | 501 / 1,442 / 2,369 / 2,923 |
| 38,400 steps, mean | 92.35 | | −10.9% | 1.39 (against the two-seed official mean, 66.26) | | |

Largest code norm: 20 with normalisation, 32,986 without (75 at step 1,600, 10,849 at step 3,200). Generator validation loss: 3.809 (base), 4.020 (no normalisation), 3.954 → 4.033 → 4.015 along the long run; it rises while the text improves.

- **Conclusion.** Removing the normalisation loses (one tokenizer seed): the code scale drifts and level 0 uses 3.4 times more codes, but the text is worse. Longer training wins, confirmed on two tokenizer seeds, but does not close the gap: the ratio goes from 1.57 to 1.39, about a third of the distance (12.2 points of 37.7). I measure no further gain after the 19,200-step snapshot (90.27 and 91.46 are equal within noise); this rests on intermediate snapshots of one run (seed 42) with the learning rate still decaying, so it is not proof of a plateau. The two 38,400-step tokenizers are almost identical internally; the base has one tokenizer seed. Code counts do not decide pairwise: level-0 codes rise from 361 to 500 with no change in the text, and the no-normalisation arm has more codes and worse text. Across all my tokenizers, however, overall code usage does rank them, while reconstruction does not (section 16). A gap of about 1.4× remains: not closed by training up to 38,400 steps (the official tokenizer saw 12.5 times more stories, so tokenizer training remains a possible explanation) while the tested changes to batch size, normalisation and numerical precision did not eliminate the gap. The generator-budget sweep used the shorter 2,400-step tokenizer and therefore does not directly explain this 1.4× residual gap. Unverified candidates: loss weights, codebook update, large batch together with long training.

## 16. Where are my codes fragile

*2026-11-20*

- **Question.** Generated-text perplexity remains about 1.6 times higher on the 2,400-step tokenizer despite numerically close validation losses (section 14); losses across different code distributions do not establish equal prediction quality. Does a wrong code damage the decoded text more in my tokenizer, and at which level?
- **Setup.** No training; scripts and raw results in `benchmarks/2026-11-20/`. On 4,096 validation stories: encode, corrupt some codes, decode, compare. *Swap* = the code of another random story at the same level and position, on 25% of positions, one level at a time (also on all levels at 5% and 10%). Main measure: damage per changed code = extra wrong tokens over the clean reconstruction, divided by the codes actually changed. Tokenizers: official (32-bit), mine at 4,800 and 38,400 steps; for the correlation, the 11 twelve-level tokenizers with a standard generator (best recipe, seed 42).
- **Decision rule.** Damage per changed code, mine (4,800 steps) against official, per level: ≥ 1.5× at some level = my codes are fragile there, a concrete candidate; within 1.25× at all levels = fragility does not explain the gap; in between = not decisive. Two additions, each rule written before its data: the generator ignores level 0 if its loss rises by less than 0.02 when level 0 comes from another story; rank correlation of fragility with Gen. PPL: ρ ≥ 0.8 = follows the text, ρ ≤ 0.5 = does not, in between = weak, read also on my 10 tokenizers alone.
- **Results.** Extra wrong tokens per changed code (128 tokens per story):

| Level (codes per story) | Official | Mine, 4,800 steps | Mine, 38,400 steps |
|---|---|---|---|
| 0 (1) | 0.078 | 5.09 (65×) | 7.20 (93×) |
| 1 (8) | 0.000 | 0.004 | 0.001 |
| 2-5 | 0.000-0.003 | 0.000-0.004 | 0.001-0.005 |
| 6 (52), 7 (64) | 0.003, 0.009 | 0.008, 0.019 | 0.006, 0.015 |
| 8-10 | 0.001-0.002 | 0.003-0.008 | 0.002-0.004 |
| 11 (128) | 0.017 | 0.039 | 0.020 |

Generator loss increase with level 0 taken from another story: official tokenizer +0.34 (generator at 51,200 stories) and +0.53 (409,600); mine +0.25 (4,800 steps), +0.25 (38,400 steps), +0.29 (2,400 steps, generator at 409,600).

| Tokenizer | Gen. PPL | Swap 5% | Swap 10% | 10% without level 0 | Level 0 only |
|---|---|---|---|---|---|
| Official | 65.89 | 0.27% | 1.09% | 1.04% | 0.07% |
| Random init, 2,400 steps | 219.24 | 0.99% | 2.83% | 2.52% | 2.64% |
| R1, 2,400 steps | 116.97 | 0.66% | 1.97% | 1.76% | 1.76% |
| R12, 2,400 steps | 106.54 | 0.71% | 2.03% | 1.68% | 3.02% |
| R12, 4,800 steps | 103.61 | 0.67% | 1.92% | 1.44% | 3.98% |
| R12, 4,800 steps, no normalisation | 115.28 | 0.63% | 1.85% | 1.50% | 2.96% |
| R12, batch 1,024 | 98.29 | 0.64% | 1.91% | 1.74% | 1.15% |
| R12, batch 4,096 | 100.36 | 0.91% | 2.46% | 2.44% | 0.05% |
| R12, snapshot at step 9,600 | 95.15 | 0.66% | 1.97% | 1.39% | 4.78% |
| R12, snapshot at step 19,200 | 90.27 | 0.62% | 1.81% | 1.16% | 5.35% |
| R12, 38,400 steps | 91.46 | 0.62% | 1.80% | 1.14% | 5.53% |

Spearman correlation with Gen. PPL (all 11 / my 10): clean reconstruction error +0.59 / +0.46; swap 5% +0.68 / +0.58; swap 10% +0.74 / +0.65; 10% without level 0 +0.81 / +0.75; level 0 only −0.21 / −0.55. Later analysis of the same 10 tokenizers of mine (from each `export/export.json`): mean fraction of codes in use over all levels −0.90 (−0.83 without the two snapshots); level-0 codes in use alone −0.76; perfect stories −0.48.

- **Conclusion.** By the first rule my codes exceed 1.5× at level 0 and at every level from 3 to 11 except level 4 (1.3×), but in absolute terms only level 0 matters.
  1. Aggregate sensitivity to the tested random errors at levels 1-11 is close to the official tokenizer (1.14% vs 1.04%, 1.10×) while text perplexity remains 1.39 times higher. This does not rule out per-level effects or sensitivity to the generator's actual errors. Their fragility partly follows quality among my tokenizers (ρ 0.75, "weak" by the rule; 0.81 with the official one) and falls with long training: it is part of what long training improves, not what is left.
  2. Level 0 plays a different role. In the official tokenizer its code hardly changes the decoded text (0.07% of tokens) while the generator relies on it heavily: a "plan" rather than content. In my 4,800- and 38,400-step tokenizers it carries content (4.0-5.5% of tokens; 1.8-5.5% across my batch-256 tokenizers) and the generator relies on it less. No generator ignores it.
  3. That difference is a candidate, not a demonstrated cause, and there is evidence against it. Among my tokenizers more content in level 0 does not go with worse text (ρ −0.55 over the 10; −0.14 without the two snapshots of the long run): long training raises both content at level 0 and quality. And the batch-4,096 tokenizer has an inert level 0 (0.05%), like the official one, yet its text (100.36) is only as good as that of the 4,800-step tokenizer (103.61, a difference within noise), far from 65.89.
  4. Reconstruction does not rank tokenizers: the batch-4,096 one gets 6.2% of tokens wrong and generates as well as the 4,800-step one (0.7%). Overall code usage does rank them, with pairwise exceptions such as the arm without level-0 normalisation (more codes in use, worse text). This is a rank correlation over 10 tokenizers, each trained once, two of them snapshots of one run.

  Limits: no training; one seed per tokenizer; random errors are not a generator's real errors, so this measures the sensitivity of decoding. Not run: a tokenizer whose level 0 does not contribute to decoding, judged with a generator.
