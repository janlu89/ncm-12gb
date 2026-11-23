# Changelog

All notable changes to this research note, its results and its code. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), adapted to research: each version lists findings, negative results, code, data and corrections, and says whether earlier conclusions changed. Every number is traceable to `results/runs/` and [EXPERIMENTS.md](EXPERIMENTS.md).

## 0.2 — 2026-11-23

DOI: [10.5281/zenodo.24478988](https://doi.org/10.5281/zenodo.24478988) (all versions: [10.5281/zenodo.24408540](https://doi.org/10.5281/zenodo.24408540)).

### Findings added
- **A looped generator can be grown in depth during training without a measurable loss of quality (no difference beyond the 4.7% two-seed threshold), at equal or lower compute counted as applied layers averaged over steps** ([README section 4](README.md#4-a-looped-generator-can-be-grown-in-depth--new-in-v02); EXPERIMENTS sections 19-22). Generator: 4 layers + a block of 4 layers applied R times with shared weights + 4 layers. Growth: training starts with one repetition; later repetitions switch on at fixed steps as exact copies of the trained block; one cosine learning-rate curve. Official tokenizer, 409,600 stories, means of two seeds: 12 layers standard 25.88; 16 applied layers from scratch 25.10; grown 12 → 16 24.52 (−2.3%, 12.5% fewer applied layers, measured training time 3–13% lower); grown 12 → 16 → 20, ending at 20 applied layers, 24.17 (−3.7%, same average number of applied layers). Both grown models tie with 16 from scratch under the 4.7% rule for two-seed means, fixed before the second-seed runs (after the seed-42 results); none is worse. No 20-layer model was trained from scratch at this budget.
- At 51,200 stories a looped 4 + 2×2 + 4 generator (12 applied layers, 74.4M parameters, −16%) ties with the standard 12-layer generator (64.18 vs 65.89, one seed).
- Not shown: that growth improves quality. The grown model is below the from-scratch model in all four long comparisons, with seeds agreeing within 0.2-0.7; both two-seed comparisons with 16 from scratch are ties, and each single-seed comparison is within the 6.7% single-run threshold. The only comparison beyond the threshold is grown 12 → 16 → 20 against the 12-layer generator (−6.6%), which uses more compute and was added to the rule after seed 42 had shown −5.2%. At 51,200 stories growth varies strongly between seeds (60.91 / 68.25).
- Confounds stated in README section 4: initialisation scaled by the full number of applied layers; 12-layer baselines on Windows 10, looped runs on Linux; shared data order and shared references; thresholds derived at 51,200 stories.

### Negative results
- Dead-code restart threshold relative to each level (EXPERIMENTS section 17): Gen. PPL 114.90 against 106.54 (one seed). A check without training confirms that the fixed threshold (0.1) lies above the uniform per-step share of a level-0 code (0.0625), so restart is expected to replace all but the most used level-0 codes; making it relative does not fill level 0.
- Codebook EMA decay 0.9968, matching the official tokenizer's memory in stories (section 18): 157.80 against 106.54 (one seed).

### Code and tooling
- `generator/stage2.py`: option `cicli` (looped generator with tied weights, separate blocks and key-value caches); `crescita_al_passo` / `crescita_ai_passi` (growth).
- `generator/train_generator.py`: growth state follows the trainer step, also on resume, and is re-applied after training so that a resume at the final step exports the right state; `--estendi` to extend a run. `generator/stage2.py` rejects growth steps beyond `train.steps`; `scripts/discesa.py` requires a source run without decay.
- `common/trainer.py`: optional warmup-stable-decay schedule (`train.schedule=wsd`), off by default. `scripts/discesa.py` (descent branch) and `scripts/valuta_checkpoint.py` (evaluate a paused checkpoint). These tools were checked without GPU training and are **not yet used** in any reported experiment.
- `evaluation/gen_ppl_run.py`: builds looped generators with the right number of layers.
- `tokenizer/quantizer.py`, `tokenizer/stage1.py`: option `riavvio_soglia_relativa` (off by default). `evaluation/soglia_riavvio_per_livello.py`: reads the restart statistics of a checkpoint.
- `common/nomi.py`: name suffixes `cicli…`, `-cresce…`, `wsd`, `rel`, `-dNNNN`.
- Linux: `scripts/avvia_detached.sh` (detached `systemd-run --user` launcher), `scripts/run_attivi.sh`; `common/rundirs.py` creates runs in `runs/` on Linux.
- `tests/test_tokenizer.py` switches off cuDNN TF32, so the accumulation check passes on Linux as on Windows; training keeps the default.
- `paper/build_paper.py`: new section and figure, `PAPER_FONTDIR` for the fonts on other systems.

### Data and runs
- 17 new run folders (15 generators, 2 tokenizers) and 7 cycle tables in `results/runs/`; `results/RUNS.md` and `runs_index.csv` regenerated (20 tokenizer runs, 92 evaluated generator runs, 5 interrupted or cancelled).
- Seven new cycle definitions in `configs/cicli/`.
- Two long runs were interrupted by unplanned shutdowns and resumed from their last checkpoints (grown 12 → 16 seed 42: reached step about 430, resumed from 267, before its switch step; 16 from scratch seed 43: reached about 2,260, resumed from 2,141); the repeated steps were recomputed.
- Six looped generators added to Hugging Face ([janlu89/ncm-12gb](https://huggingface.co/janlu89/ncm-12gb)).

### Upstream checkpoints
- All results in v0.1 and v0.2 use the authors' TinyStories release of 14–16 October 2026 (`hazyresearch/ncm-checkpoints`, revision `d342ce8aaea58830eef3b8ad473664673e41f692`). The authors replaced `tokenizer.pt`, `generator.pt` and `config.yaml` on 23 November 2026 (and report gen-PPL 15.40 for the first replacement in its commit message); the new release was not evaluated here. README, REPRODUCING.md and `third_party/README.md` now pin the revision and give the file hashes.

### Corrections
- None to v0.1 results.
- `evaluation/gen_ppl_run.py --official` failed after the looped-generator change (it read the configuration of a run that does not exist); fixed before release. A regression introduced while preparing v0.2; no published number affected.
- Wording: the paper's training budgets (122.88M and 102.4M stories) are attributed to the paper, not to the released checkpoints; training code is described as not released (upstream issue #10 open on 23 November 2026).

### Conclusions
- **Unchanged:** all v0.1 results and conclusions, including the remaining 1.39× tokenizer gap, which the two new tokenizer tests did not narrow.
- **Added:** the growth finding above, stated as "no measurable loss of quality", not as an improvement.

## 0.1 — 2026-11-20 (UTC), archived on Zenodo 2026-11-21

Initial release ([tag v0.1](https://github.com/janlu89/ncm-12gb/tree/v0.1); DOI [10.5281/zenodo.24408541](https://doi.org/10.5281/zenodo.24408541)); published on 21 November 2026 with weights on Hugging Face.

### Findings
- Initialising codebooks from data (k-means) lowers Gen. PPL from 222.78 to 115.38 (−48.2%; two generator seeds, one tokenizer per configuration). Early-only dead-code restart adds −6.2% (108.17).
- A retuned short-budget generator recipe lowers Gen. PPL from 150.67 to 66.26 on the official tokenizer (−56.0%, two seeds), mostly from raising the learning rate to 1e-3; −41.4% at 102,400 stories.
- Longer tokenizer training (38,400 steps, two tokenizer seeds) gives 92.35: a 1.39× gap to the official tokenizer (66.26) remains.
- With eight times more generator training (409,600 stories) both tokenizers improve, but the ratio stays at 1.62× for the 2,400-step tokenizer.

### Negative results
- Tokenizer batch size, removal of level-0 normalisation and fp64 encoding of the official tokenizer did not eliminate the gap.

### Diagnostics
- Mean code usage ranks our tokenizers better than reconstruction (Spearman −0.90 vs −0.48, post hoc, ten tokenizers).
- Level 0 is nearly inert in the official decoder (0.07% of tokens change when it is swapped) but carries content in ours (4.0-5.5%); a batch-4,096 tokenizer with an inert level 0 still scores 100.36. Candidate, not cause.

### Code, data and runs
- Training code for tokenizer and generator, pause and resume, cycle runner, evaluation with the authors' scripts; 16 experiments; 18 tokenizer runs, 77 evaluated generator runs, 5 interrupted or cancelled; four checkpoints on Hugging Face.
