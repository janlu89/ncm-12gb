# Reproducing the study

Gianluca Meola · [gianluca.meola89@gmail.com](mailto:gianluca.meola89@gmail.com)

[Research overview](README.md) · [Paper](paper/ncm-12gb.pdf) · [Experiment log](EXPERIMENTS.md)

## Method

- **Metric.** 1,280 stories are generated unconditionally (256 for each of the sampling seeds 42, 0, 1, 2, 3; temperature 1, no top-k or top-p truncation; the level-0 code is drawn from its empirical frequency) and scored by GPT-2 large with the authors' `eval_shard.py` and `eval_merge.py`. The ± values in `results/RUNS.md` are the spread over the five sampling seeds of one model.
- **Decision rules before data.** For each experiment I fixed how to read the result before running it; the rules are in [EXPERIMENTS.md](EXPERIMENTS.md). The first two cycles used a fixed threshold of 5 points of Gen. PPL (about 3.3% of the baseline); later ones used 6.7% between single runs and 4.7% between means of two seeds, derived from the seed noise measured at 51,200 stories (2.4% per run over 5 seed pairs), always with entropy no more than 0.1 below the reference.
- **Same validation set.** All losses are computed on the first 4,096 validation stories.
- **Pause and resume.** Training writes full checkpoints atomically and can be resumed. A short test compares a resumed run with an uninterrupted one in deterministic mode; the reported runs use the faster non-deterministic mode. Runs cut short by power cuts were either resumed from their last checkpoint (one long run in v0.1, two in v0.2) or repeated, and the interrupted folders are kept in `results/runs/`. A grown generator resumes in the right state: which repetitions are active follows the trainer's step.
- **Hardware and cost.** RTX 3080 Ti (12 GB), Intel i5-6600K, Windows 10 for the v0.1 runs and Linux for the v0.2 runs, PyTorch 2.14.1 (`torch.compile` for the generator). With the lr 1e-3 recipe on a 12-level tokenizer, a 51,200-story generator run takes 10 to 12 minutes and peaks at 7.0 GiB allocated by PyTorch; 409,600 stories take 62 to 70 minutes. A 38,400-step tokenizer takes about 3 h 15 min. Generators on tokenizers with more levels are slower (about an hour with 40 levels). Looped generators at 409,600 stories take 74 to 84 minutes and peak at 7.0 GiB (16 applied layers) or 8.1 GiB (20).

## Repository layout

| Path | Content |
|---|---|
| `tokenizer/`, `generator/`, `common/` | training code: stage 1 (VQ-VAE), stage 2 (generator), shared loop with pause and resume |
| `configs/` | base configs and every experiment cycle (`configs/cicli/`) |
| `data/` | scripts that pre-compute and verify the official tokenizer's codes |
| `evaluation/` | Gen. PPL with the authors' scripts, code-entropy analysis, reference measurements |
| `benchmarks/` | measurements without training (precision, fragility of codes, speed) |
| `figures/` | the figures above and the scripts that make them |
| `tests/` | tokenizer equivalence with upstream, resume |
| `scripts/` | cycle runner (`ciclo.py`), analysis helpers, detached launchers for Windows (`avvia_detached.ps1`) and Linux (`avvia_detached.sh`, `run_attivi.sh`), WSD descent branch (`discesa.py`) and checkpoint evaluation (`valuta_checkpoint.py`) |
| `environment/` | exact package versions |
| `third_party/` | how to obtain the upstream code, and its licence |
| `results/` | every run: settings, logs, Gen. PPL, generated samples; `runs_index.csv`; the run-by-run report |

Code comments, option names and log fields are in Italian (for example `init_dati` = initialise codebooks from data, `riavvio_fino_al_passo` = restart dead codes up to this step, `sequenze_perfette` = perfectly reconstructed stories). Some comments point to planning documents that are not part of this release.

## Reproducing

Developed on Windows; the v0.2 runs were trained on Linux with the same code. The Python code is not Windows-specific; the `.cmd` and `.ps1` launchers (which assume a virtual environment in `.venv\`) and the `triton-windows` package are. On Linux, `scripts/avvia_detached.sh '<command>' <log> [name]` starts a command as a `systemd-run --user` unit that keeps running after the terminal closes (it puts `$NCM_VENV/bin`, default `~/venv/ncm/bin`, first in `PATH`), and `scripts/run_attivi.sh` lists the running units. Chains are written with `&&`.

1. **Environment.** Python 3.10, then PyTorch 2.14.1 for CUDA 12.6 from the PyTorch index, then `pip install -r environment/requirements.txt` (on Linux replace `triton-windows` with the Triton build that matches PyTorch). The figure scripts additionally need matplotlib and Pillow.
2. **Upstream code** at the pinned commit in `third_party/ncm/`: see `third_party/README.md`.
3. **Authors' files, needed by every run.** `artifacts/tinystories/` with `config.yaml`, the BPE vocabulary and merges, and the official tokenizer and generator checkpoints (repository `hazyresearch/ncm-checkpoints` on Hugging Face, folder `tinystories`). The architecture and the BPE vocabulary are read from there even when training from scratch. **Use the October 2026 release that every result here is based on**, revision `d342ce8aaea58830eef3b8ad473664673e41f692`; the authors replaced these files on 23 November 2026 with a release that was not evaluated here. For example `huggingface-cli download hazyresearch/ncm-checkpoints --revision d342ce8aaea58830eef3b8ad473664673e41f692 --include "tinystories/*" --local-dir <dir>`, then move the contents of `<dir>/tinystories/` to `artifacts/tinystories/`. SHA-256 of the files used:
   - `tokenizer.pt` (221,861,835 bytes): `76e26003719db32dc3d542518951c286be07d6a5881f29ad5152d299764b0f81`
   - `generator.pt` (356,160,651 bytes): `0acaf4f795390f9f7ccc8f84047867619a1d00e291681e299b855d7e563be6ff`
   - `config.yaml` (2,483 bytes): `4009baccad5a159bfc12cd48f3da17c22c4a281b86f71f2fc70b7c1bbb62004b`
4. **Data.** `data/raw/tinystories/` with `train_bpe_8192.bin`, `train_bpe_8192_offsets.npy` and `valid_bpe_8192.bin` (dataset `hazyresearch/ncm-datasets`, folder `tinystories`).
5. **Judge model.** A local copy of GPT-2 large (weights and tokenizer files) in `evaluation/ref_models/gpt2-large/`; the evaluation runs offline.
6. **Only for generators on the official tokenizer:** pre-compute its codes once with `python -I data/build_codes.py` (writes `data/processed/tinystories_codes_official_L128.u16`), then check them with `python -I data/verify_codes.py`.

Tokenizer (the 38,400-step one), a generator on top of it with the settings of the 51,200-story runs, and its evaluation:

```
python -I tokenizer/train_vqvae.py --run auto train.steps=38400 train.micro_batch=256 train.accum=1 train.warmup=1920 train.eval_every=4800 tokenizer.rimedi.init_dati=true tokenizer.rimedi.riavvio_fino_al_passo=1200 export.at_steps=[9600,19200]

python -I generator/train_generator.py --run auto base_config=runs/mytok-12std-dati-rv_9830400_38400x256_s42/export/config.yaml data.codes_file=null train.steps=400 train.accum=8 train.warmup=20 train.eval_every=200 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4

python -I evaluation/gen_ppl_run.py --run gen-mytok12std-dati-rv-38400x256_51200_400x128_c50_lr1e-3_flex_s42
```

The warm-up is not part of a run's name: always pass it (5% of the steps in almost all reported runs), because the default in `configs/stage2_generator.yaml` is 500 steps. `export.at_steps` writes the two intermediate tokenizer snapshots used above. The exact settings of any run are in `results/runs/<run>/config.yaml`, and the command that started it in `meta.json`.

The looped and grown generators of v0.2 use the same command with `cicli.*` overrides, for example grown 12 → 16 → 20 at 409,600 stories:

```
python -I generator/train_generator.py --run auto train.steps=3200 train.accum=8 train.warmup=160 train.eval_every=1600 train.log_every=10 train.keep_ckpts=1 train.ckpt_every_minutes=5 train.seed=43 data.codes_file=data/processed/tinystories_codes_official_L128.u16 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4 cicli.prima=4 cicli.blocco=4 cicli.ripetizioni=3 cicli.dopo=4 "cicli.crescita_ai_passi=[1067,2133]"
```

Without `crescita_ai_passi` (or `crescita_al_passo`) the same settings train the looped model from scratch. `evaluation/gen_ppl_run.py` reads the `cicli` section of the run's `config.yaml` to build a generator with the right number of layers.

**Numerics on Linux.** `tests/test_tokenizer.py` switches off cuDNN TF32 (`torch.backends.cudnn.allow_tf32 = False`): with TF32 on, its gradient-accumulation check (T5) failed on Linux (difference 0.33) while passing on Windows, because cuDNN picks different algorithms; without TF32 the difference is 6e-07 and all checks pass. Training keeps PyTorch's default (TF32 on) on both systems, so runs on different systems may differ slightly in their numerics.

**Tools added in v0.2 but not yet used for any reported run:** a warmup-stable-decay schedule (`train.schedule=wsd train.decay_steps=<D>`, cosine stays the default), `generator/train_generator.py --run <name> --estendi <steps>` to extend a WSD run with no decay, `scripts/discesa.py --da <run> --passi <D>` for a descent branch evaluated separately, and `scripts/valuta_checkpoint.py --run <run>` to evaluate a paused run's checkpoint.

A whole experiment cycle runs with `python -I scripts/ciclo.py configs/cicli/<cycle>.yaml`. The cycle runner trains generators only: the tokenizers a cycle refers to must be trained first, and rows that point to a run made by an earlier cycle need that run to exist. `configs/cicli/` also holds a few cycles that were cancelled or never launched; they are kept as part of the record. The verdict line at the bottom of each cycle table applies an automatic 5% single-run rule against the cycle's reference run (and may compare different seeds); it is a screening aid, not the decision rule. Decisions use the rules stated in EXPERIMENTS.md.


## Interpretation of budgets and seeds

Stories seen measure data exposure, not distinct examples or equal compute. Training seeds change model initialisation while data order remains fixed. Sampling-seed standard deviations do not measure uncertainty across training runs. Different tokenizers define different code distributions, so raw cross-entropy losses cannot establish equal prediction quality.

The best 38,400-step tokenizer and the 2,400-step tokenizer used in the generator-budget sweep are separate checkpoints. The prepared 409,600-story generator on our tokenizer requires the latter.
