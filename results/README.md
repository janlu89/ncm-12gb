# Results

Raw records of every training run of this project, kept so that every number can be checked.
Model weights and checkpoints are not included (they are hundreds of MB each).

- `runs/<run name>/` — one folder per run (117 runs plus `_cicli`, the cycle tables and chain logs).
- `runs_index.csv` — one row per folder, parsed from the files below: stage, final log event, steps,
  batch (micro-batch × gradient accumulation), seed, tokenizer used by a generator (`official` or the name
  of our tokenizer run, with ` @ step N` for an intermediate snapshot), generative perplexity
  (mean, standard deviation over sampling seeds), entropy, last validation loss, notes.
- `../EXPERIMENTS.md` — the experiment log: question, setup, decision rule fixed before the data, results and conclusion for each experiment.
- `RUNS.md` — the run-by-run report, generated from these files by `make_report.py` (which also rebuilds `runs_index.csv`).

## Files in each run folder

Only the files present for that run are included.

| File | Content |
|---|---|
| `config.yaml` | Exact settings of the run (after command-line overrides). |
| `meta.json` | Launch command, Python / PyTorch / CUDA versions, GPU, driver, NCM upstream commit, full config. |
| `log.jsonl` | One JSON object per line, one per event: `start`, `train` (loss, learning rate, throughput, peak VRAM, GPU telemetry), `eval` (validation loss, per-level metrics), `checkpoint`, `resume`, `paused`, `done`. |
| `console.log` | Console output of the training process. |
| `eval/genppl/genppl.json` | Generative perplexity measured with GPT-2 large on 1,280 generated samples (256 per seed × 5 sampling seeds): mean, standard deviation over the 5 seeds, mean entropy, per-seed values. |
| `eval/genppl/campioni.txt` | The first 10 of the 1,280 generated samples used for the measurement ("campione" = sample). |
| `eval/genppl/eval_config.yaml` | Configuration passed to the upstream evaluation code. |
| `eval/genppl/generazione.log` | Log of the generation step. |
| `export/config.yaml` | Tokenizers only: the configuration a generator loads as `base_config`. |
| `export/export.json` | Tokenizer runs: reconstruction and code-usage statistics of the exported tokenizer. Generator runs: number of level-0 codes used. |
| `export/passo_*/export.json` | Same statistics for intermediate tokenizer snapshots ("passo" = step). |
| `export/storie_di_prova.txt` | A few stories generated right after training ("storie di prova" = test stories). |

`runs/_cicli/` holds the cycle tables (`.md`), chain logs and analysis text files; the cycle definitions are in `../configs/cicli/`
("ciclo" = cycle: a batch of runs launched together and compared in one table).
The verdict line at the bottom of each cycle table applies an automatic 5% single-run rule against the cycle's reference run (and may compare different seeds); it is a screening aid, not the decision rule. Decisions use the rules stated in EXPERIMENTS.md.

For looped generators (`cicli` in the name), the upstream line "NCP Transformer, Total parameter count" in `console.log` counts tied weights once per repetition (116.96M for 16 applied layers, 145.31M for 20, 88.61M for 4 + 2×2 + 4). The real counts are 88.6M (74.4M for 4 + 2×2 + 4), as printed by the `[generatore]` line.

Paths inside logs and JSON files were made relative to the project root (`.`); `~` stands for the user's
home folder.

## Run names

Names are built from the settings (`common/nomi.py`):

- Generator: `gen-<tokenizer>_<stories seen>_<steps>x<batch>_<corruption>_lr<lr>[_flex][_extra]_s<seed>`
  - `<stories seen>` = steps × batch; it comes before the steps so that runs with the same budget sort together.
  - `<corruption>`: `c50` = constant rate 0.5; `lin75-25` = linear from 0.75 at the first corrupted level to 0.25 at the last;
    `unif0-100` = one rate per story drawn between 0 and 1; `c00` = no corruption; a trailing `l0` = level 0 is
    corrupted too.
  - `_flex` = FlexAttention; `_mb4` = control run with a different micro-batch.
  - `_cicli<a>-<b>x<r>-<c>` (v0.2) = looped generator: a layers, then a block of b layers applied r times with shared
    weights, then c layers. `-cresce<k>` = the repetitions after the first are switched on at step k (before that they
    pass their input through unchanged); `-cresce<k2>-<k3>` = repetition 2 switched on at step k2, repetition 3 at k3.
  - `_wsd` / `_wsd<D>` = warmup-stable-decay learning-rate schedule (constant, then a linear decay over the last D steps).
    Implemented in v0.2 but not used by any reported run.
- Our tokenizer: `mytok-<levels><schedule>[-dati][-rv][-nonorm]_<stories seen>_<steps>x<batch>_s<seed>`
  - Level schedules: `std` (same spacing as the official tokenizer), `lin`, `quad`, `exp`.
- Tokenizer labels inside generator names:
  - `tokuff` = official tokenizer; `tokuff64` = official tokenizer with codes computed in 64-bit precision.
  - `mytok<levels><schedule>-<steps>x<batch>` = ours. A generator on our tokenizer has the same `<rest>` of
    the name as its counterpart on the official one: `gen-tokuff_<rest>` and `gen-mytok…_<rest>`.
  - `-dati` = codebooks initialized from data (k-means); `-rv` = early restart of dead codes;
    `-nonorm` = no level-0 normalization; `rel` (as in `-rvrel`) = dead-code restart threshold relative to the
    level's mean code usage; `-d9968` = codebook EMA decay 0.9968 instead of 0.95;
    `-pNNNN` = intermediate tokenizer snapshot at step NNNN;
    `-ts43` = tokenizer trained with seed 43 (the default is 42).

Runs whose name ends in `_interrotta` (interrupted) or `_annullata` (cancelled) did not finish. They are kept
for completeness and are not used in any comparison.

## Italian field names

Some keys in logs and JSON files are in Italian. The most common:

| Key | Meaning |
|---|---|
| `sequenze_perfette` | Fraction of stories reconstructed perfectly by the tokenizer. |
| `codici_usati_per_livello` | Fraction of codebook entries used, per level. |
| `codici_usati_media` | Mean of the above across levels. |
| `acc_per_livello` | Token reconstruction accuracy when decoding with the first k levels (k = 1, 2, …). |
| `livelli` | Number of codes (positions) at each level; their sum is `codici_per_storia`. |
| `passo` | Step. |
| `val_per_livello` | Validation loss per level (generator). |
| `nll_media` | Mean negative log-likelihood. |
| `level0_codici_usati` | Number of distinct level-0 codes used. |
| `secondi_generazione`, `secondi_totali` | Generation time and total evaluation time, in seconds. |
| `comando`, `creato`, `data`, `piattaforma` | Command, created, date, platform. |
