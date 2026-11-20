# Ciclo 2026-11-18_f2-calibrazione

Aggiornato: 2026-11-19 10:11. Comuni a tutti i run: `train.steps=400 train.accum=8 train.warmup=20 train.eval_every=200 train.log_every=10 train.keep_ckpts=1`

Rumore tra seed (Gen. PPL, run UFF_best42, UFF_best43): 0.73

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| UFF_best42 | tokenizer ufficiale (~30.000 passi × 4.096) (`train.seed=42`) | 3.302 | 6.684 | 0.789 | 65.89 | 1.27 | 3.939 | +0.00 | 104 | 10 |
| UFF_best43 | tokenizer ufficiale, seed 43 (`train.seed=43`) | 3.269 | 6.685 | 0.775 | 66.62 | 1.60 | 3.927 | +0.73 | 111 | 10 |
| T2400 | nostro, 2.400 passi (614.400 storie) (`train.seed=42`) | 3.875 | 6.547 | 1.320 | 106.54 | 2.26 | 3.873 | +40.65 | 93 | 11 |
| T2400s43 | nostro, 2.400 passi, seed 43 (`train.seed=43`) | 3.917 | 6.547 | 1.353 | 109.80 | 1.97 | 3.870 | +43.91 | 91 | 12 |
| T4800 | nostro, 4.800 passi (1.228.800 storie) (`train.seed=42 base_config=runs/mytok-12std-dati-rv_1228800_4800x256_s42/export/config.yaml data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 3.809 | 6.688 | 1.318 | 103.61 | 2.25 | 3.910 | +37.72 | 97 | 11 |
| UFF_102k | ufficiale, generatore a 102.400 storie (`train.seed=42`) | 2.590 | 6.612 | 0.543 | 43.35 | 0.79 | 3.951 | -22.54 | 110 | 18 |
| T2400_102k | nostro 2.400 passi, generatore a 102.400 storie (`train.seed=42 train.steps=800 train.warmup=40 train.eval_every=400 base_config=runs/mytok-12std-dati-rv_614400_2400x256_s42/export/config.yaml data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 2.963 | 6.349 | 0.822 | 65.68 | 1.23 | 3.918 | -0.21 | 97 | 20 |
| T4800_102k | nostro 4.800 passi, generatore a 102.400 storie (`train.seed=42 train.steps=800 train.warmup=40 train.eval_every=400 base_config=runs/mytok-12std-dati-rv_1228800_4800x256_s42/export/config.yaml data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 3.021 | 6.512 | 0.918 | 67.59 | 1.33 | 3.939 | +1.70 | 105 | 19 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.

- T2400 (+61.7%): nessun miglioramento oltre il 5%
- T2400s43 (+66.6%): nessun miglioramento oltre il 5%
- T4800 (+57.2%): nessun miglioramento oltre il 5%
- UFF_102k (-34.2%): MIGLIORA
- T2400_102k (-0.3%): nessun miglioramento oltre il 5%
- T4800_102k (+2.6%): nessun miglioramento oltre il 5%
