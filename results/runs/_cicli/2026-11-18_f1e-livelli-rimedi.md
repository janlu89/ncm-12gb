# Ciclo 2026-11-18_f1e-livelli-rimedi

Aggiornato: 2026-11-18 22:38. Comuni a tutti i run: `train.steps=400 train.accum=8 train.warmup=20 train.eval_every=200 train.log_every=10 train.keep_ckpts=1`

Rumore tra seed (Gen. PPL, run L12_paper42, L12_paper43): 12.52

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| L12_paper42 | 12 livelli R12, ricetta del paper (`train.seed=42`) | 4.869 | 6.797 | 1.901 | 225.86 | 2.76 | 3.859 | +0.00 | 73 | 13 |
| L12_paper43 | 12 livelli R12, ricetta del paper, seed 43 (`train.seed=43`) | 4.917 | 6.776 | 1.972 | 213.34 | 5.05 | 3.816 | -12.52 | 70 | 13 |
| L12_best42 | 12 livelli R12, nostra migliore (`train.seed=42`) | 3.875 | 6.547 | 1.320 | 106.54 | 2.26 | 3.873 | -119.32 | 93 | 11 |
| L12_best43 | 12 livelli R12, nostra migliore, seed 43 (`train.seed=43`) | 3.917 | 6.547 | 1.353 | 109.80 | 1.97 | 3.870 | -116.06 | 91 | 12 |
| L8_paper42 | 8 livelli R12, ricetta del paper (`train.seed=42 base_config=runs/mytok-8lin-dati-rv_614400_2400x256_s42/export/config.yaml data.codes_file=null`) | 4.886 | 6.873 | — | 335.06 | 8.35 | 3.914 | +109.20 | 100 | 10 |
| L8_best42 | 8 livelli R12, nostra migliore (`train.seed=42 base_config=runs/mytok-8lin-dati-rv_614400_2400x256_s42/export/config.yaml data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 3.906 | 6.701 | — | 170.65 | 7.27 | 3.937 | -55.21 | 121 | 9 |
| L16_paper42 | 16 livelli R12, ricetta del paper (`train.seed=42 base_config=runs/mytok-16lin-dati-rv_614400_2400x256_s42/export/config.yaml data.codes_file=null train.micro_batch=8 train.accum=16 eval.batch=16 eval.encode_batch=64`) | 5.734 | 6.759 | — | 307.24 | 10.99 | 3.777 | +81.38 | 41 | 23 |
| L16_best42 | 16 livelli R12, nostra migliore (`train.seed=42 base_config=runs/mytok-16lin-dati-rv_614400_2400x256_s42/export/config.yaml data.codes_file=null train.micro_batch=8 train.accum=16 eval.batch=16 eval.encode_batch=64 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 4.522 | 6.554 | — | 115.54 | 3.38 | 3.841 | -110.32 | 59 | 19 |
| L26_paper42 | 26 livelli R12, ricetta del paper (`train.seed=42 base_config=runs/mytok-26lin-dati-rv_614400_2400x256_s42/export/config.yaml data.codes_file=null train.micro_batch=8 train.accum=16 eval.batch=8 eval.encode_batch=64`) | 6.151 | 6.412 | — | 418.63 | 5.85 | 3.710 | +192.77 | 21 | 44 |
| L26_best42 | 26 livelli R12, nostra migliore (`train.seed=42 base_config=runs/mytok-26lin-dati-rv_614400_2400x256_s42/export/config.yaml data.codes_file=null train.micro_batch=8 train.accum=16 eval.batch=8 eval.encode_batch=64 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 5.057 | 6.160 | — | 144.73 | 2.68 | 3.780 | -81.13 | 37 | 33 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.

- L12_best42 (-52.8%): MIGLIORA
- L12_best43 (-51.4%): MIGLIORA
- L8_paper42 (+48.3%): nessun miglioramento oltre il 5%
- L8_best42 (-24.4%): MIGLIORA
- L16_paper42 (+36.0%): nessun miglioramento oltre il 5%
- L16_best42 (-48.8%): MIGLIORA
- L26_paper42 (+85.3%): nessun miglioramento oltre il 5%
- L26_best42 (-35.9%): MIGLIORA
