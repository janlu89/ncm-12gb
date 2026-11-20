# Ciclo 2026-11-19_b2-training-lungo

Aggiornato: 2026-11-20 16:01. Comuni a tutti i run: `train.steps=400 train.accum=8 train.warmup=20 train.eval_every=200 train.log_every=10 train.keep_ckpts=1`

Rumore tra seed (Gen. PPL, run UFF_best42, UFF_best43): 0.73

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| UFF_best42 | tokenizer ufficiale (`train.seed=42`) | 3.302 | 6.684 | 0.789 | 65.89 | 1.27 | 3.939 | -37.72 | 104 | 10 |
| UFF_best43 | tokenizer ufficiale, seed 43 (`train.seed=43`) | 3.269 | 6.685 | 0.775 | 66.62 | 1.60 | 3.927 | -36.99 | 111 | 10 |
| T4800 | nostro, 4.800 passi (gia fatto) (`train.seed=42`) | 3.809 | 6.688 | 1.318 | 103.61 | 2.25 | 3.910 | +0.00 | 97 | 11 |
| T38400 | nostro, 38.400 passi (`train.seed=42 base_config=runs/mytok-12std-dati-rv_9830400_38400x256_s42/export/config.yaml data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 4.015 | 7.063 | 1.557 | 91.46 | 1.66 | 3.934 | -12.15 | 99 | 11 |
| T38400_p9600 | copia del tokenizer lungo al passo 9.600 (`train.seed=42 base_config=runs/mytok-12std-dati-rv_9830400_38400x256_s42/export/passo_00009600/config.yaml data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 3.954 | 6.951 | 1.499 | 95.15 | 1.76 | 3.902 | -8.46 | 98 | 11 |
| T38400_p19200 | copia del tokenizer lungo al passo 19.200 (`train.seed=42 base_config=runs/mytok-12std-dati-rv_9830400_38400x256_s42/export/passo_00019200/config.yaml data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 4.033 | 7.066 | 1.549 | 90.27 | 2.73 | 3.925 | -13.34 | 97 | 11 |
| T38400_ts43 | nostro, 38.400 passi, tokenizer seed 43 (`train.seed=42 train.ckpt_every_minutes=5 base_config=runs/mytok-12std-dati-rv_9830400_38400x256_s43/export/config.yaml data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 4.002 | 7.056 | 1.548 | 93.23 | 2.46 | 3.941 | -10.38 | 102 | 11 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.

- T38400 (-11.7%): MIGLIORA
- T38400_p9600 (-8.2%): MIGLIORA
- T38400_p19200 (-12.9%): MIGLIORA
- T38400_ts43 (-10.0%): MIGLIORA
