# Ciclo 2026-11-19_a-generatore-lungo

Aggiornato: 2026-11-20 18:18. Comuni a tutti i run: `train.steps=3200 train.accum=8 train.warmup=160 train.eval_every=1600 train.log_every=10 train.keep_ckpts=1 train.ckpt_every_minutes=5`

Rumore tra seed (Gen. PPL, run UFF_best42, UFF_best43): 0.73

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| UFF_best42 | ufficiale, 51.200 storie (gia fatto) (`train.seed=42`) | 3.302 | 6.684 | 0.789 | 65.89 | 1.27 | 3.939 | +39.85 | 104 | 10 |
| UFF_best43 | ufficiale, 51.200 storie, seed 43 (gia fatto) (`train.seed=43`) | 3.269 | 6.685 | 0.775 | 66.62 | 1.60 | 3.927 | +40.58 | 111 | 10 |
| UFF_409k_s42 | ufficiale, generatore a 409.600 storie (`train.seed=42 data.codes_file=data/processed/tinystories_codes_official_L128.u16 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 1.920 | 6.518 | 0.394 | 26.04 | 0.48 | 3.952 | +0.00 | 116 | 256 |
| R12_409k_s42 | nostro R12 2.400 passi, generatore a 409.600 storie (`train.seed=42 base_config=runs/mytok-12std-dati-rv_614400_2400x256_s42/export/config.yaml data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 1.946 | 5.126 | 0.498 | 42.76 | 1.05 | 3.965 | +16.72 | 105 | 70 |
| UFF_409k_s43 | ufficiale, generatore a 409.600 storie, seed 43 (`train.seed=43 data.codes_file=data/processed/tinystories_codes_official_L128.u16 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 1.915 | 6.515 | 0.393 | 25.71 | 0.12 | 3.946 | -0.33 | 116 | 63 |
| R12_409k_s43 | nostro R12 2.400 passi, generatore a 409.600 storie, seed 43 (`train.seed=43 base_config=runs/mytok-12std-dati-rv_614400_2400x256_s42/export/config.yaml data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 1.950 | 5.126 | 0.498 | 41.24 | 1.36 | 3.961 | +15.20 | 102 | 70 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.

- R12_409k_s42 (+64.2%): nessun miglioramento oltre il 5%
- UFF_409k_s43 (-1.3%): nessun miglioramento oltre il 5%
- R12_409k_s43 (+58.4%): nessun miglioramento oltre il 5%
