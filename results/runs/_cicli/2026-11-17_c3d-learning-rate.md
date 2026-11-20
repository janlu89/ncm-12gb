# Ciclo 2026-11-17_c3d-learning-rate

Aggiornato: 2026-11-17 16:11. Comuni a tutti i run: `train.steps=400 train.warmup=20 train.eval_every=200 train.accum=8 train.log_every=10 train.keep_ckpts=1 data.codes_file=data/processed/tinystories_codes_official_L128.u16`

Rumore tra seed (Gen. PPL, run ): non ancora misurato

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| LR3e-4 | V3V4, lr 3e-4 (ciclo 2) (`train.seed=42 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64`) | 4.383 | 6.783 | 1.913 | 131.48 | 2.57 | 3.888 | +0.00 | 111 | 10 |
| LR1e-4 | V3V4, lr 1e-4 (`train.seed=42 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-4 train.min_lr=1.0e-5`) | 6.096 | 6.874 | 4.938 | 358.09 | 10.31 | 3.803 | +226.61 | 109 | 10 |
| LR1e-3 | V3V4, lr 1e-3 (`train.seed=42 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 3.302 | 6.684 | 0.789 | 65.89 | 1.27 | 3.939 | -65.59 | 104 | 10 |
| LR3e-3 | V3V4, lr 3e-3 (`train.seed=42 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=3.0e-3 train.min_lr=3.0e-4`) | 4.084 | 6.674 | 1.305 | 76.87 | 0.98 | 3.971 | -54.61 | 105 | 10 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.

- LR1e-4 (+172.4%): nessun miglioramento oltre il 5%
- LR1e-3 (-49.9%): MIGLIORA
- LR3e-3 (-41.5%): MIGLIORA
