# Ciclo 2026-11-22_cicli-generatore

Aggiornato: 2026-11-22 18:09. Comuni a tutti i run: `train.steps=400 train.accum=8 train.warmup=20 train.eval_every=200 train.log_every=10 train.keep_ckpts=1`

Rumore tra seed (Gen. PPL, run UFF_best42, UFF_best43): 0.73

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| UFF_best42 | 12 strati normali (gia fatto) (`train.seed=42`) | 3.302 | 6.684 | 0.789 | 65.89 | 1.27 | 3.939 | +0.00 | 104 | 10 |
| UFF_best43 | 12 strati normali, seed 43 (gia fatto) (`train.seed=43`) | 3.269 | 6.685 | 0.775 | 66.62 | 1.60 | 3.927 | +0.73 | 111 | 10 |
| C4-2x2-4 | 4 + 2x2 + 4: stesso calcolo, 10 strati di pesi (`train.seed=42 data.codes_file=data/processed/tinystories_codes_official_L128.u16 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4 cicli.prima=4 cicli.blocco=2 cicli.ripetizioni=2 cicli.dopo=4`) | 3.360 | 6.689 | 0.809 | 64.18 | 1.33 | 3.940 | -1.71 | 109 | 11 |
| C4-4x2-4 | 4 + 4x2 + 4: stessi pesi, +33% calcolo (`train.seed=42 data.codes_file=data/processed/tinystories_codes_official_L128.u16 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4 cicli.prima=4 cicli.blocco=4 cicli.ripetizioni=2 cicli.dopo=4`) | 3.309 | 6.685 | 0.794 | 64.85 | 0.33 | 3.947 | -1.04 | 84 | 14 |
| C4-4x3-4 | 4 + 4x3 + 4: stessi pesi, +67% calcolo (`train.seed=42 data.codes_file=data/processed/tinystories_codes_official_L128.u16 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4 cicli.prima=4 cicli.blocco=4 cicli.ripetizioni=3 cicli.dopo=4`) | 3.300 | 6.684 | 0.804 | 64.43 | 1.02 | 3.936 | -1.46 | 68 | 18 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.

- C4-2x2-4 (-2.6%): nessun miglioramento oltre il 5%
- C4-4x2-4 (-1.6%): nessun miglioramento oltre il 5%
- C4-4x3-4 (-2.2%): nessun miglioramento oltre il 5%
