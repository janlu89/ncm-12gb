# Ciclo 2026-11-17_ciclo2-corruzione

Aggiornato: 2026-11-17 16:11. Comuni a tutti i run: `train.steps=400 train.accum=8 train.warmup=20 train.eval_every=200 train.log_every=10 train.keep_ckpts=1 data.codes_file=data/processed/tinystories_codes_official_L128.u16`

Rumore tra seed (Gen. PPL, run A42, A43): 2.50

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| A42 | base (ciclo 1): lineare 0,75→0,25 (`train.seed=42`) | 4.444 | 6.783 | 1.688 | 149.42 | 4.64 | 3.897 | +0.00 | 84 | 11 |
| A43 | base, seed 43 (ciclo 1) (`train.seed=43`) | 4.445 | 6.781 | 1.679 | 151.92 | 3.04 | 3.913 | +2.50 | 84 | 11 |
| V3 | costante 0,5 (ciclo 1) (`train.seed=42 corruption.rate_first=0.5 corruption.rate_last=0.5`) | 4.402 | 6.786 | 1.909 | 134.72 | 2.53 | 3.871 | -14.70 | 84 | 11 |
| V4 | FlexAttention (ciclo 1) (`train.seed=42 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64`) | 4.397 | 6.773 | 1.675 | 142.64 | 4.70 | 3.899 | -6.78 | 111 | 11 |
| V3s43 | costante 0,5, seed 43 (conferma di V3; da confrontare con A43) (`train.seed=43 corruption.rate_first=0.5 corruption.rate_last=0.5`) | 4.417 | 6.785 | 1.950 | 140.92 | 3.21 | 3.895 | -8.50 | 84 | 12 |
| V4s43 | FlexAttention, seed 43 (conferma di V4; da confrontare con A43) (`train.seed=43 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64`) | 4.413 | 6.772 | 1.686 | 143.01 | 3.99 | 3.892 | -6.41 | 111 | 10 |
| C35 | costante 0,35 (`train.seed=42 corruption.rate_first=0.35 corruption.rate_last=0.35`) | 3.886 | 6.785 | 1.479 | 131.31 | 1.80 | 3.923 | -18.11 | 83 | 12 |
| C65 | costante 0,65 (`train.seed=42 corruption.rate_first=0.65 corruption.rate_last=0.65`) | 5.139 | 6.769 | 2.813 | 170.56 | 2.58 | 3.850 | +21.14 | 83 | 12 |
| R | lineare rovesciata 0,25→0,75 (più corruzione sui livelli fini) (`train.seed=42 corruption.rate_first=0.25 corruption.rate_last=0.75`) | 4.422 | 6.777 | 2.192 | 144.71 | 3.52 | 3.889 | -4.71 | 83 | 12 |
| V3V4 | costante 0,5 + FlexAttention (si sommano?) (`train.seed=42 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64`) | 4.383 | 6.783 | 1.913 | 131.48 | 2.57 | 3.888 | -17.94 | 111 | 10 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.

- V3 (-9.8%): MIGLIORA
- V4 (-4.5%): nessun miglioramento oltre il 5%
- V3s43 (-5.7%): MIGLIORA
- V4s43 (-4.3%): nessun miglioramento oltre il 5%
- C35 (-12.1%): MIGLIORA
- C65 (+14.1%): nessun miglioramento oltre il 5%
- R (-3.2%): nessun miglioramento oltre il 5%
- V3V4 (-12.0%): MIGLIORA
