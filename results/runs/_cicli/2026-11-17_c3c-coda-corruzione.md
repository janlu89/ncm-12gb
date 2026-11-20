# Ciclo 2026-11-17_c3c-coda-corruzione

Aggiornato: 2026-11-17 16:11. Comuni a tutti i run: `train.steps=400 train.warmup=20 train.eval_every=200 train.accum=8 train.log_every=10 train.keep_ckpts=1 data.codes_file=data/processed/tinystories_codes_official_L128.u16`

Rumore tra seed (Gen. PPL, run A42, A43): 2.50

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| A42 | base (ciclo 1): lineare 0,75→0,25 (`train.seed=42`) | 4.444 | 6.783 | 1.688 | 149.42 | 4.64 | 3.897 | +0.00 | 84 | 11 |
| A43 | base, seed 43 (ciclo 1) (`train.seed=43`) | 4.445 | 6.781 | 1.679 | 151.92 | 3.04 | 3.913 | +2.50 | 84 | 11 |
| C65 | costante 0,65 (ciclo 2) (`train.seed=42 corruption.rate_first=0.65 corruption.rate_last=0.65`) | 5.139 | 6.769 | 2.813 | 170.56 | 2.58 | 3.850 | +21.14 | 83 | 12 |
| C50 | costante 0,5 (ciclo 1, V3) (`train.seed=42 corruption.rate_first=0.5 corruption.rate_last=0.5`) | 4.402 | 6.786 | 1.909 | 134.72 | 2.53 | 3.871 | -14.70 | 84 | 11 |
| C50s43 | costante 0,5, seed 43 (ciclo 2, V3s43) (`train.seed=43 corruption.rate_first=0.5 corruption.rate_last=0.5`) | 4.417 | 6.785 | 1.950 | 140.92 | 3.21 | 3.895 | -8.50 | 84 | 12 |
| C35 | costante 0,35 (ciclo 2) (`train.seed=42 corruption.rate_first=0.35 corruption.rate_last=0.35`) | 3.886 | 6.785 | 1.479 | 131.31 | 1.80 | 3.923 | -18.11 | 83 | 12 |
| C20 | costante 0,20 (`train.seed=42 corruption.rate_first=0.2 corruption.rate_last=0.2`) | 3.477 | 6.795 | 1.233 | 144.61 | 1.97 | 3.931 | -4.81 | 83 | 12 |
| C10 | costante 0,10 (`train.seed=42 corruption.rate_first=0.1 corruption.rate_last=0.1`) | 3.264 | 6.796 | 1.139 | 173.86 | 5.71 | 3.957 | +24.44 | 80 | 12 |
| C00 | costante 0,00 (nessuna corruzione) (`train.seed=42 corruption.rate_first=0.0 corruption.rate_last=0.0`) | 3.124 | 6.799 | 1.109 | 224.80 | 4.99 | 3.965 | +75.38 | 78 | 12 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.

- C65 (+14.1%): nessun miglioramento oltre il 5%
- C50 (-9.8%): MIGLIORA
- C50s43 (-5.7%): MIGLIORA
- C35 (-12.1%): MIGLIORA
- C20 (-3.2%): nessun miglioramento oltre il 5%
- C10 (+16.4%): nessun miglioramento oltre il 5%
- C00 (+50.4%): nessun miglioramento oltre il 5%
