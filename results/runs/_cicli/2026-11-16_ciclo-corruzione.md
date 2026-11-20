# Ciclo 2026-11-16_ciclo-corruzione

Aggiornato: 2026-11-17 16:11. Comuni a tutti i run: `train.steps=400 train.accum=8 train.warmup=20 train.eval_every=200 train.log_every=10 train.keep_ckpts=1 data.codes_file=data/processed/tinystories_codes_official_L128.u16`

Rumore tra seed (Gen. PPL, run A42, A43): 2.50

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| A42 | base: corruzione lineare 0,75→0,25, livello 0 escluso (`train.seed=42`) | 4.444 | 6.783 | 1.688 | 149.42 | 4.64 | 3.897 | +0.00 | 84 | 11 |
| A43 | base, seed diverso (misura del rumore) (`train.seed=43`) | 4.445 | 6.781 | 1.679 | 151.92 | 3.04 | 3.913 | +2.50 | 84 | 11 |
| V1 | livello 0 incluso nella corruzione (`train.seed=42 corruption.skip_level0=false`) | 4.563 | 7.057 | 1.744 | 150.65 | 2.58 | 3.906 | +1.23 | 84 | 11 |
| V2 | tasso per esempio ~ U[0,1], livello 0 escluso (`train.seed=42 corruption.mode=per_example_uniform`) | 4.189 | 6.777 | 1.711 | 169.47 | 3.10 | 3.904 | +20.05 | 84 | 12 |
| V3 | tasso costante 0,5 (`train.seed=42 corruption.rate_first=0.5 corruption.rate_last=0.5`) | 4.402 | 6.786 | 1.909 | 134.72 | 2.53 | 3.871 | -14.70 | 84 | 11 |
| V4 | FlexAttention blocchi 64 (più veloce; senza dropout dell'attenzione) (`train.seed=42 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64`) | 4.397 | 6.773 | 1.675 | 142.64 | 4.70 | 3.899 | -6.78 | 111 | 11 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.

- V1 (+0.8%): nessun miglioramento oltre il 5%
- V2 (+13.4%): nessun miglioramento oltre il 5%
- V3 (-9.8%): MIGLIORA
- V4 (-4.5%): nessun miglioramento oltre il 5%
