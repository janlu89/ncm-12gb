# Ciclo 2026-11-19_c-ufficiale-64bit

Aggiornato: 2026-11-19 13:03. Comuni a tutti i run: `train.steps=400 train.accum=8 train.warmup=20 train.eval_every=200 train.log_every=10 train.keep_ckpts=1`

Rumore tra seed (Gen. PPL, run UFF32_s42, UFF32_s43): 0.73

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| UFF32_s42 | ufficiale, codici a 32 bit (`train.seed=42`) | 3.302 | 6.684 | 0.789 | 65.89 | 1.27 | 3.939 | +0.00 | 104 | 10 |
| UFF32_s43 | ufficiale, codici a 32 bit, seed 43 (`train.seed=43`) | 3.269 | 6.685 | 0.775 | 66.62 | 1.60 | 3.927 | +0.73 | 111 | 10 |
| UFF64_s42 | ufficiale, codici a 64 bit (esatti) (`train.seed=42 encode_fp64=true data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 3.290 | 6.970 | 0.781 | 66.42 | 1.09 | 3.950 | +0.53 | 47 | 20 |
| UFF64_s43 | ufficiale, codici a 64 bit (esatti), seed 43 (`train.seed=43 encode_fp64=true data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 3.249 | 6.971 | 0.772 | 66.63 | 0.66 | 3.925 | +0.74 | 47 | 20 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.

- UFF64_s42 (+0.8%): nessun miglioramento oltre il 5%
- UFF64_s43 (+1.1%): nessun miglioramento oltre il 5%
