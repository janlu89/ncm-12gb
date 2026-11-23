# Ciclo 2026-11-22_soglia-relativa

Aggiornato: 2026-11-22 15:40. Comuni a tutti i run: `train.steps=400 train.accum=8 train.warmup=20 train.eval_every=200 train.log_every=10 train.keep_ckpts=1`

Rumore tra seed (Gen. PPL, run R12, R12_s43): 3.26

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| R12 | R12, soglia fissa (gia fatto) (`train.seed=42`) | 3.875 | 6.547 | 1.320 | 106.54 | 2.26 | 3.873 | +0.00 | 93 | 11 |
| R12_s43 | R12, soglia fissa, seed 43 (gia fatto) (`train.seed=43`) | 3.917 | 6.547 | 1.353 | 109.80 | 1.97 | 3.870 | +3.26 | 91 | 12 |
| REL | R12 con soglia relativa (`train.seed=42 base_config=runs/mytok-12std-dati-rvrel_614400_2400x256_s42/export/config.yaml data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 3.944 | 6.606 | 1.369 | 114.90 | 2.05 | 3.867 | +8.36 | 100 | 12 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.

- REL (+7.8%): nessun miglioramento oltre il 5%
