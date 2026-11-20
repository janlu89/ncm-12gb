# Ciclo 2026-11-17_c3e-conferme

Aggiornato: 2026-11-17 16:11. Comuni a tutti i run: `train.accum=8 train.log_every=10 train.keep_ckpts=1 data.codes_file=data/processed/tinystories_codes_official_L128.u16`

Rumore tra seed (Gen. PPL, run ): non ancora misurato

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| A43_800 | base, seed 43, 800 passi (rumore a 800) (`train.steps=800 train.warmup=40 train.eval_every=400 train.seed=43`) | 3.300 | 6.696 | 0.707 | 76.11 | 1.57 | 3.939 | — | 84 | 23 |
| V3s43_800 | costante 0,5, seed 43, 800 passi (tenuta di V3) (`train.steps=800 train.warmup=40 train.eval_every=400 train.seed=43 corruption.rate_first=0.5 corruption.rate_last=0.5`) | 3.238 | 6.698 | 0.800 | 71.67 | 1.23 | 3.917 | — | 81 | 23 |
| A43_200 | base, seed 43, 200 passi (rumore a 200) (`train.steps=200 train.warmup=10 train.eval_every=100 train.seed=43`) | 5.941 | 6.884 | 4.155 | 350.15 | 5.39 | 3.839 | — | 79 | 7 |
| C35s43 | costante 0,35, seed 43, 400 passi (0,35 contro 0,5) (`train.steps=400 train.warmup=20 train.eval_every=200 train.seed=43 corruption.rate_first=0.35 corruption.rate_last=0.35`) | 3.883 | 6.796 | 1.486 | 138.28 | 2.04 | 3.932 | — | 80 | 12 |
| V3V4s43 | costante 0,5 + Flex, seed 43, 400 passi (conferma di V3V4) (`train.steps=400 train.warmup=20 train.eval_every=200 train.seed=43 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64`) | 4.394 | 6.774 | 1.952 | 134.74 | 2.79 | 3.893 | — | 104 | 10 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.
