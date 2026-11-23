# Ciclo 2026-11-22_crescita-conferma-lunga

Aggiornato: 2026-11-23 10:36. Comuni a tutti i run: `train.steps=3200 train.accum=8 train.warmup=160 train.eval_every=1600 train.log_every=10 train.keep_ckpts=1 train.ckpt_every_minutes=5`

Rumore tra seed (Gen. PPL, run UFF_409k_s42, UFF_409k_s43): 0.33

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| UFF_409k_s42 | 12 strati normali (gia fatto) (`train.seed=42`) | 1.920 | 6.518 | 0.394 | 26.04 | 0.48 | 3.952 | +0.84 | 116 | 256 |
| UFF_409k_s43 | 12 strati normali, seed 43 (gia fatto) (`train.seed=43`) | 1.915 | 6.515 | 0.393 | 25.71 | 0.12 | 3.946 | +0.51 | 116 | 63 |
| C16_409k | 16 strati da zero (`train.seed=42 data.codes_file=data/processed/tinystories_codes_official_L128.u16 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4 cicli.prima=4 cicli.blocco=4 cicli.ripetizioni=2 cicli.dopo=4`) | 1.891 | 6.515 | 0.389 | 25.20 | 0.46 | 3.964 | +0.00 | 90 | 80 |
| CRESCE_409k | cresce 12->16 al passo 1600 (`train.seed=42 data.codes_file=data/processed/tinystories_codes_official_L128.u16 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4 cicli.prima=4 cicli.blocco=4 cicli.ripetizioni=2 cicli.dopo=4 cicli.crescita_al_passo=1600`) | 1.905 | 6.516 | 0.394 | 24.78 | 0.55 | 3.955 | -0.42 | 88 | 718 |
| CRESCE3_409k | cresce 12->16->20 ai passi 1067 e 2133 (`train.seed=42 data.codes_file=data/processed/tinystories_codes_official_L128.u16 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4 cicli.prima=4 cicli.blocco=4 cicli.ripetizioni=3 cicli.dopo=4 cicli.crescita_ai_passi=[1067,2133]`) | 1.898 | 6.516 | 0.392 | 24.53 | 0.83 | 3.952 | -0.67 | 71 | 82 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.

- CRESCE_409k (-1.7%): nessun miglioramento oltre il 5%
- CRESCE3_409k (-2.7%): nessun miglioramento oltre il 5%
