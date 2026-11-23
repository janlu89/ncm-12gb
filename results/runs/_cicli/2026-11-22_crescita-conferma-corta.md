# Ciclo 2026-11-22_crescita-conferma-corta

Aggiornato: 2026-11-23 08:01. Comuni a tutti i run: `train.steps=400 train.accum=8 train.warmup=20 train.eval_every=200 train.log_every=10 train.keep_ckpts=1`

Rumore tra seed (Gen. PPL, run UFF_best42, UFF_best43): 0.73

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| UFF_best42 | 12 strati normali (gia fatto) (`train.seed=42`) | 3.302 | 6.684 | 0.789 | 65.89 | 1.27 | 3.939 | +1.04 | 104 | 10 |
| UFF_best43 | 12 strati normali, seed 43 (gia fatto) (`train.seed=43`) | 3.269 | 6.685 | 0.775 | 66.62 | 1.60 | 3.927 | +1.77 | 111 | 10 |
| C4-4x2-4 | 16 strati da zero (gia fatto) (`train.seed=42`) | 3.309 | 6.685 | 0.794 | 64.85 | 0.33 | 3.947 | +0.00 | 84 | 14 |
| C4-4x3-4 | 20 strati da zero (gia fatto) (`train.seed=42`) | 3.300 | 6.684 | 0.804 | 64.43 | 1.02 | 3.936 | -0.42 | 68 | 18 |
| CRESCE200 | cresce 12->16 (gia fatto) (`train.seed=42`) | 3.283 | 6.686 | 0.789 | 60.91 | 0.85 | 3.946 | -3.94 | 84 | 12 |
| C4-4x2-4_s43 | 16 strati da zero, seed 43 (`train.seed=43 data.codes_file=data/processed/tinystories_codes_official_L128.u16 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4 cicli.prima=4 cicli.blocco=4 cicli.ripetizioni=2 cicli.dopo=4`) | 3.274 | 6.682 | 0.788 | 66.25 | 1.51 | 3.947 | +1.40 | 82 | 13 |
| CRESCE200_s43 | cresce 12->16 al passo 200, seed 43 (`train.seed=43 data.codes_file=data/processed/tinystories_codes_official_L128.u16 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4 cicli.prima=4 cicli.blocco=4 cicli.ripetizioni=2 cicli.dopo=4 cicli.crescita_al_passo=200`) | 3.320 | 6.684 | 0.802 | 68.25 | 0.73 | 3.938 | +3.40 | 82 | 12 |
| CRESCE3 | cresce 12->16->20 ai passi 133 e 266 (`train.seed=42 data.codes_file=data/processed/tinystories_codes_official_L128.u16 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4 cicli.prima=4 cicli.blocco=4 cicli.ripetizioni=3 cicli.dopo=4 cicli.crescita_ai_passi=[133,266]`) | 3.296 | 6.683 | 0.790 | 62.46 | 0.36 | 3.933 | -2.39 | 72 | 14 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.

- C4-4x3-4 (-0.6%): nessun miglioramento oltre il 5%
- CRESCE200 (-6.1%): MIGLIORA
- C4-4x2-4_s43 (+2.2%): nessun miglioramento oltre il 5%
- CRESCE200_s43 (+5.2%): nessun miglioramento oltre il 5%
- CRESCE3 (-3.7%): nessun miglioramento oltre il 5%
