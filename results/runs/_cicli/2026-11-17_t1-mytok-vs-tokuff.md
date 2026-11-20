# Ciclo 2026-11-17_t1-mytok-vs-tokuff

Aggiornato: 2026-11-17 17:33. Comuni a tutti i run: `train.steps=400 train.accum=8 train.warmup=20 train.eval_every=200 train.log_every=10 train.keep_ckpts=1 data.codes_file=data/processed/tinystories_codes_official_L128.u16`

Rumore tra seed (Gen. PPL, run UFF_paper42, UFF_paper43): 2.50

| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |
|---|---|---|---|---|---|---|---|---|---|---|
| UFF_paper42 | tokenizer ufficiale, ricetta del paper (`train.seed=42`) | 4.444 | 6.783 | 1.688 | 149.42 | 4.64 | 3.897 | +0.00 | 84 | 11 |
| UFF_paper43 | tokenizer ufficiale, ricetta del paper, seed 43 (`train.seed=43`) | 4.445 | 6.781 | 1.679 | 151.92 | 3.04 | 3.913 | +2.50 | 84 | 11 |
| UFF_best42 | tokenizer ufficiale, costante 0,5 + Flex + lr 1e-3 (`train.seed=42`) | 3.302 | 6.684 | 0.789 | 65.89 | 1.27 | 3.939 | -83.53 | 104 | 10 |
| MY_paper42 | tokenizer nostro, ricetta del paper (`train.seed=42 base_config=runs/mytok-12std_614400_2400x256_s42/export/config.yaml data.codes_file=null`) | 5.364 | 6.478 | 2.343 | 474.54 | 13.20 | 3.834 | +325.12 | 78 | 12 |
| MY_best42 | tokenizer nostro, costante 0,5 + Flex + lr 1e-3 (`train.seed=42 base_config=runs/mytok-12std_614400_2400x256_s42/export/config.yaml data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 4.615 | 6.316 | 1.698 | 219.24 | 5.39 | 3.857 | +69.82 | 101 | 10 |
| MY_paper43 | tokenizer nostro, ricetta del paper, seed 43 (`train.seed=43 base_config=runs/mytok-12std_614400_2400x256_s42/export/config.yaml data.codes_file=null`) | 5.347 | 6.493 | 2.278 | 463.36 | 7.05 | 3.833 | +313.94 | 79 | 12 |
| UFF_best43 | tokenizer ufficiale, costante 0,5 + Flex + lr 1e-3, seed 43 (`train.seed=43 corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 3.269 | 6.685 | 0.775 | 66.62 | 1.60 | 3.927 | -82.80 | 111 | 10 |
| MY_best43 | tokenizer nostro, costante 0,5 + Flex + lr 1e-3, seed 43 (`train.seed=43 base_config=runs/mytok-12std_614400_2400x256_s42/export/config.yaml data.codes_file=null corruption.rate_first=0.5 corruption.rate_last=0.5 model_overrides.use_flex_attention=true model_overrides.flex_block_size=64 train.lr=1.0e-3 train.min_lr=1.0e-4`) | 4.578 | 6.311 | 1.660 | 226.31 | 4.69 | 3.869 | +76.89 | 102 | 10 |

Regola di decisione (docs/piani/ciclo3-operativo.md, 1b): una variante vince solo se la sua Gen. PPL è più bassa di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1.

- UFF_best42 (-55.9%): MIGLIORA
- MY_paper42 (+217.6%): nessun miglioramento oltre il 5%
- MY_best42 (+46.7%): nessun miglioramento oltre il 5%
- MY_paper43 (+210.1%): nessun miglioramento oltre il 5%
- UFF_best43 (-55.4%): MIGLIORA
- MY_best43 (+51.5%): nessun miglioramento oltre il 5%
