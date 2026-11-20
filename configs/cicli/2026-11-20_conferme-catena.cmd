@echo off
rem Passi 2 e 3 del 2026-11-20: conferme con un secondo seed prima di pubblicare.
rem Ripresa dopo un'interruzione: rilanciare questo stesso file (ogni pezzo riprende dal checkpoint o viene saltato).
rem Avvio: scripts\avvia_detached.ps1 -Command 'configs\cicli\2026-11-20_conferme-catena.cmd' -Log 'runs\_cicli\2026-11-20_conferme.log'
cd /d "%~dp0..\.."

echo [conferme] 1/3 tokenizer lungo, seed 43 (38.400 passi)
.venv\Scripts\python.exe -u -I tokenizer\train_vqvae.py --run auto train.steps=38400 train.micro_batch=256 train.accum=1 train.warmup=1920 train.eval_every=4800 train.log_every=50 train.ckpt_every_minutes=5 train.seed=43 tokenizer.rimedi.init_dati=true tokenizer.rimedi.riavvio_fino_al_passo=1200
if errorlevel 1 goto fermata

echo [conferme] 2/3 generatore sopra il tokenizer lungo seed 43
.venv\Scripts\python.exe -u -I scripts\ciclo.py configs\cicli\2026-11-19_b2-training-lungo.yaml
if errorlevel 1 goto fermata

echo [conferme] 3/3 prova A con il seed 43 (due generatori a 409.600 storie)
.venv\Scripts\python.exe -u -I scripts\ciclo.py configs\cicli\2026-11-19_a-generatore-lungo.yaml
if errorlevel 1 goto fermata

echo [conferme] completata
exit /b 0

:fermata
echo [conferme] fermata: un pezzo non e' stato completato
exit /b 1
