@echo off
rem Prova B (2026-11-19): catena completa. Si ferma al primo pezzo non completato (pausa, STOP, errore).
rem Ripresa dopo un'interruzione: rilanciare questo stesso file (ogni pezzo riprende dal suo checkpoint o viene saltato).
rem Avvio: scripts\avvia_detached.ps1 -Command 'configs\cicli\2026-11-19_b-catena.cmd' -Log 'runs\_cicli\2026-11-19_b.log'
cd /d "%~dp0..\.."

echo [catena B] 1/4 tokenizer senza normalizzazione L0, 4.800 passi
.venv\Scripts\python.exe -u -I tokenizer\train_vqvae.py --run auto train.steps=4800 train.micro_batch=256 train.accum=1 train.warmup=240 train.eval_every=1600 train.log_every=50 train.ckpt_every_minutes=5 tokenizer.rimedi.init_dati=true tokenizer.rimedi.riavvio_fino_al_passo=1200 tokenizer.norma_livello0=false
if errorlevel 1 goto fermata

echo [catena B] 2/4 generatore sopra il tokenizer senza normalizzazione
.venv\Scripts\python.exe -u -I scripts\ciclo.py configs\cicli\2026-11-19_b1-norma-l0.yaml
if errorlevel 1 goto fermata

echo [catena B] 3/4 tokenizer lungo, 38.400 passi, copie ai passi 9.600 e 19.200
.venv\Scripts\python.exe -u -I tokenizer\train_vqvae.py --run auto train.steps=38400 train.micro_batch=256 train.accum=1 train.warmup=1920 train.eval_every=4800 train.log_every=50 train.ckpt_every_minutes=5 tokenizer.rimedi.init_dati=true tokenizer.rimedi.riavvio_fino_al_passo=1200 export.at_steps=[9600,19200]
if errorlevel 1 goto fermata

echo [catena B] 4/4 generatori sopra il tokenizer lungo e le sue copie
.venv\Scripts\python.exe -u -I scripts\ciclo.py configs\cicli\2026-11-19_b2-training-lungo.yaml
if errorlevel 1 goto fermata

echo [catena B] completata
exit /b 0

:fermata
echo [catena B] fermata: un pezzo non e' stato completato
exit /b 1
