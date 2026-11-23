#!/usr/bin/env bash
# Versione Linux di avvia_detached.ps1 (che resta per Windows).
# Avvia un comando staccato dalla sessione come unita' systemd utente: continua anche se il terminale,
# o la dashboard si chiudono (linger attivo). L'output va in coda al file di log.
#
# Uso:   scripts/avvia_detached.sh '<comando>' <log> [nome]
# Esempi:
#   scripts/avvia_detached.sh 'python -u -I generator/train_generator.py --run prova' runs/prova/console.log prova
#   scripts/avvia_detached.sh 'python -u -I scripts/ciclo.py configs/cicli/x.yaml && python -u -I scripts/ciclo.py configs/cicli/y.yaml' runs/_cicli/catena.log catena
# "python" e' quello di ~/venv/ncm (messo in testa al PATH). Catene con &&.
# Controllo:  scripts/run_attivi.sh      Fermare un training: file runs/<nome>/STOP (salva e si ferma).
set -euo pipefail
if [ $# -lt 2 ]; then sed -n '2,12p' "$0"; exit 1; fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CMD="$1"; LOG="$2"; NOME="${3:-run}"
case "$LOG" in /*) ;; *) LOG="$ROOT/$LOG" ;; esac
mkdir -p "$(dirname "$LOG")"
VENV="${NCM_VENV:-$HOME/venv/ncm}"
[ -x "$VENV/bin/python" ] || { echo "manca $VENV/bin/python" >&2; exit 1; }
UNIT="ncm-$(echo "$NOME" | tr -c 'A-Za-z0-9_.\n-' '_')-$(date +%Y%m%d%H%M%S)"
systemd-run --user --collect --unit "$UNIT" --working-directory "$ROOT" \
  -E PYTHONIOENCODING=utf-8 -E PATH="$VENV/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
  -p StandardOutput="append:$LOG" -p StandardError="append:$LOG" \
  /bin/bash -c "$CMD"
echo "avviato: unita' $UNIT, log $LOG"
echo "stato: systemctl --user status $UNIT   |   output: tail -f '$LOG'"
