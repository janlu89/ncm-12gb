#!/usr/bin/env bash
# Linux: elenca i training avviati con avvia_detached.sh o dal pulsante della dashboard (unita' ncm-*),
# e l'uso della GPU. Solo lettura.
systemctl --user list-units --type=service --all --no-legend 'ncm-*' | grep -v 'ncm-dashboard' || echo "nessun training attivo"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
