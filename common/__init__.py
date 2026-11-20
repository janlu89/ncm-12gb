"""Codice condiviso dai training del progetto (dati, checkpoint, pausa, log, ciclo di training)."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCM_ROOT = os.path.join(PROJECT_ROOT, "third_party", "ncm")
if NCM_ROOT not in sys.path:
    sys.path.insert(0, NCM_ROOT)
