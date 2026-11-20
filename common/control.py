"""Pausa pulita del training e blocco della sospensione di Windows.

Il training si ferma alla fine del passo in corso, salva e termina quando:
- si preme Ctrl+C una volta (una seconda pressione interrompe subito, senza salvare);
- esiste il file STOP nella cartella del run (il file viene rimosso alla ripresa).
"""
import ctypes
import os
import signal
import sys


class StopControl:
    def __init__(self, run_dir):
        self.stop_file = os.path.join(run_dir, "STOP")
        self.requested = False
        self.reason = None
        self._old = {}

    def __enter__(self):
        if os.path.exists(self.stop_file):
            os.remove(self.stop_file)
        for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", None)):
            if sig is not None:
                self._old[sig] = signal.signal(sig, self._handler)
        _prevent_sleep(True)
        return self

    def __exit__(self, *exc):
        for sig, old in self._old.items():
            signal.signal(sig, old)
        _prevent_sleep(False)
        return False

    def _handler(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested, self.reason = True, f"segnale {signum}"
        print("\n[stop] richiesta pausa: salvo alla fine del passo in corso "
              "(premi di nuovo Ctrl+C per uscire subito senza salvare)", flush=True)

    def should_stop(self):
        if not self.requested and os.path.exists(self.stop_file):
            self.requested, self.reason = True, "file STOP"
        return self.requested

    def request(self, reason):
        self.requested, self.reason = True, reason


def _prevent_sleep(enable):
    if sys.platform != "win32":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if enable else ES_CONTINUOUS
    ctypes.windll.kernel32.SetThreadExecutionState(flags)
