"""Cartella del run: metadati, log JSONL e telemetria della GPU."""
import json
import os
import platform
import subprocess
import sys
import time

import torch

from . import NCM_ROOT, PROJECT_ROOT


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"errore: {e!r}"


def write_metadata(run_dir, config, argv):
    """Alla prima partenza: configurazione, versioni, commit NCM, pip freeze."""
    meta = {
        "creato": time.strftime("%Y-%m-%d %H:%M:%S"),
        "comando": [sys.executable] + list(argv),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "ncm_commit": _cmd(["git", "-C", NCM_ROOT, "rev-parse", "HEAD"]),
        "piattaforma": platform.platform(),
        "config": config,
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(run_dir, "pip-freeze.txt"), "w", encoding="utf-8") as f:
        f.write(_cmd([sys.executable, "-m", "pip", "freeze"]))


def gpu_telemetry():
    out = _cmd(["nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"])
    try:
        t, p, c, u, m = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"gpu_temp_c": int(t), "gpu_power_w": float(p), "gpu_clock_mhz": int(c),
                "gpu_util_pct": int(u), "gpu_mem_mib": int(m)}
    except Exception:
        return {"gpu_telemetry": out}


class JsonlLog:
    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, "log.jsonl")

    def write(self, rec):
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
