"""Restart statistics of a tokenizer checkpoint, per level (no training).

Reads the EMA usage statistics `ema_cluster_size_<i>` of each level and prints: vectors seen per step, the mean
statistic per code if usage were uniform, and how many codes fall below the fixed restart threshold 0.1.
Used in EXPERIMENTS.md section 17.

    python -I evaluation/soglia_riavvio_per_livello.py runs/<tokenizer run>/checkpoints/step_<N>.pt
"""
import sys, torch, json
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = ck.get("model", ck)
if not isinstance(sd, dict): sd = sd.state_dict()
keys = [k for k in sd if "ema_cluster_size_" in k]
print("chiavi top:", list(ck.keys())[:10] if isinstance(ck, dict) else type(ck))
for k in sorted(keys, key=lambda x: int(x.rsplit("_",1)[1])):
    cs = sd[k].float(); i = int(k.rsplit("_",1)[1])
    n = cs.sum().item(); V = cs.numel()
    print(f"L{i:2d} V={V:5d} somma={n:9.1f} atteso/codice={n/V:7.3f} <0.1: {(cs<0.1).sum().item():5d}  >=1: {(cs>=1).sum().item():5d}  max={cs.max().item():.1f}")
