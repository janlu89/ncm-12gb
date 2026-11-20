"""Sola lettura: confronta token grezzi, vocabolario e decodifica."""
import os, sys
import numpy as np
from omegaconf import OmegaConf

root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "third_party", "ncm"))
from tokenization.bpe_tokenizer import BPETokenizer

art = os.path.join(root, "artifacts", "tinystories")
cfg = OmegaConf.load(os.path.join(art, "config.yaml")).tokenizer.pre_tokenizer
cfg.vocab_file_path = os.path.join(art, "bpe_vocab_8192.pickle")
cfg.merges_file_path = os.path.join(art, "bpe_merges_8192.pickle")
tok = BPETokenizer.from_config(cfg)
print("tipo vocab:", type(tok.vocab).__name__, "| voci:", len(tok.vocab))
items = list(tok.vocab.items())[:5] if isinstance(tok.vocab, dict) else list(enumerate(tok.vocab))[:5]
print("prime voci:", items)

train = np.memmap(os.path.join(root, "data", "raw", "tinystories", "train_bpe_8192.bin"), dtype=np.uint16, mode="r")
ids = [int(x) for x in train[:40]]
print("\nprimi 40 id:", ids)
inv = tok.vocab if isinstance(tok.vocab, dict) else dict(enumerate(tok.vocab))
print("token per token:", [inv.get(i) for i in ids])

sample = "Once upon a time, there was a little boy named Ben. He saw the vase."
enc = tok.encode(sample)
print("\nencode di una frase di prova:", enc)
print("voci:", [inv.get(i) for i in enc])
print("decode:", tok.decode(enc))

# parole frequenti: compaiono come token?
targets = [b" a", b" the", b" was", b" there", b" is", b"a", b"the", b"was"]
vals = {v: k for k, v in inv.items()} if isinstance(next(iter(inv.values())), (bytes, str)) else {}
for t in targets:
    key = t if any(isinstance(v, bytes) for v in list(inv.values())[:300]) else t.decode()
    print(repr(t), "->", vals.get(key))
