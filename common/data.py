"""Dati TinyStories tokenizzati: righe di lunghezza fissa e campionamento riproducibile e riprendibile.

Formato dei file (vedi docs/riferimenti/dati-tinystories.md): token uint16, una storia termina con EOS;
il train ha un file di offset (int64), il validation si separa con EOS.
"""
import os

import numpy as np
import torch

EOS_ID = 0
PAD_ID = 1


def _to_rows(stream, starts, ends, length, pad_id):
    rows = np.full((len(starts), length), pad_id, dtype=np.int64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        seg = stream[s:min(e, s + length)]
        rows[i, :len(seg)] = seg
    return rows


class TrainStories:
    """Campiona storie del train in ordine casuale, un'epoca alla volta, senza ripetizioni.

    L'ordine dipende solo da (seed, epoca): lo stato da salvare è {"epoch", "pos"}.
    Ogni storia viene troncata o completata con PAD fino a `length` token.
    """

    def __init__(self, data_dir, length, seed, pad_id=PAD_ID, prefix="train_bpe_8192"):
        self.stream = np.memmap(os.path.join(data_dir, f"{prefix}.bin"), dtype=np.uint16, mode="r")
        off = np.load(os.path.join(data_dir, f"{prefix}_offsets.npy"), allow_pickle=False)
        self.starts = off if off[-1] < len(self.stream) else off[:-1]
        self.ends = np.append(self.starts[1:], len(self.stream))
        self.n = len(self.starts)
        self.length, self.seed, self.pad_id = int(length), int(seed), int(pad_id)
        self.epoch, self.pos = 0, 0
        self._perm = self._make_perm(self.epoch)

    def _make_perm(self, epoch):
        return np.random.default_rng([self.seed, epoch]).permutation(self.n)

    def next_batch(self, batch_size):
        out = []
        need = batch_size
        while need > 0:
            take = min(need, self.n - self.pos)
            out.append(self._perm[self.pos:self.pos + take])
            self.pos += take
            need -= take
            if self.pos == self.n:
                self.epoch += 1
                self.pos = 0
                self._perm = self._make_perm(self.epoch)
        idx = np.concatenate(out)
        rows = _to_rows(self.stream, self.starts[idx], self.ends[idx], self.length, self.pad_id)
        return torch.from_numpy(rows)

    def state_dict(self):
        return {"epoch": self.epoch, "pos": self.pos, "seed": self.seed, "n": self.n, "length": self.length}

    def load_state_dict(self, st):
        assert st["seed"] == self.seed and st["n"] == self.n and st["length"] == self.length, \
            f"stato dati incompatibile: {st} contro seed={self.seed} n={self.n} length={self.length}"
        self.epoch, self.pos = int(st["epoch"]), int(st["pos"])
        self._perm = self._make_perm(self.epoch)


class TrainCodes:
    """Come TrainStories, ma restituisce i codici VQ pre-calcolati (data/build_codes.py) invece dei token.

    Stessa permutazione per (seed, epoca) e stesso stato {"epoch", "pos"}: con lo stesso seed le storie
    arrivano nello stesso ordine di TrainStories.
    """

    def __init__(self, codes_path, seed):
        meta = os.path.splitext(codes_path)[0] + ".json"
        import json
        with open(meta, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.n, self.width = int(self.meta["rows"]), int(self.meta["codes_per_row"])
        self.codes = np.memmap(codes_path, dtype=np.uint16, mode="r", shape=(self.n, self.width))
        self.seed, self.length = int(seed), int(self.meta["length"])
        self.epoch, self.pos = 0, 0
        self._perm = self._make_perm(self.epoch)

    _make_perm = TrainStories._make_perm

    def next_batch(self, batch_size):
        out = []
        need = batch_size
        while need > 0:
            take = min(need, self.n - self.pos)
            out.append(self._perm[self.pos:self.pos + take])
            self.pos += take
            need -= take
            if self.pos == self.n:
                self.epoch += 1
                self.pos = 0
                self._perm = self._make_perm(self.epoch)
        idx = np.sort(np.concatenate(out))           # lettura ordinata dal disco ...
        order = np.argsort(np.argsort(np.concatenate(out)))
        rows = self.codes[idx][order]                # ... poi riportata all'ordine della permutazione
        return torch.from_numpy(rows.astype(np.int64))

    def state_dict(self):
        return {"epoch": self.epoch, "pos": self.pos, "seed": self.seed, "n": self.n, "length": self.length}

    load_state_dict = TrainStories.load_state_dict


def load_valid_rows(data_dir, length, max_stories=None, eos_id=EOS_ID, pad_id=PAD_ID,
                    prefix="valid_bpe_8192"):
    """Le prime `max_stories` storie del validation (separate con EOS), come tensore [N, length]."""
    stream = np.fromfile(os.path.join(data_dir, f"{prefix}.bin"), dtype=np.uint16)
    ends = np.flatnonzero(stream == eos_id) + 1
    starts = np.concatenate([[0], ends[:-1]])
    if max_stories is not None:
        starts, ends = starts[:max_stories], ends[:max_stories]
    return torch.from_numpy(_to_rows(stream, starts, ends, length, pad_id))
