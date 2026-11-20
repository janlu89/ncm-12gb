"""Stadio 2 (generatore NCM sopra un tokenizer VQ-VAE congelato): costruzione di modello, loss,
valutazione ed export. Usato da generator/train_generator.py e dai benchmark.

    s = build(config)       # config = OmegaConf della run (configs/stage2_generator.yaml + modifiche)
    s.net, s.loss_fn, s.eval_fn, s.encode, s.inputs_from_idx, s.export(model, run_dir)
"""
import json
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from common import PROJECT_ROOT
from common.data import EOS_ID, PAD_ID, TrainStories, load_valid_rows
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer


def P(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def build(config, dev="cuda"):
    s = SimpleNamespace(config=config)
    base = OmegaConf.load(P(config.base_config))
    if config.get("model_overrides"):
        # Modifiche all'architettura del generatore (es. use_flex_attention). Ogni override cambia il modello:
        # va dichiarato nell'esperimento.
        base.model = OmegaConf.merge(base.model, config.model_overrides)
    s.base = base

    # ---------- tokenizer congelato ----------
    art = os.path.dirname(P(config.base_config))
    tok_cfg = base.tokenizer
    tok_cfg.vqvae.checkpoint_path = (os.path.join(art, base.checkpoints.tokenizer)
                                     if config.tokenizer_checkpoint == "official" else P(config.tokenizer_checkpoint))
    tok_cfg.pre_tokenizer.vocab_file_path = os.path.join(art, base.checkpoints.bpe_vocab)
    tok_cfg.pre_tokenizer.merges_file_path = os.path.join(art, base.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(tok_cfg)
    vqvae = tk.vqvae.to(dev).eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    vq = vqvae.vq
    levels = list(vq.levels)
    L0 = levels[0]
    assert tk.pre_tokenizer.eos_token_id == EOS_ID and tk.pre_tokenizer.pad_token_id == PAD_ID
    s.tk, s.vqvae, s.vq, s.levels = tk, vqvae, vq, levels

    # ---------- generatore (pesi nuovi; dopo il tokenizer: copia i codebook) ----------
    torch.manual_seed(int(config.train.seed))
    net = NextConceptTransformer(vqvae_local=vqvae, config=base.model).to(dev)
    Pn = int(base.model.prefix_len)
    assert bool(base.model.raw_token_prefix) and not bool(base.model.predict_level0) \
        and not bool(base.model.query_level0)
    s.net = net

    # ---------- loss e corruzione ----------
    tgt_levels = levels[1:]                   # livelli predetti: 1..K-1
    seg, o = [], 0
    for lv in tgt_levels:
        seg.append((o, o + lv))
        o += lv
    s.seg = seg
    alpha = float(config.loss.alpha)
    den = sum(lv ** (1 - alpha) for lv in tgt_levels)
    w_per_code = torch.cat([torch.full((lv,), lv ** (-alpha) / den) for lv in tgt_levels]).to(dev)

    in_levels = list(range(len(levels) - 1))  # livelli i cui codici formano l'input (0..K-2)
    corr_levels = [k for k in in_levels if not (config.corruption.skip_level0 and k == 0)]
    rates = torch.zeros(len(levels))
    if config.corruption.enabled and corr_levels:
        n_c = len(corr_levels)
        for j, k in enumerate(corr_levels):
            t = j / max(n_c - 1, 1)
            rates[k] = config.corruption.rate_first + t * (config.corruption.rate_last - config.corruption.rate_first)
    rate_per_code = torch.cat([torch.full((lv,), float(rates[k])) for k, lv in enumerate(levels)]).to(dev)
    vocab_per_code = torch.cat([torch.full((lv,), vq.vqvae_vocab_sizes[k]) for k, lv in enumerate(levels)]).to(dev)
    corr_code_mask = torch.cat([torch.full((lv,), k in corr_levels) for k, lv in enumerate(levels)]).to(dev)
    corr_mode = str(config.corruption.get("mode", "linear"))
    assert corr_mode in ("linear", "per_example_uniform"), f"corruption.mode sconosciuto: {corr_mode}"
    rate_max = float(config.corruption.get("rate_max", 1.0))
    s.rates = rates
    if corr_mode == "linear":
        print("[corruzione] lineare, tasso per livello:", [round(float(r), 3) for r in rates], flush=True)
    else:
        print(f"[corruzione] per esempio, tasso ~ U[0, {rate_max}] sui livelli {corr_levels}", flush=True)

    # encode_fp64: i codici si calcolano a 64 bit con una copia del tokenizer (codici "esatti", indipendenti dal
    # batch). Tutto il resto (input del generatore, teste, decodifica) resta a 32 bit come sempre.
    enc64 = bool(config.get("encode_fp64", False))
    if enc64:
        import copy
        from tokenizer.codifica64 import codifica_fp64
        vqvae64 = copy.deepcopy(vqvae).double()
        print("[tokenizer] codifica a 64 bit attiva", flush=True)

    @torch.no_grad()
    def encode(x):
        with torch.autocast("cuda", enabled=False):
            if enc64:
                return codifica_fp64(vqvae64, x)
            return vqvae.encode(x).view(x.shape[0], -1)

    @torch.no_grad()
    def inputs_from_idx(idx):
        B = idx.shape[0]
        with torch.autocast("cuda", enabled=False):
            z_prefix = vqvae.encoder.wte(torch.full((B, Pn), EOS_ID, device=idx.device))
            ncm_in = vq.idx_to_ncm_input(idx)
        return torch.cat([z_prefix, ncm_in], dim=1)

    def corrupt(idx):
        if not config.corruption.enabled:
            return idx
        if corr_mode == "linear":
            rate = rate_per_code
        else:  # un tasso per esempio, uguale su tutti i livelli corrotti (come Infinity / config LM1B upstream)
            rate = torch.rand(idx.shape[0], 1, device=idx.device) * rate_max * corr_code_mask
        mask = torch.rand(idx.shape, device=idx.device) < rate
        rnd = (torch.rand(idx.shape, device=idx.device) * vocab_per_code).long()
        return torch.where(mask, rnd, idx)

    def per_code_nll(model, x_in, target):
        logits = model(x_in).float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                               reduction="none").view(target.shape), logits

    def loss_from_idx(model, idx):
        nll, _ = per_code_nll(model, inputs_from_idx(corrupt(idx)), idx[:, L0:])
        loss = (nll * w_per_code).sum(1).mean()
        # Metriche come tensori sulla GPU: il trainer le legge solo quando scrive il log (niente sincronizzazioni).
        nd = nll.detach()
        m = {"nll_media": nd.mean()}
        for name, j in (("nll_L1", 0), ("nll_L6", 5), ("nll_L11", len(seg) - 1)):
            a, b = seg[j]
            m[name] = nd[:, a:b].mean()
        return loss, m

    def loss_fn(model, tokens):
        return loss_from_idx(model, encode(tokens))

    s.encode, s.inputs_from_idx, s.corrupt = encode, inputs_from_idx, corrupt
    s.per_code_nll, s.loss_from_idx, s.loss_fn = per_code_nll, loss_from_idx, loss_fn

    # ---------- valutazione (teacher forcing pulito, FP32, stessa metrica del riferimento) ----------
    valid_rows = load_valid_rows(P(config.data.dir), int(config.data.length), int(config.eval.n_stories))
    # Batch di codifica: 256 con il tokenizer ufficiale (i suoi codici dipendono dal batch: va tenuto fisso);
    # con tokenizer a molti livelli serve più piccolo (la codifica upstream tiene in memoria tutti i livelli).
    eb = int(config.eval.get("encode_batch", 256))
    valid_idx = torch.cat([encode(valid_rows[i:i + eb].to(dev)) for i in range(0, len(valid_rows), eb)])

    def eval_fn(model):
        tot = torch.zeros(len(seg), dtype=torch.float64)
        ok = torch.zeros(len(seg), dtype=torch.float64)
        bs = int(config.eval.batch)
        for i in range(0, len(valid_idx), bs):
            idx = valid_idx[i:i + bs]
            with torch.autocast("cuda", enabled=False):
                nll, logits = per_code_nll(model, inputs_from_idx(idx), idx[:, L0:])
            pred = logits.argmax(-1)
            tgt = idx[:, L0:]
            for j, (a, b) in enumerate(seg):
                tot[j] += nll[:, a:b].sum().item()
                ok[j] += (pred[:, a:b] == tgt[:, a:b]).sum().item()
        n = len(valid_idx)
        counts = torch.tensor([n * (b - a) for a, b in seg], dtype=torch.float64)
        m = {"val_loss": float(tot.sum() / counts.sum())}
        for j in (0, 5, len(seg) - 1):
            m[f"val_L{j + 1}"] = float(tot[j] / counts[j])
            m[f"val_acc_L{j + 1}"] = float(ok[j] / counts[j])
        m["val_per_livello"] = [round(float(tot[j] / counts[j]), 4) for j in range(len(seg))]
        return m

    s.eval_fn = eval_fn

    # ---------- export: file compatibile con eval_shard.py + storie di prova ----------
    @torch.no_grad()
    def export(model, run_dir):
        out_dir = os.path.join(run_dir, "export")
        os.makedirs(out_dir, exist_ok=True)
        model.eval()
        torch.cuda.empty_cache()  # libera la memoria lasciata dal training
        # Unigramma del livello 0 contato su sequenze di train (seed diverso da quello del training).
        ds = TrainStories(P(config.data.dir), int(config.data.length), seed=int(config.data.seed) + 999)
        counts = torch.zeros(vq.vqvae_vocab_sizes[0], dtype=torch.float64, device=dev)
        n_seq, done = int(config.export.level0_sequences), 0
        while done < n_seq:
            b = min(128, n_seq - done)
            idx = encode(ds.next_batch(b).to(dev))
            counts += torch.bincount(idx[:, :L0].reshape(-1), minlength=counts.numel()).double()
            done += b
        unigram = (counts / counts.sum()).float().cpu()
        torch.save({"model": model.state_dict(), "inference": {"level0_unigram": unigram}},
                   os.path.join(out_dir, "generator.pt"))
        g = base.generation
        n_s = int(config.export.samples)
        torch.manual_seed(0)
        l0 = torch.multinomial(unigram, n_s * L0, replacement=True).view(n_s, L0).to(dev)
        gen = model.generate(l0, torch.full((n_s, Pn), EOS_ID, device=dev), temperature=float(g.temperature),
                             top_k=int(g.top_k), top_p=float(g.top_p))
        _, zl = vq.idx_to_zhats(gen, return_zlevels=True)
        ids = vqvae.decoder(zl[-1]).argmax(-1).tolist()
        texts = []
        for r in ids:
            if EOS_ID in r:
                r = r[:r.index(EOS_ID)]
            texts.append(tk.pre_tokenizer.decode([t for t in r if t not in (EOS_ID, PAD_ID)],
                                                 skip_special_tokens=True))
        with open(os.path.join(out_dir, "storie_di_prova.txt"), "w", encoding="utf-8") as f:
            for i, t in enumerate(texts):
                f.write(f"===== storia {i + 1} =====\n{t}\n\n")
        with open(os.path.join(out_dir, "export.json"), "w", encoding="utf-8") as f:
            json.dump({"level0_sequences": n_seq, "level0_codici_usati": int((unigram > 0).sum()),
                       "samples": n_s}, f, indent=2)
        print(f"[export] {out_dir}: generator.pt, storie_di_prova.txt", flush=True)

    s.export = export
    return s
