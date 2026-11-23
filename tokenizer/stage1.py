"""Stadio 1 (tokenizer VQ-VAE nostro): costruzione del modello, loss, valutazione ed export.
Usato da tokenizer/train_vqvae.py e dai test.

    s = build(config)       # config = OmegaConf della run (configs/stage1_tokenizer.yaml + modifiche)
    s.net, s.vq, s.levels, s.loss_fn, s.eval_fn, s.dopo_passo, s.export(model, run_dir)

Il modello usa le classi upstream (importate da third_party/ncm, non modificate) con il nostro quantizzatore
(tokenizer/quantizer.py). L'export è nel formato upstream: si carica nel codice ufficiale così com'è.
Vedi docs/piani/tokenizer-operativo.md.
"""
import copy
import json
import os
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from common import PROJECT_ROOT
from common.data import PAD_ID, load_valid_rows
from models.vqvae import VQVAE
from tokenizer.livelli import livelli, posizioni_generatore
from tokenizer.quantizer import NOME_NOSTRO, NOME_UPSTREAM


def P(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def config_vqvae(config):
    """Sezione `tokenizer.vqvae` nel formato upstream per il tokenizer descritto da `config`."""
    base = OmegaConf.load(P(config.base_config))
    t = config.tokenizer
    lv = livelli(int(t.livelli), str(t.schema), int(config.data.length))
    vq_cfg = copy.deepcopy(base.tokenizer.vqvae)
    vq_cfg.context_length = int(config.data.length)
    vq_cfg.vqvae_vocab_size = [int(t.codebook)] * len(lv)
    vq_cfg.checkpoint_path = None
    q = vq_cfg.vector_quantizer_config
    q.clss = NOME_UPSTREAM
    q.levels = {"use_manual_levels": True, "manual_levels": lv}
    q.learned_l1_cascade_norm_final = bool(t.norma_livello0)
    if t.get("vqvae_overrides"):
        vq_cfg = OmegaConf.merge(vq_cfg, t.vqvae_overrides)
    return base, vq_cfg, lv


def build(config, dev="cuda"):
    s = SimpleNamespace(config=config)
    base, vq_cfg, levels = config_vqvae(config)
    s.base, s.vq_cfg, s.levels = base, vq_cfg, levels
    K, L = len(levels), int(config.data.length)

    train_cfg = copy.deepcopy(vq_cfg)
    train_cfg.vector_quantizer_config.clss = NOME_NOSTRO
    rim = config.tokenizer.get("rimedi") or {}
    # Le opzioni dei rimedi vanno solo al quantizzatore nostro (training), non al config esportato (upstream).
    train_cfg.vector_quantizer_config.riavvio_fino_al_passo = int(rim.get("riavvio_fino_al_passo", 0))
    train_cfg.vector_quantizer_config.riavvio_soglia = float(rim.get("riavvio_soglia", 0.1))
    train_cfg.vector_quantizer_config.riavvio_soglia_relativa = bool(rim.get("riavvio_soglia_relativa", False))
    torch.manual_seed(int(config.train.seed))
    net = VQVAE(**train_cfg).to(dev)
    vq = net.vq
    assert list(vq.levels) == levels
    s.net, s.vq = net, vq
    print(f"[tokenizer] {K} livelli ({config.tokenizer.schema}), {sum(levels)} codici per storia, "
          f"{posizioni_generatore(levels)} posizioni per il generatore: {levels}", flush=True)

    # ---------- loss: ricostruzione (padding incluso) + termini VQ ----------
    def loss_fn(model, x):
        logits, vq_loss, *_ = model(x)
        ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x.reshape(-1))
        return ce + vq_loss, {"ce": ce.detach(), "vq_loss": vq_loss.detach()}

    def dopo_passo(_step):
        vq.ema_step()

    @torch.no_grad()
    def init_dai_dati(n_storie=None):
        """ R1: codebook dai dati, su storie del train prese con un seed diverso da quello del training. """
        from common.data import TrainStories
        n_storie = int(n_storie or rim.get("init_storie", 8192))
        ds = TrainStories(P(config.data.dir), L, seed=int(config.data.seed) + 777)
        era = net.training
        net.eval()
        z = torch.cat([net._encode_pre_quant(ds.next_batch(256).to(dev)).float() for _ in range(n_storie // 256)])
        vq.init_dai_dati(z)
        if era:
            net.train()
        print(f"[tokenizer] codebook inizializzati dai dati ({n_storie} storie)", flush=True)

    s.loss_fn, s.dopo_passo, s.init_dai_dati = loss_fn, dopo_passo, init_dai_dati

    # ---------- valutazione: ricostruzione delle storie di validazione, FP32 ----------
    valid_rows = load_valid_rows(P(config.data.dir), L, int(config.eval.n_stories))
    s.valid_rows = valid_rows

    @torch.no_grad()
    def eval_fn(model):
        v = model.vq
        ok = torch.zeros(K, dtype=torch.float64, device=dev)
        ok_nopad = torch.zeros(K, dtype=torch.float64, device=dev)
        seq_ok = torch.zeros(K, dtype=torch.float64, device=dev)
        ce_sum = torch.zeros((), dtype=torch.float64, device=dev)
        usati = [torch.zeros(vs, dtype=torch.bool, device=dev) for vs in v.vqvae_vocab_sizes]
        n_tok = n_nopad = 0
        bs = int(config.eval.batch)
        with torch.autocast("cuda", enabled=False):
            for b in range(0, len(valid_rows), bs):
                x = valid_rows[b:b + bs].to(dev)
                B = x.shape[0]
                idx = model.encode(x).view(B, -1)
                nopad = x != PAD_ID
                z_acc = torch.zeros(B, L, model.embed_dim, device=dev)
                o = 0
                for li, lv in enumerate(levels):
                    codes = idx[:, o:o + lv]
                    usati[li][codes.reshape(-1)] = True
                    emb = v.embeddings[li](codes)
                    q = v._up_to_full(emb, li, L) if li != K - 1 else emb
                    z_acc = z_acc + v._apply_quant_resi(q, li)
                    logits = model.decoder(z_acc).float()
                    good = logits.argmax(-1) == x
                    ok[li] += good.sum()
                    ok_nopad[li] += (good & nopad).sum()
                    seq_ok[li] += good.all(dim=1).sum()
                    if li == K - 1:
                        ce_sum += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x.reshape(-1),
                                                  reduction="sum").double()
                    o += lv
                n_tok += x.numel()
                n_nopad += int(nopad.sum())
        norme = [v.embeddings[i].weight.float().norm(dim=-1) for i in range(K)]
        n = len(valid_rows)
        return {
            "val_loss": float(ce_sum / n_tok),                       # CE di ricostruzione, tutti i livelli
            "acc_token": float(ok[-1] / n_tok),
            "acc_token_senza_pad": float(ok_nopad[-1] / n_nopad),
            "sequenze_perfette": float(seq_ok[-1] / n),
            "norma_codebook_max": float(max(x.max() for x in norme)),
            "norma_codebook_l0_mediana": float(norme[0].median()),
            "codici_usati_media": float(sum(u.float().mean() for u in usati) / K),
            "acc_per_livello": [round(float(a / n_tok), 5) for a in ok],          # usando i primi k livelli
            "codici_usati_per_livello": [round(float(u.float().mean()), 4) for u in usati],
        }

    s.eval_fn = eval_fn

    @torch.no_grad()
    def stabilita(model, n=1024):
        """Frazione di codici che cambia codificando le stesse storie a batch 256 e a batch 16."""
        x = valid_rows[:n].to(dev)
        with torch.autocast("cuda", enabled=False):
            a = torch.cat([model.encode(x[i:i + 256]).view(-1, sum(levels)) for i in range(0, n, 256)])
            b = torch.cat([model.encode(x[i:i + 16]).view(-1, sum(levels)) for i in range(0, n, 16)])
        return float((a != b).float().mean())

    s.stabilita = stabilita

    # ---------- export nel formato upstream ----------
    def config_export():
        cfg = copy.deepcopy(base)
        art = os.path.dirname(P(config.base_config))
        cfg.tokenizer.vqvae = copy.deepcopy(vq_cfg)
        cfg.tokenizer.vqvae.checkpoint_path = "tokenizer.pt"
        cfg.model.vocab_size = [int(config.tokenizer.codebook)] * (K - 1)
        cfg.checkpoints.tokenizer = "tokenizer.pt"
        # Percorsi assoluti: os.path.join(cartella, percorso assoluto) restituisce il percorso assoluto.
        cfg.checkpoints.bpe_vocab = os.path.join(art, os.path.basename(str(base.checkpoints.bpe_vocab)))
        cfg.checkpoints.bpe_merges = os.path.join(art, os.path.basename(str(base.checkpoints.bpe_merges)))
        return cfg

    @torch.no_grad()
    def export(model, run_dir, sub=None, passo=None):
        """Scrive tokenizer.pt + config.yaml + export.json. Verifica che il file si carichi nel codice upstream
        e dia gli stessi codici."""
        from tokenization.vqvae_tokenizer import VQVAETokenizer
        out_dir = os.path.join(run_dir, "export", sub) if sub else os.path.join(run_dir, "export")
        os.makedirs(out_dir, exist_ok=True)
        era_training = model.training
        model.eval()
        tmp = os.path.join(out_dir, "tokenizer.pt.tmp")
        torch.save({"model": model.state_dict()}, tmp)
        os.replace(tmp, os.path.join(out_dir, "tokenizer.pt"))
        cfg = config_export()
        OmegaConf.save(cfg, os.path.join(out_dir, "config.yaml"))

        chk = copy.deepcopy(cfg.tokenizer)
        chk.vqvae.checkpoint_path = os.path.join(out_dir, "tokenizer.pt")
        chk.pre_tokenizer.vocab_file_path = str(cfg.checkpoints.bpe_vocab)
        chk.pre_tokenizer.merges_file_path = str(cfg.checkpoints.bpe_merges)
        up = VQVAETokenizer.from_config(chk).vqvae.to(dev).eval()      # strict=True dentro from_config
        x = valid_rows[:64].to(dev)   # batch piccolo: la codifica upstream con molti livelli occupa gigabyte
        with torch.autocast("cuda", enabled=False):
            uguali = float((up.encode(x) == model.encode(x)).float().mean())
        del up
        m = eval_fn(model)
        info = {"passo": passo, "data": time.strftime("%Y-%m-%d %H:%M:%S"), "livelli": levels,
                "schema": str(config.tokenizer.schema), "codici_per_storia": sum(levels),
                "posizioni_generatore": posizioni_generatore(levels),
                "codici_uguali_a_upstream": uguali, "codici_instabili_batch256_vs_16": stabilita(model), **m}
        with open(os.path.join(out_dir, "export.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        if era_training:
            model.train()
        print(f"[export] {out_dir}: acc_token {m['acc_token']:.5f}, sequenze perfette {m['sequenze_perfette']:.4f}, "
              f"codici uguali a upstream {uguali:.6f}, instabili {info['codici_instabili_batch256_vs_16']:.5f}",
              flush=True)
        return info

    s.export, s.config_export = export, config_export
    return s
