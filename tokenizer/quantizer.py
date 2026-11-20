# Portions of this file are adapted from models/quant.py of https://github.com/HazyResearch/ncm
# (MIT License, Copyright (c) 2025 hermann kumbong). The upstream licence text is in third_party/NCM_LICENSE.
"""Quantizzatore del nostro tokenizer.

Sottoclasse del quantizzatore upstream: stessi pesi e stessi nomi (un tokenizer addestrato qui si carica nel
codice ufficiale senza modifiche), ma con un percorso di training nostro:

- i codebook si aggiornano una volta per passo, con le statistiche sommate su tutti i micro-batch
  (`ema_step()`), come un batch unico; nel codice upstream l'aggiornamento avviene a ogni forward;
- le statistiche EMA partono da 1 (con 0 i codici non usati al primo passo vengono divisi per ~1e-5);
- tutto il quantizzatore lavora in fp32 anche dentro l'autocast;
- statistiche con `index_add` invece della matrice one-hot (stessa matematica, meno memoria);
- nested dropout come nel config ufficiale (`aux.fine_drop`): con probabilità `prob` il decoder riceve
  la ricostruzione fermata a un livello scelto a caso.

La codifica (`z_to_idx`) e la decodifica restano quelle upstream.
Vedi docs/piani/tokenizer-operativo.md.
"""
import torch
import torch.nn.functional as F

from common import NCM_ROOT  # noqa: F401  (mette third_party/ncm nel percorso di import)
from models.quant import MultiscaleResidualVectorQuantizer
from utils.registry import vector_quantizers

NOME_UPSTREAM = "multiscale_residual_vector_quantizer"
NOME_NOSTRO = "quantizzatore_nostro"


class QuantizzatoreNostro(MultiscaleResidualVectorQuantizer):
    def __init__(self, *args, riavvio_fino_al_passo=0, riavvio_soglia=0.1, riavvio_campioni=4096, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.restart_threshold == 0, "riavvio upstream disattivato: si usa riavvio_fino_al_passo"
        # Rimedio R2 contro il collasso dei codici: fino al passo indicato, i codici quasi inutilizzati
        # (statistica EMA sotto la soglia) vengono rimpiazzati con input veri del passo. Dopo, mai più
        # (gli autori sconsigliano il riavvio per tutto il training: peggiora la generazione).
        self.riavvio_fino_al_passo = int(riavvio_fino_al_passo)
        self.riavvio_soglia = float(riavvio_soglia)
        self.riavvio_campioni = int(riavvio_campioni)
        self.register_buffer("_passo_ema", torch.zeros((), dtype=torch.long), persistent=False)
        self._campioni = [None] * len(self.vqvae_vocab_sizes)
        for i, vs in enumerate(self.vqvae_vocab_sizes):
            getattr(self, f"ema_cluster_size_{i}").fill_(1.0)
            # Accumulatori del passo in corso: non persistenti, quindi fuori dallo state_dict esportato;
            # i checkpoint di training li salvano comunque (common/checkpoint.py).
            self.register_buffer(f"acc_n_{i}", torch.zeros(vs), persistent=False)
            self.register_buffer(f"acc_w_{i}", torch.zeros(vs, self.embed_dim), persistent=False)

    def _distances(self, flat_input, emb_w):
        with torch.autocast("cuda", enabled=False):
            x, e = flat_input.float(), emb_w.float()
            return (x * x).sum(1, keepdim=True) + (e * e).sum(1) - 2.0 * (x @ e.t())

    def z_to_idx(self, z, return_zhats=False, return_zlevels=False):
        with torch.autocast("cuda", enabled=False):
            if return_zhats or return_zlevels or getattr(self, "_stoch_encode", False):
                return super().z_to_idx(z.float(), return_zhats=return_zhats, return_zlevels=return_zlevels)
            # Stesse operazioni di upstream, senza le liste delle ricostruzioni intermedie: con molti
            # livelli occupano gigabyte (40 livelli, batch 256: ~3 GB) anche quando non servono.
            z = z.float()
            B, L, C = z.shape
            z_rest = z.detach().clone()
            n_livelli = len(self.levels)
            idx = []
            for i, level in enumerate(self.levels):
                emb_i = self.embeddings[i]
                flat = (self._down_for_quant(z_rest, i, level) if i != n_livelli - 1 else z_rest).reshape(-1, C)
                indices = torch.argmin(self._distances(flat, emb_i.weight), dim=1).view(B, level)
                idx.append(indices)
                q = self._up_to_full(emb_i(indices), i, L) if i != n_livelli - 1 else emb_i(indices)
                z_rest -= self._apply_quant_resi(q, i)
            return torch.cat(idx, dim=-1)

    def forward(self, z):
        with torch.autocast("cuda", enabled=False):
            return self._forward_fp32(z.float())

    def _taglio(self, n_livelli):
        """Indice dell'ultimo livello dato al decoder. n_livelli - 1 = nessun taglio."""
        p = self._current_fine_drop_prob()
        if self.training:
            self._fd_step += 1
        if self.training and p > 0.0 and float(torch.rand(())) < p:
            kmin, kmax = max(self.fine_drop_min_keep - 1, 0), n_livelli - 2
            if kmax >= kmin:
                return int(torch.randint(kmin, kmax + 1, ()))
        return n_livelli - 1

    def _forward_fp32(self, z):
        B, L, C = z.shape
        z_no_grad = z.detach()
        z_rest = z_no_grad.clone()
        z_hat = torch.zeros_like(z_rest)
        n_livelli = len(self.levels)
        taglio = self._taglio(n_livelli)
        z_hat_taglio = None
        vq_loss = 0

        for i, level in enumerate(self.levels):
            emb_i = self.embeddings[i]
            x_BLC = self._down_for_quant(z_rest, i, level) if i != n_livelli - 1 else z_rest
            flat = x_BLC.reshape(-1, C)
            indices = torch.argmin(self._distances(flat, emb_i.weight), dim=1)

            if self.training and self.riavvio_fino_al_passo > 0:
                with torch.no_grad():   # un campione degli input del passo, per il riavvio (R2)
                    sel = torch.randint(0, flat.shape[0], (min(self.riavvio_campioni, flat.shape[0]),), device=flat.device)
                    self._campioni[i] = flat.detach()[sel]
            if self.training:
                with torch.no_grad():
                    getattr(self, f"acc_n_{i}").index_add_(
                        0, indices, torch.ones(indices.shape[0], device=z.device))
                    getattr(self, f"acc_w_{i}").index_add_(0, indices, flat.detach())

            q_BLC = emb_i(indices).view(B, level, C)
            # Straight-through solo al livello 0 (riduzione e salita apprese), come upstream.
            if i == 0 and self.learned_l1_sampling:
                q_BLC = x_BLC + (q_BLC - x_BLC).detach()
            quantized = self._up_to_full(q_BLC, i, L) if i != n_livelli - 1 else q_BLC
            quantized = self._apply_quant_resi(quantized, i)

            z_hat = z_hat + quantized
            z_rest = z_rest - quantized
            vq_loss = vq_loss + self.commitment_cost * F.mse_loss(z_hat.detach(), z)
            vq_loss = vq_loss + F.mse_loss(z_hat, z_no_grad)
            getattr(self, f"codebook_hit_{i}")[indices] = 1
            if i == taglio:
                z_hat_taglio = z_hat

        vq_loss = vq_loss / n_livelli
        z_out = z + (z_hat_taglio - z).detach()
        # Stessa forma dell'uscita upstream; le statistiche d'uso si calcolano in valutazione (stage1.py),
        # non a ogni forward (richiederebbero una sincronizzazione con la GPU per livello).
        return z_out, vq_loss, [], 0.0, 0.1

    @torch.no_grad()
    def ema_step(self):
        """Aggiorna i codebook con le statistiche accumulate dall'ultimo aggiornamento, poi le azzera."""
        for i, vs in enumerate(self.vqvae_vocab_sizes):
            decay = self._get_ema_decay_level(i)
            cs, w = getattr(self, f"ema_cluster_size_{i}"), getattr(self, f"ema_w_{i}")
            acc_n, acc_w = getattr(self, f"acc_n_{i}"), getattr(self, f"acc_w_{i}")
            cs.mul_(decay).add_(acc_n, alpha=1 - decay)
            n = cs.sum()
            smoothed = (cs + self.epsilon) / (n + vs * self.epsilon) * n
            w.mul_(decay).add_(acc_w, alpha=1 - decay)
            self.embeddings[i].weight.data.copy_(w / smoothed.unsqueeze(1))
            acc_n.zero_()
            acc_w.zero_()
        self._passo_ema += 1
        if 0 < int(self._passo_ema) <= self.riavvio_fino_al_passo:
            self._riavvia()

    @torch.no_grad()
    def _riavvia(self):
        """ R2: codici con statistica EMA sotto la soglia -> rimpiazzati da input veri del passo (+ rumore minimo
        per non creare doppioni identici). Statistiche coerenti: peso 1, somma = vettore. """
        for i in range(len(self.vqvae_vocab_sizes)):
            x = self._campioni[i]
            if x is None:
                continue
            cs, w = getattr(self, f"ema_cluster_size_{i}"), getattr(self, f"ema_w_{i}")
            morti = (cs < self.riavvio_soglia).nonzero(as_tuple=True)[0]
            if morti.numel() == 0:
                continue
            nuovi = x[torch.randint(0, x.shape[0], (morti.numel(),), device=x.device)].float()
            nuovi = nuovi + 1e-3 * x.float().std() * torch.randn_like(nuovi)
            self.embeddings[i].weight.data[morti] = nuovi
            w[morti] = nuovi
            cs[morti] = 1.0
            self._campioni[i] = None

    @torch.no_grad()
    def init_dai_dati(self, z, iterazioni=10, max_campioni=65536):
        """ R1: codebook inizializzati dai dati invece che a caso. z = vettori del codificatore per alcune migliaia di
        storie [B, L, C]. Livello per livello (il residuo di un livello dipende dai precedenti): k-means sugli input
        del livello, codebook = centri, statistiche EMA coerenti (peso 1). """
        z = z.float()
        B, L, C = z.shape
        z_rest = z.clone()
        n = len(self.levels)
        g = torch.Generator(device=z.device).manual_seed(1234)
        S = 512  # storie per blocco: con 8.192 storie i tensori interi occupano ~1,6 GB ciascuno

        def input_livello(i, level, a, b):
            return (self._down_for_quant(z_rest[a:b], i, level) if i != n - 1 else z_rest[a:b]).reshape(-1, C)

        for i, level in enumerate(self.levels):
            flat = torch.cat([input_livello(i, level, a, a + S) for a in range(0, B, S)])
            V = self.vqvae_vocab_sizes[i]
            campioni = flat[torch.randperm(flat.shape[0], device=z.device, generator=g)[:max_campioni]]
            if campioni.shape[0] >= V:
                centri = campioni[torch.randperm(campioni.shape[0], device=z.device, generator=g)[:V]].clone()
            else:
                centri = campioni[torch.randint(0, campioni.shape[0], (V,), device=z.device, generator=g)].clone()
                centri += 1e-3 * campioni.std() * torch.randn(centri.shape, device=z.device, generator=g)
            for _ in range(iterazioni):
                assegna = torch.cat([self._distances(campioni[j:j + 8192], centri).argmin(1)
                                     for j in range(0, campioni.shape[0], 8192)])
                somma = torch.zeros_like(centri).index_add_(0, assegna, campioni)
                conta = torch.zeros(V, device=z.device).index_add_(0, assegna, torch.ones_like(assegna, dtype=torch.float))
                pieni = conta > 0
                centri[pieni] = somma[pieni] / conta[pieni].unsqueeze(1)
            self.embeddings[i].weight.data.copy_(centri)
            getattr(self, f"ema_w_{i}").copy_(centri)
            getattr(self, f"ema_cluster_size_{i}").fill_(1.0)
            del flat, campioni
            for a in range(0, B, S):   # residuo per il livello successivo, a blocchi di storie
                f = input_livello(i, level, a, a + S)
                idx = self._distances(f, centri).argmin(1).view(-1, level)
                q = self._up_to_full(centri[idx], i, L) if i != n - 1 else centri[idx]
                z_rest[a:a + S] -= self._apply_quant_resi(q, i)


# Registrazione nel registro upstream (solo in memoria, nessun file upstream modificato).
vector_quantizers[NOME_NOSTRO] = "tokenizer.quantizer.QuantizzatoreNostro"
