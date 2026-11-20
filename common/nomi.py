"""Nomi standard delle run, costruiti dalla configurazione: chi legge il nome sa con che impostazioni è stata fatta.

Generatore:  gen-<tokenizer>_<storie viste>_<passi>x<storie per passo>_<corruzione>_lr<lr>[_flex][_extra]_s<seed>
    gen-tokuff_51200_400x128_lin75-25_lr3e-4_s42        ricetta del paper, tokenizer ufficiale
    gen-tokuff_51200_800x64_c50_lr1e-3_flex_s42         corruzione costante 0,5, lr 1e-3, FlexAttention
    gen-mytok12std-2400x256_51200_400x128_c50_lr3e-4_s42   stesso generatore sopra un tokenizer NOSTRO
Tokenizer nostro:   mytok-<livelli><schema>_<storie viste>_<passi>x<storie per passo>_s<seed>
    mytok-12std_614400_2400x256_s42

"tokuff" = tokenizer ufficiale; "mytok" = tokenizer nostro. Due generatori con le stesse impostazioni su tokenizer
diversi hanno lo stesso nome dopo la prima parte: gen-tokuff_<resto> e gen-mytok..._<resto> sono controparti.

Le storie viste vengono prima dei passi: in ordine alfabetico le run con lo stesso budget stanno vicine.
Corruzione: c50 = costante 0,5; lin75-25 = lineare da 0,75 a 0,25; unif0-100 = un tasso per storia tra 0 e 1;
"l0" in fondo = anche il livello 0 corrotto. Schemi dei livelli: std (come l'ufficiale), lin, quad, exp.
"""
import os

# "std" = stessa spaziatura dei livelli del tokenizer ufficiale ("uff" resta solo per il tokenizer ufficiale).
SCHEMI = {"ufficiale": "std", "lineare": "lin", "quadratica": "quad", "esponenziale": "exp"}
MIO = "mytok"   # tutto ciò che è fatto con il nostro tokenizer porta questo prefisso


def _g(cfg, path, default=None):
    cur = cfg
    for k in path.split("."):
        if cur is None:
            return default
        try:
            cur = cur[k]
        except (KeyError, TypeError, AttributeError):
            return default
    return default if cur is None else cur


def _num(x):
    """3e-4 -> '3e-4', 0.0015 -> '1.5e-3'."""
    m, e = f"{float(x):.6e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"{m}e{int(e)}"


def _pct(x):
    return f"{int(round(float(x) * 100)):02d}"


def nome_tokenizer_da_base(base_config):
    """Etichetta del tokenizer usato da un generatore, dal suo base_config."""
    bc = str(base_config or "").replace("\\", "/")
    if bc in ("", "artifacts/tinystories/config.yaml") or bc.endswith("/artifacts/tinystories/config.yaml"):
        return "tokuff"
    parti = bc.split("/")
    if "runs" in parti:   # runs/<run del tokenizer>/export[/passo_XXXXXXXX]/config.yaml
        run = parti[parti.index("runs") + 1]
        pezzi = run.split("_")
        if run.startswith(MIO + "-") and len(pezzi) >= 3:   # mytok-32lin_614400_2400x256_s42 -> mytok32lin-2400x256
            etichetta = f"{MIO}{pezzi[0][len(MIO) + 1:]}-{pezzi[2]}"
            if len(pezzi) >= 4 and pezzi[3] != "s42":   # tokenizer con un seed diverso da 42: "-ts43"
                etichetta += f"-t{pezzi[3]}"
        else:
            etichetta = run
        copia = [x for x in parti if x.startswith("passo_")]
        if copia:                                         # copia intermedia del tokenizer
            etichetta += f"-p{int(copia[0][6:])}"
        return etichetta
    return os.path.basename(os.path.dirname(bc)) or "tok"


def nome_generatore(cfg):
    t = _g(cfg, "train")
    batch = int(t["micro_batch"]) * int(t["accum"])
    passi = int(t["steps"])
    tok = "tokuff" if str(_g(cfg, "tokenizer_checkpoint", "official")) == "official" and \
        nome_tokenizer_da_base(_g(cfg, "base_config")) == "tokuff" else nome_tokenizer_da_base(_g(cfg, "base_config"))
    if _g(cfg, "encode_fp64", False):   # codici calcolati a 64 bit: gen-tokuff64_..., controparte di gen-tokuff_...
        tok += "64"
    if not _g(cfg, "corruption.enabled", True):
        corr = "c00"
    elif str(_g(cfg, "corruption.mode", "linear")) == "per_example_uniform":
        corr = f"unif0-{_pct(_g(cfg, 'corruption.rate_max', 1.0))}"
    else:
        a, b = float(_g(cfg, "corruption.rate_first")), float(_g(cfg, "corruption.rate_last"))
        corr = f"c{_pct(a)}" if abs(a - b) < 1e-9 else f"lin{_pct(a)}-{_pct(b)}"
    if _g(cfg, "corruption.enabled", True) and not _g(cfg, "corruption.skip_level0", True):
        corr += "l0"
    parti = [f"gen-{tok}", str(passi * batch), f"{passi}x{batch}", corr, f"lr{_num(t['lr'])}"]
    if abs(float(t["min_lr"]) - float(t["lr"]) / 10) > 1e-12:
        parti.append(f"min{_num(t['min_lr'])}")
    if _g(cfg, "model_overrides.use_flex_attention", False):
        parti.append("flex")
    if _g(cfg, "nome_extra"):          # etichetta libera per run di controllo (es. micro-batch diverso)
        parti.append(str(_g(cfg, "nome_extra")))
    parti.append(f"s{int(t['seed'])}")
    return "_".join(parti)


def nome_tokenizer(cfg):
    t = _g(cfg, "train")
    batch = int(t["micro_batch"]) * int(t["accum"])
    passi = int(t["steps"])
    k = int(_g(cfg, "tokenizer.livelli"))
    schema = SCHEMI.get(str(_g(cfg, "tokenizer.schema")), str(_g(cfg, "tokenizer.schema")))
    rim = _g(cfg, "tokenizer.rimedi") or {}
    extra = ("-dati" if _g(rim, "init_dati", False) else "") + ("-rv" if int(_g(rim, "riavvio_fino_al_passo", 0) or 0) > 0 else "")
    if _g(cfg, "tokenizer.norma_livello0", True) is False:   # prova B: senza la normalizzazione del livello 0
        extra += "-nonorm"
    return f"{MIO}-{k}{schema}{extra}_{passi * batch}_{passi}x{batch}_s{int(t['seed'])}"  # -dati / -rv = rimedi R1 / R2


def nome_standard(cfg):
    return nome_tokenizer(cfg) if _g(cfg, "tokenizer.livelli") is not None else nome_generatore(cfg)
