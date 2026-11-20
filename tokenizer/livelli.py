"""Liste dei livelli del tokenizer: quante posizioni ha ogni livello, dal più grossolano al più fine.

    livelli(12, "ufficiale")   -> [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]
    livelli(32, "lineare")     -> 32 lunghezze da 1 a 128, a passo costante
    livelli(32, "quadratica")  -> più livelli corti, pochi lunghi
    livelli(32, "esponenziale")

Ogni lista parte da 1, finisce alla lunghezza del testo ed è strettamente crescente.

Uso da riga di comando (stampa le liste e il numero di posizioni):
    .venv\\Scripts\\python.exe -I tokenizer\\livelli.py
"""
UFFICIALE_12 = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]
SCHEMI = ("ufficiale", "lineare", "quadratica", "esponenziale")


def livelli(k, schema="lineare", lunghezza=128):
    k, lunghezza = int(k), int(lunghezza)
    if schema == "ufficiale":
        if k != 12 or lunghezza != 128:
            raise ValueError("lo schema 'ufficiale' esiste solo per 12 livelli e 128 token")
        return list(UFFICIALE_12)
    if not 2 <= k <= lunghezza:
        raise ValueError(f"servono tra 2 e {lunghezza} livelli, non {k}")
    if schema == "lineare":
        f = lambda t: 1 + (lunghezza - 1) * t  # noqa: E731
    elif schema == "quadratica":
        f = lambda t: 1 + (lunghezza - 1) * t * t  # noqa: E731
    elif schema == "esponenziale":
        f = lambda t: float(lunghezza) ** t  # noqa: E731
    else:
        raise ValueError(f"schema sconosciuto: {schema} (validi: {SCHEMI})")
    out = [1]
    for i in range(1, k - 1):
        v = max(int(round(f(i / (k - 1)))), out[-1] + 1)       # strettamente crescente
        out.append(min(v, lunghezza - (k - 1 - i)))            # lascia posto ai livelli che restano
    out.append(lunghezza)
    assert len(out) == k and all(b > a for a, b in zip(out, out[1:])), out
    return out


def posizioni_generatore(lv, prefisso=1):
    """Posizioni su cui lavora il generatore: prefisso + tutti i livelli tranne il primo."""
    return prefisso + sum(lv[1:])


if __name__ == "__main__":
    base = posizioni_generatore(UFFICIALE_12)
    print(f"ufficiale    K=12  posizioni {base:5d}  x1.00  {UFFICIALE_12}")
    for schema in ("lineare", "quadratica", "esponenziale"):
        for k in (12, 16, 24, 32, 40):
            lv = livelli(k, schema)
            p = posizioni_generatore(lv)
            print(f"{schema:12s} K={k:2d}  posizioni {p:5d}  x{p / base:.2f}  {lv}")
