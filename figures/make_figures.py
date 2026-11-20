"""Figures for the README: three charts (SVG, light and dark) and the coarse-to-fine animation (GIF, light and dark).
All numbers are copied from results/ (see the table under each figure in the README); nothing is computed here.

    uv run --with matplotlib --with pillow python figures/make_figures.py
"""
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df",
              "ours": "#2a78d6", "ref": "#eb6834"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "grid": "#33332f",
             "ours": "#3987e5", "ref": "#d95926"},
}


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["text2"], length=0, labelsize=10)
    ax.xaxis.label.set_color(t["text2"])
    ax.yaxis.label.set_color(t["text2"])


def fig_tokenizer(t, mode):
    rows = [  # label, Gen. PPL, who, note
        ("Trained here: random codebook init, 2,400 steps", 222.78, "ours", "2 seeds"),
        ("  codebooks initialised from data", 115.38, "ours", "2 seeds"),
        ("  + early dead-code restart (R12)", 108.17, "ours", "2 seeds"),
        ("  R12, 4,800 steps", 103.61, "ours", "1 seed"),
        ("  R12, 38,400 steps", 92.35, "ours", "2 tokenizer seeds"),
        ("Official tokenizer (authors)", 66.26, "ref", "2 seeds"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=100)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    y = list(range(len(rows)))[::-1]
    for yi, (lab, v, who, note) in zip(y, rows):
        ax.barh(yi, v, height=0.62, color=t[who], edgecolor=t["surface"], linewidth=2)
        ax.text(v + 3, yi, f"{v:.1f}", va="center", ha="left", fontsize=10, color=t["text"])
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=t["text"])
    ax.set_xlim(0, 250)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Generative perplexity (GPT-2 large), lower is better")
    fig.text(0.012, 0.97, "Same generator, different tokenizers (51,200 training stories)", ha="left", va="top", color=t["text"], fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(HERE, f"tokenizer_progress_{mode}.{ext}"), facecolor=t["surface"], dpi=150)
    plt.close(fig)


def fig_budget(t, mode):
    x = [51200, 102400, 409600]
    official = [66.26, 43.84, 25.88]
    ours = [108.17, 65.68, 42.00]
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=100)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    ax.plot(x, ours, color=t["ours"], linewidth=2, marker="o", markersize=8,
            markeredgecolor=t["surface"], markeredgewidth=2, label="Tokenizer trained here (R12, 2,400 steps)")
    ax.plot(x, official, color=t["ref"], linewidth=2, marker="o", markersize=8,
            markeredgecolor=t["surface"], markeredgewidth=2, label="Official tokenizer")
    for xi, a, b in zip(x, ours, official):
        ax.text(xi, a + 4, f"{a:.1f}", ha="center", va="bottom", fontsize=9.5, color=t["text"])
        ax.text(xi, b - 4, f"{b:.1f}", ha="center", va="top", fontsize=9.5, color=t["text"])
        ax.text(xi, (a + b) / 2, f"{a / b:.2f}x", ha="left", va="center", fontsize=9.5, color=t["text2"],
                bbox=dict(boxstyle="round,pad=0.2", fc=t["surface"], ec="none"))
    ax.axhline(16.58, color=t["text2"], linewidth=1, linestyle=(0, (4, 3)))
    ax.text(52000, 17.5, "authors' released model (102M training stories), measured here: 16.58", fontsize=9,
            color=t["text2"], va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["51,200", "102,400", "409,600"], color=t["text2"])
    ax.minorticks_off()
    ax.set_xlim(42000, 520000)
    ax.set_ylim(0, 125)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Stories seen by the generator (log scale)")
    ax.set_ylabel("Generative perplexity")
    fig.text(0.012, 0.97, "An 8x larger generator budget does not close the tokenizer gap", ha="left", va="top", color=t["text"], fontsize=12)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(HERE, f"generator_budget_{mode}.{ext}"), facecolor=t["surface"], dpi=150)
    plt.close(fig)


def fig_recipe(t, mode):
    rows = [  # official tokenizer, 51,200 stories (400 steps x 128), mean of 2 seeds
        ("Paper recipe as reconstructed here\n(linear corruption 0.75 to 0.25, lr 3e-4)", 150.67),
        ("Constant corruption 0.5", 137.82),
        ("+ FlexAttention", 133.11),
        ("+ learning rate 1e-3", 66.26),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.2), dpi=100)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    y = list(range(len(rows)))[::-1]
    for yi, (lab, v) in zip(y, rows):
        ax.barh(yi, v, height=0.62, color=t["ours"], edgecolor=t["surface"], linewidth=2)
        delta = "" if v == rows[0][1] else f"  ({(v - rows[0][1]) / rows[0][1]:+.1%})"
        ax.text(v + 2, yi, f"{v:.1f}{delta}", va="center", ha="left", fontsize=10, color=t["text"])
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=t["text"])
    ax.set_xlim(0, 190)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Generative perplexity (GPT-2 large), lower is better")
    fig.text(0.012, 0.97, "Generator recipe at 51,200 stories: most of the gain is the learning rate", ha="left", va="top", color=t["text"], fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(HERE, f"generator_recipe_{mode}.{ext}"), facecolor=t["surface"], dpi=150)
    plt.close(fig)


def font(size, bold=False):
    names = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"])
    for n in names:
        for d in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", ""):
            try:
                return ImageFont.truetype(os.path.join(d, n) if d else n, size)
            except OSError:
                continue
    return ImageFont.load_default()


def gif(t, mode):
    data = json.load(open(os.path.join(HERE, "coarse_to_fine.json"), encoding="utf-8"))
    panels = [("Official tokenizer", data["frames"]["official"], t["ref"]),
              ("Tokenizer trained here (38,400 steps)", data["frames"]["ours (38,400 steps)"], t["ours"])]
    W, H, pad = 900, 560, 28
    f_title, f_head, f_body, f_small = font(22, True), font(17, True), font(16), font(14)
    frames = []
    for k in range(12):
        im = Image.new("RGB", (W, H), t["surface"])
        d = ImageDraw.Draw(im)
        head = "Decoding one story from level 0 only" if k == 0 else f"Decoding one story from levels 0-{k} (of 0-11)"
        d.text((pad, pad - 4), head, font=f_title, fill=t["text"])
        bar_w = (W - 2 * pad) * (k + 1) / 12
        d.rounded_rectangle([pad, pad + 34, W - pad, pad + 40], radius=3, fill=t["grid"])
        d.rounded_rectangle([pad, pad + 34, pad + bar_w, pad + 40], radius=3, fill=t["text2"])
        y = pad + 58
        for name, fr, col in panels:
            f = fr[k]
            d.rounded_rectangle([pad, y + 3, pad + 10, y + 19], radius=2, fill=col)
            d.text((pad + 18, y), f"{name}  -  {f['token_accuracy']:.0%} of tokens correct", font=f_head,
                   fill=t["text"])
            body = " ".join(f["text"].split())
            body = body[:330] + (" ..." if len(body) > 330 else "")
            lines = textwrap.wrap(body, 92)[:5]
            yy = y + 28
            for ln in lines:
                d.text((pad + 18, yy), ln, font=f_body, fill=t["text"])
                yy += 23
            y += 28 + 5 * 23 + 22
        d.text((pad, H - pad - 14), "Each level adds detail. Level 0 is a single code per story; level 11 has 128.",
               font=f_small, fill=t["text2"])
        frames.append(im)
    durations = [900] * 11 + [3500]
    frames[0].save(os.path.join(HERE, f"coarse_to_fine_{mode}.gif"), save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)


for mode, t in THEMES.items():
    fig_tokenizer(t, mode)
    fig_budget(t, mode)
    fig_recipe(t, mode)
    gif(t, mode)
for f in os.listdir(HERE):
    if f.endswith(".svg"):
        s = open(os.path.join(HERE, f), encoding="utf-8").read()
        s2 = s[:s.index("<svg")] .replace(s[s.index("<!DOCTYPE"):s.index("<svg")], "") + s[s.index("<svg"):] if "<!DOCTYPE" in s else s
        open(os.path.join(HERE, f), "w", encoding="utf-8").write(s2)
print("done:", sorted(f for f in os.listdir(HERE) if f.endswith((".svg", ".png", ".gif"))))
