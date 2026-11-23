# Building the research note

The PDF is built from `build_paper.py` with matplotlib, NumPy and Pillow. It contains vector charts, embedded TrueType fonts, Unicode text maps and clickable reference/contact links. No model is loaded and no training is started.

```sh
python paper/build_paper.py
```

The principal charts read the existing `results/runs_index.csv` and record their exact input run IDs in `figure-data.json`. Other reported diagnostics are transcribed from `EXPERIMENTS.md`; the illustrative reconstruction reads `figures/coarse_to_fine.json`.

Windows Georgia and Segoe UI fonts are used when installed; DejaVu fonts bundled with matplotlib are the portable fallback. On another system, set `PAPER_FONTDIR` to a folder containing those font files (for example the `Windows/Fonts` folder of a mounted Windows partition). Font changes may slightly affect wrapping, which the build checks against the page bottom.

The v0.1 PDF was built on Windows with Python 3.12 and matplotlib 3.11.2. The v0.2 PDF was built on Linux with Python 3.12, matplotlib 3.11.2 and the same Windows fonts through `PAPER_FONTDIR`; the four v0.1 charts it regenerates are pixel-identical to the v0.1 files. The script also writes PNG previews of each page to `publication-preview/`, a folder that is not part of the repository.
