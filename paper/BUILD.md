# Building the research note

The PDF is built from `build_paper.py` with matplotlib, NumPy and Pillow. It contains vector charts, embedded TrueType fonts, Unicode text maps and clickable reference/contact links. No model is loaded and no training is started.

```sh
python paper/build_paper.py
```

The principal charts read the existing `results/runs_index.csv` and record their exact input run IDs in `figure-data.json`. Other reported diagnostics are transcribed from `EXPERIMENTS.md`; the illustrative reconstruction reads `figures/coarse_to_fine.json`.

Windows Georgia and Segoe UI fonts are used when installed; DejaVu fonts bundled with matplotlib are the portable fallback. Font changes may slightly affect wrapping, which the build checks against the page bottom.

The supplied PDF was built with Python 3.12 and matplotlib 3.11.2. The script also writes PNG previews of each page to `publication-preview/`, a folder that is not part of the repository.
