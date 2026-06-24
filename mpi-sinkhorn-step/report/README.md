# Report Build Notes

This folder is self-contained for editing and rebuilding the final report and
slides on another machine.

## Files

- `main.tex`: final report source.
- `main.pdf`: compiled final report.
- `slides.tex`: Beamer slide source.
- `slides.pdf`: compiled slides.
- `slides_script.md`: speaker script for the slide deck.
- `assets/`: the single shared folder containing every logo, photo, and chart
  referenced by both LaTeX files.

Both `main.tex` and `slides.tex` use:

```tex
\graphicspath{{assets/}}
```

Therefore, keep the `assets/` folder next to the `.tex` files when moving this
report to another machine.

## Build

From this folder:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error slides.tex
```

If `latexmk` is unavailable, run `pdflatex` repeatedly until references settle.
