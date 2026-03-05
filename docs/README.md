# Research Paper Formatting Details (IEEE)

When submitting or preparing a research paper for a computer science or engineering domain based on this OCR extraction project, the standard is the **IEEE Conference or Journal format**.

In this `docs/` folder, I've created the following files to help you get started:

1. `IEEE_Paper_Draft.tex` - This is a raw LaTeX file, fully coded into the IEEE template format. 
2. `IEEE_Paper_Draft.md` - This is a simple Markdown readable version of the paper's contents.

## How to use the IEEE LaTeX template (`.tex`):

**Option 1: Overleaf (Easiest)**
1. Go to [Overleaf](https://www.overleaf.com/).
2. Create a new "Blank Project".
3. Copy the contents of `IEEE_Paper_Draft.tex` and paste it into Overleaf's `main.tex`.
4. The paper will automatically compile and produce an IEEE-formatted PDF!

**Option 2: Local Compilation**
1. You need a LaTeX distribution on your Linux machine (e.g., `texlive`).
2. Run the command: `pdflatex IEEE_Paper_Draft.tex` in the `docs` folder.
3. This will generate `IEEE_Paper_Draft.pdf`.

## What to add to the paper to make it complete:
To make this a strong research paper, you should consider adding the following data to the draft:
- **Experimental Results**: Run `ocr_preprocessor` on several test images (you have some in the `testimages/` folder). Note the processing speed in milliseconds and insert those metrics in the "Evaluation and Results" section. 
- **Comparison figures**: Add screenshots comparing the original distorted text versus the unwarped and binarized output your tool yields. (Note: standard `\includegraphics{}` works in LaTeX for this).
- **Authors**: Replace the placeholder Author Names and Organizations in the `.tex` and `.md` files.
