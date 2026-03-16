"""
OCR Evaluation Script
=====================
Compares three OCR pipelines:
  (A) Baseline Tesseract
  (B) My Preprocessor + Tesseract
  (C) PaddleOCR  (English only — Malayalam not available in pip version)

Dataset layout expected:
  evaluation/dataset/[Language]/[Category]/[image_name].jpg
  evaluation/dataset/[Language]/[Category]/[image_name].txt  (ground truth)

Outputs (all inside evaluation/):
  - results/detailed_results.csv       — per-file CER & WER for every method
  - results/summary_report.csv         — averages grouped by Language & Category
  - results/logs/[Lang]/[Cat]/[file]_[method].txt — raw OCR text for inspection

Usage:
  python evaluate_ocr.py
"""

import os
import sys
import logging
import traceback
import subprocess
from pathlib import Path

import cv2
import pytesseract
import pandas as pd
from paddleocr import PaddleOCR
from jiwer import cer, wer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "results" / "evaluation.log",
                            mode="w", encoding="utf-8"),
    ],
)
# Suppress verbose PaddleOCR / PaddlePaddle internal logs
logging.getLogger("ppocr").setLevel(logging.WARNING)
logging.getLogger("paddle").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVAL_DIR   = Path(__file__).parent           # .../evaluation/
DATASET    = EVAL_DIR / "dataset"
RESULTS    = EVAL_DIR / "results"
LOGS_ROOT  = RESULTS / "logs"

RESULTS.mkdir(parents=True, exist_ok=True)
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global PaddleOCR cache
# ---------------------------------------------------------------------------
paddle_ocr_cache = {}

def get_paddle_ocr(lang: str) -> PaddleOCR:
    if lang not in paddle_ocr_cache:
        paddle_ocr_cache[lang] = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    return paddle_ocr_cache[lang]


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------
def run_tesseract_baseline(image_path: str, lang: str) -> str:
    """Run vanilla Tesseract on the raw image file."""
    try:
        text = pytesseract.image_to_string(image_path, lang=lang).strip()
        return text
    except Exception as exc:
        logger.warning("  [Tesseract-Baseline] Error on %s: %s", image_path, exc)
        return ""


def run_tesseract_preprocessed(image_path: str, lang: str) -> str:
    """Run Tesseract on the image preprocessed by C++ ocr_preprocessor."""
    try:
        # Run C++ preprocessor
        # Output is saved to output/masked.png by default
        preprocessor_cmd = [
            "./build/ocr_preprocessor",
            image_path,
            "det_v5_ir9.onnx"
        ]
        
        result = subprocess.run(
            preprocessor_cmd, 
            capture_output=True, 
            text=True, 
            cwd=str(EVAL_DIR.parent) # Run from ocp root
        )
        
        if result.returncode != 0:
            logger.warning("  [Preprocessor] Error on %s: %s", image_path, result.stderr)
            return ""

        output_img_path = str(EVAL_DIR.parent / "output" / "masked.png")
        if not os.path.exists(output_img_path):
             logger.warning("  [Preprocessor] Output file not found for %s", image_path)
             return ""

        # Run Tesseract on the preprocessed image
        text = pytesseract.image_to_string(output_img_path, lang=lang).strip()
        return text
    except Exception as exc:
        logger.warning("  [Tesseract-Preprocessed] Error on %s: %s", image_path, exc)
        return ""


def run_paddleocr(image_path: str, lang: str) -> str:
    """Run PaddleOCR and concatenate all detected text lines.
    Note: Only English is supported via pip; Malayalam is not available."""
    try:
        ocr = get_paddle_ocr(lang)
        result = ocr.ocr(image_path, cls=True)
        if not result or not result[0]:
            return ""
        lines = [line[1][0] for block in result for line in block if line]
        return " ".join(lines).strip()
    except Exception as exc:
        logger.warning("  [PaddleOCR] Error on %s: %s", image_path, exc)
        return ""



# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def safe_cer(reference: str, hypothesis: str) -> float:
    """Return CER, clamped to [0, 1]. Returns 1.0 if reference is empty."""
    if not reference:
        return 1.0
    if not hypothesis:
        return 1.0
    try:
        return min(cer(reference, hypothesis), 1.0)
    except Exception:
        return 1.0


def safe_wer(reference: str, hypothesis: str) -> float:
    """Return WER, clamped to [0, 1]. Returns 1.0 if reference is empty."""
    if not reference:
        return 1.0
    if not hypothesis:
        return 1.0
    try:
        return min(wer(reference, hypothesis), 1.0)
    except Exception:
        return 1.0

# ---------------------------------------------------------------------------
# Log writer
# ---------------------------------------------------------------------------
def write_ocr_log(language: str, category: str, stem: str,
                  method: str, text: str) -> None:
    """Save raw OCR output to results/logs/[Lang]/[Cat]/[stem]_[method].txt"""
    log_dir = LOGS_ROOT / language / category
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{stem}_{method}.txt"
    log_file.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------
def evaluate() -> None:
    records = []  # list[dict] – one entry per image × method

    if not DATASET.exists():
        logger.error("Dataset directory not found: %s", DATASET)
        logger.error("Expected structure: evaluation/dataset/[Language]/[Category]/*.jpg")
        sys.exit(1)

    languages = sorted([p for p in DATASET.iterdir() if p.is_dir()])
    if not languages:
        logger.warning("No language sub-directories found inside %s", DATASET)
        return

    for lang_dir in languages:
        language = lang_dir.name
        categories = sorted([p for p in lang_dir.iterdir() if p.is_dir()])

        if not categories:
            logger.info("[%s] No category sub-directories found — skipping.", language)
            continue

        for cat_dir in categories:
            category = cat_dir.name
            jpg_files = sorted(cat_dir.glob("*.jpg"))

            # ── Handle empty category folders ──────────────────────────────
            if not jpg_files:
                logger.info("[%s/%s] No .jpg files found — skipping (empty folder).",
                            language, category)
                continue

            logger.info("[%s/%s] Processing %d image(s)...",
                        language, category, len(jpg_files))

            for img_path in jpg_files:
                stem      = img_path.stem
                gt_path   = img_path.with_suffix(".txt")
                image_str = str(img_path)

                # ── Ground truth ───────────────────────────────────────────
                if not gt_path.exists():
                    logger.warning("  [SKIP] Missing ground truth: %s", gt_path)
                    continue

                ground_truth = gt_path.read_text(encoding="utf-8").strip()
                if not ground_truth:
                    logger.warning("  [SKIP] Empty ground truth file: %s", gt_path)
                    continue

                logger.info("  Processing: %s", img_path.name)

                # ── Map languages ──────────────────────────────────────────
                tess_lang = "eng"
                is_malayalam = "malayalam" in language.lower() or "mal" in language.lower()
                if is_malayalam:
                    tess_lang = "mal"

                # ── Run pipelines ──────────────────────────────────────────
                pipelines = {
                    "tesseract_baseline":     lambda img: run_tesseract_baseline(img, tess_lang),
                    "tesseract_preprocessed": lambda img: run_tesseract_preprocessed(img, tess_lang),
                }

                # PaddleOCR — English only (Malayalam not available in pip)
                if not is_malayalam:
                    pipelines["paddleocr"] = lambda img: run_paddleocr(img, "en")

                for method_name, ocr_fn in pipelines.items():
                    try:
                        ocr_text = ocr_fn(image_str)
                    except Exception as exc:
                        logger.error("    [%s] Unexpected error: %s", method_name, exc)
                        logger.debug(traceback.format_exc())
                        ocr_text = ""

                    # Metrics
                    char_err = safe_cer(ground_truth, ocr_text)
                    word_err = safe_wer(ground_truth, ocr_text)

                    logger.info("    %-25s  CER=%.4f  WER=%.4f",
                                method_name, char_err, word_err)

                    # Save OCR log
                    write_ocr_log(language, category, stem, method_name, ocr_text)

                    records.append({
                        "language":   language,
                        "category":   category,
                        "filename":   img_path.name,
                        "method":     method_name,
                        "cer":        round(char_err, 6),
                        "wer":        round(word_err, 6),
                    })

    if not records:
        logger.warning("No records were produced — check that ground truth files exist.")
        return

    # ── Save detailed_results.csv ─────────────────────────────────────────
    df = pd.DataFrame(records)
    detailed_csv = RESULTS / "detailed_results.csv"
    df.to_csv(detailed_csv, index=False)
    logger.info("Saved detailed results → %s", detailed_csv)

    # ── Save summary_report.csv ───────────────────────────────────────────
    summary = (
        df.groupby(["language", "category", "method"])
          .agg(avg_cer=("cer", "mean"), avg_wer=("wer", "mean"), num_files=("filename", "count"))
          .round(6)
          .reset_index()
    )
    summary_csv = RESULTS / "summary_report.csv"
    summary.to_csv(summary_csv, index=False)
    logger.info("Saved summary report → %s", summary_csv)

    # ── Print comparison table to stdout ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE — METHOD COMPARISON")
    print("=" * 70)
    comparison = (
        df.groupby("method")
          .agg(avg_cer=("cer", "mean"), avg_wer=("wer", "mean"), total_files=("filename", "count"))
          .round(6)
          .sort_values("avg_cer")
          .reset_index()
    )
    print(comparison.to_string(index=False))
    print("=" * 70 + "\n")

    # ── Identify best method ──────────────────────────────────────────────
    best_row = comparison.iloc[0]
    logger.info("Best method by avg CER: '%s'  (CER=%.4f, WER=%.4f)",
                best_row["method"], best_row["avg_cer"], best_row["avg_wer"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting OCR evaluation...")
    logger.info("Dataset  : %s", DATASET)
    logger.info("Results  : %s", RESULTS)
    logger.info("Logs     : %s", LOGS_ROOT)
    logger.info("-" * 60)

    try:
        evaluate()
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user.")
    except Exception as exc:
        logger.critical("Fatal error: %s", exc)
        logger.debug(traceback.format_exc())
        sys.exit(1)

    logger.info("Done.")
