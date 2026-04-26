"""
ocr_engine.py  —  MediSimplify
================================
OCR pipeline for extracting structured lab results from uploaded images.

Pipeline:
  1. Preprocess  – Grayscale + contrast enhancement (PIL) + denoising (OpenCV if available)
  2. OCR         – Pytesseract extracts raw text
  3. Keyword     – Regex locates known lab-test names
  4. Value match – Finds the numeric value (and optional unit) that follows each keyword
  5. Output      – Returns a list of dicts: {test_name, test_value, unit}
                   Falls back gracefully when no values can be parsed.

Public API (matches imports in main.py exactly):
  process_lab_image(image_bytes) -> tuple[str, list[dict]]
  extract_numbers(raw_text)      -> dict   (legacy quick-lookup dict)
"""

from __future__ import annotations

import re
import io
import logging
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

# ── Optional heavy deps (graceful degradation) ────────────────────────────────
try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("opencv-python not installed — skipping advanced denoising step.")

try:
    import pytesseract
    _TESS_AVAILABLE = True
except ImportError:
    _TESS_AVAILABLE = False
    logger.warning("pytesseract not installed — OCR will return empty text.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. KNOWN LAB TESTS  (keyword → canonical name + default unit)
# ─────────────────────────────────────────────────────────────────────────────

# Each entry:  regex_pattern  ->  (canonical_name, default_unit)
# Patterns are matched case-insensitively against the OCR text.
LAB_PATTERNS: list[tuple[str, str, str]] = [
    # ── Diabetes markers ──────────────────────────────────────────────────────
    (r"hb\s*a\s*1\s*c|glycated\s+hemo(?:globin)?|glyco(?:sylated)?\s+hb",
     "HbA1c", "%"),
    (r"fasting\s+(?:blood\s+)?(?:glucose|sugar)|fbs|fbg|fasting\s+plasma\s+glucose",
     "Fasting Blood Sugar", "mg/dL"),
    (r"pp\s*(?:bs|bg)|post(?:\s*prandial)?\s+(?:blood\s+)?(?:glucose|sugar)|2\s*hr\s+(?:pp|glucose)",
     "Post-Prandial Blood Sugar", "mg/dL"),
    (r"(?:random\s+)?(?:blood\s+)?(?:glucose|sugar)(?!\s+fasting|\s+pp|\s+post)",
     "Blood Glucose", "mg/dL"),

    # ── Lipid panel ───────────────────────────────────────────────────────────
    (r"total\s+cholesterol|serum\s+cholesterol",
     "Total Cholesterol", "mg/dL"),
    (r"ldl[\s\-]*(?:cholesterol|chol|c)?|low[\s\-]density\s+lipoprotein",
     "LDL Cholesterol", "mg/dL"),
    (r"hdl[\s\-]*(?:cholesterol|chol|c)?|high[\s\-]density\s+lipoprotein",
     "HDL Cholesterol", "mg/dL"),
    (r"triglycerides?|tg\b|trigs?\b",
     "Triglycerides", "mg/dL"),
    (r"vldl[\s\-]*(?:cholesterol|chol|c)?",
     "VLDL Cholesterol", "mg/dL"),

    # ── Complete Blood Count ──────────────────────────────────────────────────
    (r"hemoglobin|haemoglobin|hgb|hb\b(?!\s*a)",
     "Hemoglobin", "g/dL"),
    (r"hematocrit|haematocrit|hct|packed\s+cell\s+volume|pcv",
     "Hematocrit", "%"),
    (r"(?:total\s+)?(?:rbc|red\s+blood\s+(?:cell|corpuscle)\s+count)",
     "RBC Count", "mill/µL"),
    (r"(?:total\s+)?(?:wbc|white\s+blood\s+(?:cell|corpuscle)\s+count|leucocyte\s+count|leukocyte\s+count)",
     "WBC Count", "cells/µL"),
    (r"platelet\s+count|plt\b|thrombocyte\s+count",
     "Platelet Count", "lakh/µL"),
    (r"mcv\b|mean\s+corpuscular\s+volume",
     "MCV", "fL"),
    (r"mch\b|mean\s+corpuscular\s+hemoglobin(?!\s+conc)",
     "MCH", "pg"),
    (r"mchc\b|mean\s+corp(?:uscular)?\s+hemo(?:globin)?\s+conc",
     "MCHC", "g/dL"),

    # ── Kidney function ───────────────────────────────────────────────────────
    (r"serum\s+creatinine|s\.?\s*creatinine|creatinine\b",
     "Serum Creatinine", "mg/dL"),
    (r"blood\s+urea\s+nitrogen|bun\b",
     "Blood Urea Nitrogen", "mg/dL"),
    (r"serum\s+urea|blood\s+urea\b|urea\b",
     "Blood Urea", "mg/dL"),
    (r"uric\s+acid|serum\s+uric\s+acid",
     "Uric Acid", "mg/dL"),
    (r"egfr\b|estimated\s+(?:gfr|glomerular\s+filtration)",
     "eGFR", "mL/min/1.73m²"),

    # ── Liver function ────────────────────────────────────────────────────────
    (r"sgot|ast\b|aspartate\s+(?:amino)?transferase",
     "SGOT (AST)", "U/L"),
    (r"sgpt|alt\b|alanine\s+(?:amino)?transferase",
     "SGPT (ALT)", "U/L"),
    (r"alkaline\s+phosphatase|alp\b",
     "Alkaline Phosphatase", "U/L"),
    (r"(?:total\s+)?bilirubin\b(?!\s+direct|\s+indirect)",
     "Total Bilirubin", "mg/dL"),
    (r"direct\s+bilirubin|conjugated\s+bilirubin",
     "Direct Bilirubin", "mg/dL"),
    (r"indirect\s+bilirubin|unconjugated\s+bilirubin",
     "Indirect Bilirubin", "mg/dL"),
    (r"(?:serum\s+)?albumin\b",
     "Serum Albumin", "g/dL"),
    (r"(?:total\s+)?protein\b|serum\s+protein",
     "Total Protein", "g/dL"),

    # ── Thyroid ───────────────────────────────────────────────────────────────
    (r"tsh\b|thyroid\s+stimulating\s+hormone",
     "TSH", "µIU/mL"),
    (r"free\s+t4|ft4\b|thyroxine\s+free",
     "Free T4", "ng/dL"),
    (r"free\s+t3|ft3\b|triiodothyronine\s+free",
     "Free T3", "pg/mL"),
    (r"t4\b|total\s+t4|serum\s+thyroxine",
     "T4 (Thyroxine)", "ng/dL"),
    (r"t3\b|total\s+t3|serum\s+triiodothyronine",
     "T3 (Triiodothyronine)", "pg/mL"),

    # ── Electrolytes ──────────────────────────────────────────────────────────
    (r"serum\s+sodium|sodium\b|na\+?\b",
     "Sodium", "mEq/L"),
    (r"serum\s+potassium|potassium\b|k\+?\b(?!\s*cal)",
     "Potassium", "mEq/L"),
    (r"serum\s+chloride|chloride\b|cl\-?\b",
     "Chloride", "mEq/L"),
    (r"serum\s+calcium|calcium\b|ca\+?\b",
     "Calcium", "mg/dL"),
    (r"serum\s+magnesium|magnesium\b|mg\b(?!\/)",
     "Magnesium", "mg/dL"),

    # ── Vitamins / minerals ───────────────────────────────────────────────────
    (r"vitamin\s+d\s*(?:total|25\s*oh)?|25[\s\-]hydroxy\s*vitamin\s*d|25\s*ohd",
     "Vitamin D", "ng/mL"),
    (r"vitamin\s+b\s*12|cobalamin|cyanocobalamin",
     "Vitamin B12", "pg/mL"),
    (r"serum\s+iron|iron\b(?!\s+binding)",
     "Serum Iron", "µg/dL"),
    (r"ferritin\b",
     "Ferritin", "ng/mL"),
    (r"tibc\b|total\s+iron[\s\-]binding\s+capacity",
     "TIBC", "µg/dL"),

    # ── Inflammation ──────────────────────────────────────────────────────────
    (r"crp\b|c[\s\-]reactive\s+protein",
     "CRP", "mg/L"),
    (r"esr\b|erythrocyte\s+sedimentation\s+rate",
     "ESR", "mm/hr"),

    # ── Cardiac ───────────────────────────────────────────────────────────────
    (r"troponin[\s\-]?[it]?\b",
     "Troponin", "ng/mL"),
    (r"ck[\s\-]mb\b|creatine\s+kinase[\s\-]mb",
     "CK-MB", "U/L"),
]

# Unit synonyms for post-normalisation
_UNIT_ALIASES: dict[str, str] = {
    "mg/dl":  "mg/dL",
    "mg/l":   "mg/L",
    "g/dl":   "g/dL",
    "g/l":    "g/L",
    "iu/ml":  "IU/mL",
    "ug/dl":  "µg/dL",
    "ug/ml":  "µg/mL",
    "pg/ml":  "pg/mL",
    "ng/dl":  "ng/dL",
    "ng/ml":  "ng/mL",
    "meq/l":  "mEq/L",
    "u/l":    "U/L",
    "fl":     "fL",
    "pg":     "pg",
    "mm/hr":  "mm/hr",
    "cells/ul": "cells/µL",
    "mill/ul":  "mill/µL",
    "lakh/ul":  "lakh/µL",
    "uiu/ml": "µIU/mL",
}

# Regex that matches a number optionally followed by a unit
_VALUE_UNIT_RE = re.compile(
    r"""
    (?::|=|\s)+                          # separator (colon, equals, or whitespace)
    (?P<value>
        \d{1,6}                          # integer part
        (?:[.,]\d{1,3})?                 # optional decimal (period OR comma locale)
    )
    \s*
    (?P<unit>
        (?:mg|g|µg|ug|ng|pg|mEq|meq|
           mIU|uIU|IU|iu|mL|ml|L|l|
           µL|ul|dL|dl|fL|fl|
           mm|hr|%|\+|-)*
        (?:\/
           (?:dL|dl|L|l|mL|ml|µL|ul|hr|min|
              1\.73m²|1\.73m2|mm³|mm3|µL|ul)
        )?
        (?:\d+(?:\.\d+)?m²)?            # optional trailing exponent like /1.73m²
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_image(image_bytes: bytes) -> Image.Image:
    """
    Convert to grayscale, boost contrast, and optionally denoise with OpenCV.
    Returns a PIL Image ready for Tesseract.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ── Step 1: Resize if small (Tesseract accuracy drops below ~300 DPI) ──
    min_dim = 1200
    w, h = img.size
    if max(w, h) < min_dim:
        scale = min_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # ── Step 2: Grayscale ────────────────────────────────────────────────────
    gray = img.convert("L")

    # ── Step 3: Contrast enhancement (PIL) ──────────────────────────────────
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    gray = ImageEnhance.Sharpness(gray).enhance(1.5)

    # ── Step 4: OpenCV denoising + Otsu threshold (if available) ─────────────
    if _CV2_AVAILABLE:
        arr = np.array(gray)
        arr = cv2.fastNlMeansDenoising(arr, h=10, templateWindowSize=7, searchWindowSize=21)
        # Otsu's binarisation gives Tesseract clean black-on-white text
        _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray = Image.fromarray(arr)
    else:
        # Fallback: PIL unsharp mask gives a modest sharpness boost
        gray = gray.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

    return gray


# ─────────────────────────────────────────────────────────────────────────────
# 3. OCR
# ─────────────────────────────────────────────────────────────────────────────

def _run_tesseract(pil_image: Image.Image) -> str:
    """Run pytesseract and return raw text, or empty string on failure."""
    if not _TESS_AVAILABLE:
        return ""
    try:
        # PSM 6: Assume a single uniform block of text — best for structured reports
        config = r"--oem 3 --psm 6"
        return pytesseract.image_to_string(pil_image, config=config)
    except Exception as exc:
        logger.error("Tesseract error: %s", exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# 4. KEYWORD EXTRACTION + VALUE MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_unit(raw_unit: str) -> str:
    """Lower-case lookup in alias table; return original if not found."""
    if not raw_unit:
        return ""
    return _UNIT_ALIASES.get(raw_unit.lower().strip(), raw_unit.strip())


def _parse_lab_results(text: str) -> list[dict[str, Any]]:
    """
    Scan ``text`` for each known lab keyword, then look ahead for a numeric
    value + optional unit on the same line.

    Returns a list of dicts with keys: test_name, test_value, unit.
    Duplicate test names keep the first match (most prominently listed result).
    """
    results: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    lines = text.splitlines()
    full_text = text  # also search full text for multi-word patterns that span formatting

    for pattern, canonical_name, default_unit in LAB_PATTERNS:
        if canonical_name in seen_names:
            continue

        compiled = re.compile(pattern, re.IGNORECASE)

        # Search line-by-line first (most reliable)
        matched_line: str | None = None
        match_end: int = 0
        for line in lines:
            m = compiled.search(line)
            if m:
                matched_line = line
                match_end = m.end()
                break

        if matched_line is None:
            # Last-resort: search entire text blob
            m = compiled.search(full_text)
            if not m:
                continue
            # Grab text from match end to end-of-line
            eol = full_text.find("\n", m.end())
            if eol == -1:
                eol = len(full_text)
            matched_line = full_text[m.start():eol]
            match_end = m.end() - m.start()

        # Extract value + unit from the remainder of the matched line
        remainder = matched_line[match_end:]
        vm = _VALUE_UNIT_RE.search(remainder)
        if not vm:
            continue

        raw_value = vm.group("value").replace(",", ".")   # handle European decimals
        try:
            test_value = float(raw_value)
        except ValueError:
            continue

        raw_unit = (vm.group("unit") or "").strip()
        unit = _normalise_unit(raw_unit) if raw_unit else default_unit

        results.append({
            "test_name":  canonical_name,
            "test_value": test_value,
            "unit":       unit,
        })
        seen_names.add(canonical_name)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_MESSAGE = "Could not parse values, please enter manually."


def process_lab_image(image_bytes: bytes) -> tuple[str, list[dict]]:
    """
    Full OCR pipeline.

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        (raw_text, parsed_results)

        raw_text       – the raw Tesseract output string (used by extract_numbers)
        parsed_results – list of dicts [{test_name, test_value, unit}, ...]
                         Empty list means no structured data could be extracted.

    The caller (main.py) checks `if not parsed_results` to decide whether to
    show the fallback message.
    """
    try:
        pil_image = _preprocess_image(image_bytes)
    except Exception as exc:
        logger.error("Image preprocessing failed: %s", exc)
        return FALLBACK_MESSAGE, []

    raw_text = _run_tesseract(pil_image)

    if not raw_text.strip():
        return FALLBACK_MESSAGE, []

    parsed = _parse_lab_results(raw_text)

    if not parsed:
        return raw_text, []

    return raw_text, parsed


def extract_numbers(raw_text: str) -> dict[str, Any]:
    """
    Legacy quick-lookup dict consumed by main.py's auto-fill strip and
    session_state bridge keys (prefill_hba1c / prefill_glucose / prefill_cholesterol).

    Returns a flat dict with well-known short keys:
        {
            "hba1c":       float | None,
            "glucose":     float | None,
            "cholesterol": float | None,
            ... (one key per LAB_PATTERNS canonical name, snake_cased)
        }

    Also includes an "all_results" key with the full parsed list so callers
    that want every detected test can iterate over it.
    If no values are found, all short keys are None and "all_results" is [].
    """
    if not raw_text or raw_text == FALLBACK_MESSAGE:
        return {
            "hba1c": None,
            "glucose": None,
            "cholesterol": None,
            "all_results": [],
        }

    parsed = _parse_lab_results(raw_text)

    # Build lookup: canonical_name → test_value
    name_map: dict[str, float] = {r["test_name"]: r["test_value"] for r in parsed}

    def _get(*names: str) -> float | None:
        for n in names:
            v = name_map.get(n)
            if v is not None:
                return v
        return None

    return {
        # Short keys expected by main.py session_state bridge
        "hba1c":       _get("HbA1c"),
        "glucose":     _get("Fasting Blood Sugar", "Blood Glucose", "Post-Prandial Blood Sugar"),
        "cholesterol": _get("Total Cholesterol"),

        # Full parsed list — main.py stores this in st.session_state["ocr_prefill"]
        # as an iterable of OCR candidate dicts
        "all_results": parsed,

        # Individual extended keys (bonus: available if main.py ever needs them)
        "ldl":         _get("LDL Cholesterol"),
        "hdl":         _get("HDL Cholesterol"),
        "triglycerides": _get("Triglycerides"),
        "hemoglobin":  _get("Hemoglobin"),
        "tsh":         _get("TSH"),
        "creatinine":  _get("Serum Creatinine"),
        "vitamin_d":   _get("Vitamin D"),
        "vitamin_b12": _get("Vitamin B12"),
    }