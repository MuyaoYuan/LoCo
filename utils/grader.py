from math_verify import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    parse as _mv_parse,
    verify as _mv_verify,
)

from .parser import strip_string


_MV_EXTRACTION_CONFIGS = [LatexExtractionConfig(), ExprExtractionConfig()]
_MATH_VERIFY_TIMEOUT_SECONDS = 5


def _mv_parse_one(text: str):
    try:
        return _mv_parse(text, extraction_config=_MV_EXTRACTION_CONFIGS)
    except Exception:
        return []


def _mv_parse_robust(text: str):
    text = (text or "").strip()
    if not text:
        return []
    parsed = _mv_parse_one(text)
    if parsed:
        return parsed
    if "$" not in text:
        parsed = _mv_parse_one(f"${text}$")
        if parsed:
            return parsed
    return _mv_parse_one(f"\\boxed{{{text}}}") or []


def check_is_correct(pred, gt, timeout=True):
    pred_str = strip_string(pred)
    gt_str = strip_string(gt)
    try:
        gold = _mv_parse_robust(gt_str)
        if not gold:
            return False
        pred_parsed = _mv_parse_robust(pred_str)
        if not pred_parsed:
            return False
        return bool(
            _mv_verify(
                gold,
                pred_parsed,
                timeout_seconds=_MATH_VERIFY_TIMEOUT_SECONDS,
            )
        )
    except Exception:
        return False
