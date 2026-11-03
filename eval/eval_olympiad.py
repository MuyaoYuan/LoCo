import argparse
import re

from eval.eval_common import add_common_args, run_bench
from utils.extraction import extract_all_boxed_content, extract_answer_llm
from utils.grader import check_is_correct


DEFAULT_DATA_PATH = "data/Math_Olympiad/test.jsonl"
PROMPT_SUFFIX = r" Please reason step by step, and put your final answer within \boxed{}."

_MARKER_PATTERNS = [
    re.compile(r"<answer>(.*?)</answer>", re.DOTALL),
    re.compile(r"\*\*Answer:\*\*\s*(.+)"),
    re.compile(r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*(.+)"),
]


def _strip_boxed_wrapper(s: str) -> str:
    s = s or ""
    if s.startswith(r"\boxed{") and s.endswith("}"):
        return s[len(r"\boxed{"):-1]
    return s


def extract_boxed(text: str):
    boxed = extract_all_boxed_content(str(text or ""))
    if not boxed:
        return None
    return _strip_boxed_wrapper(boxed[-1])


def extract_markers(text: str):
    text = str(text or "")
    for pat in _MARKER_PATTERNS:
        m = pat.findall(text)
        if m:
            return m[-1].strip()
    return None


def extract_bare(text: str):
    pred = extract_answer_llm(str(text or ""))
    if pred and pred != "INVALID_ANS":
        return pred
    return None


def judge(pred, gt) -> bool:
    if pred is None:
        return False
    return check_is_correct(pred, gt)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, DEFAULT_DATA_PATH, default_batch_size=500)
    args = p.parse_args()
    run_bench("Olympiad", PROMPT_SUFFIX, extract_boxed, extract_markers, extract_bare, judge, args)


if __name__ == "__main__":
    main()
