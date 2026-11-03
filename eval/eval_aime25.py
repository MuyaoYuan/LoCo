import argparse

from eval.eval_aime24 import (
    PROMPT_SUFFIX, extract_boxed, extract_markers, extract_bare, judge,
)
from eval.eval_common import add_common_args, run_bench


DEFAULT_DATA_PATH = "data/Math_AIME2025/test.jsonl"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, DEFAULT_DATA_PATH, default_batch_size=30)
    args = p.parse_args()
    run_bench("AIME2025", PROMPT_SUFFIX, extract_boxed, extract_markers, extract_bare, judge, args)


if __name__ == "__main__":
    main()
