import argparse
import json
import random
from decimal import Decimal
from pathlib import Path

import ijson


def _json_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="judge json to sample")
    p.add_argument("--output", required=True, help="sampled judge json output")
    p.add_argument("--max-steps-per-question", type=int, default=10,
                   help="Max OVERTHINK steps kept per question; the rest are flipped to NORMAL.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"streaming {args.input} ...", flush=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_steps_kept = 0
    n_steps_orig = 0
    with open(args.input, "rb") as fin, open(out_path, "w", encoding="utf-8") as fout:
        fout.write('{"records": [')
        first = True
        for r in ijson.items(fin, "records.item"):
            ov_idxs = [i for i, s in enumerate(r.get("steps", []))
                       if s.get("judge_gold_label") == "OVERTHINK"]
            n_steps_orig += len(ov_idxs)
            if len(ov_idxs) > args.max_steps_per_question:
                keep_ov = set(rng.sample(ov_idxs, args.max_steps_per_question))
            else:
                keep_ov = set(ov_idxs)
            n_steps_kept += len(keep_ov)
            for i, s in enumerate(r.get("steps", [])):
                if s.get("judge_gold_label") == "OVERTHINK" and i not in keep_ov:
                    s["judge_gold_label"] = "NORMAL"
                    s["label"] = "NORMAL"
            if not first:
                fout.write(",")
            fout.write(json.dumps(r, ensure_ascii=False, default=_json_default))
            first = False
            n_written += 1
        fout.write("]}")

    summary = {
        "input": args.input,
        "output": args.output,
        "num_questions": n_written,
        "max_steps_per_question": args.max_steps_per_question,
        "seed": args.seed,
        "overthink_steps_original": n_steps_orig,
        "overthink_steps_kept": n_steps_kept,
        "estimated_regenerate_tasks": n_steps_kept,
    }
    sum_path = out_path.with_suffix(".sampling_summary.json")
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
