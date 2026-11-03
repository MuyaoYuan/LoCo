import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

PROMPT_SUFFIX = r" Please reason step by step, and put your final answer within \boxed{}."


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partials-dir",
        required=True,
        help="Directory with worker_*_results.json (regenerate output, "
        "provides chosen + prefix_cot).",
    )
    parser.add_argument(
        "--source-records",
        required=True,
        help="Path to data_filter's shortest.json / all_valid.json. "
        "Provides the ORIGINAL full responses used as rejected, keyed by "
        "(question_id, run_id).",
    )
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer path/name. Required when --max-prompt-tokens / "
        "--max-response-tokens is set; used purely for length filtering.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after writing this many DPO rows. Useful for smoke tests.",
    )

    parser.add_argument(
        "--min-token-saved",
        type=int,
        default=300,
        help="Drop pairs where the regenerated suffix doesn't save at least "
        "this many tokens vs the original suffix. Pairs that don't differ "
        "much don't give DPO useful signal. Default 300.",
    )
    parser.add_argument(
        "--min-token-delta-ratio",
        type=float,
        default=None,
        help="Optional alternative / additional filter: drop pairs whose "
        "savings ratio (saved / source_cot_tokens) is below this. e.g. 0.10 "
        "keeps only pairs that save >=10%% of the source tokens.",
    )
    parser.add_argument(
        "--max-pairs-per-question",
        type=int,
        default=3,
        help="Cap how many DPO pairs a single question_id can contribute. "
        "Long CoTs naturally produce more OVERTHINK boundaries; capping "
        "reduces dataset skew toward long-question records. Default 3.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Drop pairs where (question + prompt_suffix + prefix_cot) "
        "exceeds this token budget. Requires --tokenizer.",
    )
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=None,
        help="Drop pairs where EITHER chosen OR rejected exceeds this many "
        "tokens. Requires --tokenizer. Useful for fitting training context.",
    )
    parser.add_argument(
        "--min-pair-token-gap",
        type=int,
        default=300,
        help="DPO needs chosen to be meaningfully shorter than rejected (in "
        "tokens). Drop pairs where rejected_tokens - chosen_tokens is below "
        "this threshold. This counts the FULL continuation (cot + final), "
        "while --min-token-saved only counts cot. Setting both to the same "
        "value is a reasonable default. Default 300. Requires --tokenizer.",
    )
    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="Include a 'meta' object on each row.",
    )
    parser.add_argument(
        "--max-pair-token-gap",
        type=int,
        default=None,
        help="Optional UPPER bound on pair_token_gap: drop pairs whose gap is "
        ">= this. Used to build rescue datasets that only keep WEAK-signal "
        "pairs (e.g. 100 <= gap < 300) excluded from the main dataset.",
    )
    parser.add_argument(
        "--exclude-question-ids-from",
        default=None,
        help="Optional path to an existing DPO jsonl. Any question_id already "
        "present there is EXCLUDED from this build (no question overlap).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reserved for future shuffling; currently unused.",
    )
    return parser.parse_args()



def _stream_partials(partials_dir: Path) -> Iterator[dict]:
    files = sorted(partials_dir.glob("worker_*_results.json"))
    if not files:
        raise FileNotFoundError(f"no worker_*_results.json under {partials_dir}")
    for f in files:
        print(f"  reading {f.name} ({f.stat().st_size / 1024**3:.1f} GB)...", flush=True)
        results = json.loads(f.read_text())
        for task in results:
            yield task
        del results


def _load_source_records(source_records: Path) -> dict:
    print(f"  reading source records: {source_records} "
          f"({source_records.stat().st_size / 1024**2:.1f} MB)...",
          flush=True)
    rows = json.loads(source_records.read_text())
    if isinstance(rows, dict):
        rows = rows.get("records") or rows.get("data") or []
    index: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("question_id"), r.get("run_id"))
        index.setdefault(key, r)
    print(f"  indexed {len(index)} (question_id, run_id) source records")
    return index



def _strip_prefix(full: str, prefix: str) -> str | None:
    if not full:
        return None
    if not prefix:
        return full
    if full.startswith(prefix):
        return full[len(prefix):]
    p_strip = prefix.rstrip()
    if p_strip and full.startswith(p_strip):
        return full[len(p_strip):]
    return None


def derive_chosen(task: dict) -> str | None:
    sel_full = task.get("selected_llm_response") or ""
    prefix = task.get("prefix_cot_response") or ""
    return _strip_prefix(sel_full, prefix)


def derive_rejected(task: dict, source_record: dict | None) -> str | None:
    if source_record is None:
        return None
    src_full = source_record.get("llm_response") or ""
    prefix = task.get("prefix_cot_response") or ""
    return _strip_prefix(src_full, prefix)


def build_row(task: dict, source_record: dict | None, include_meta: bool) -> dict | None:
    chosen = derive_chosen(task)
    if not chosen:
        return None
    rejected = derive_rejected(task, source_record)
    if not rejected:
        return None
    if chosen == rejected:
        return None
    row = {
        "question": task.get("question") or "",
        "prompt_suffix": PROMPT_SUFFIX,
        "prefix_cot": task.get("prefix_cot_response") or "",
        "chosen": chosen,
        "rejected": rejected,
    }
    if include_meta:
        row["meta"] = {
            "question_id": task.get("question_id"),
            "run_id": task.get("run_id"),
            "overthink_step_idx": task.get("overthink_step_idx"),
            "source_cot_tokens": task.get("source_cot_tokens"),
            "selected_cot_tokens": task.get("selected_cot_tokens"),
            "token_delta_vs_source": task.get("selected_token_delta_vs_source"),
        }
    return row



def main():
    args = parse_args()

    if (args.max_response_tokens is not None or args.max_prompt_tokens is not None) \
       and args.tokenizer is None:
        raise SystemExit(
            "--max-response-tokens / --max-prompt-tokens require --tokenizer "
            "(we need it to count tokens)."
        )

    tokenizer = None
    if args.tokenizer is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    source_index = _load_source_records(Path(args.source_records))
    source_iter = _stream_partials(Path(args.partials_dir))
    source_label = (
        f"partials:{args.partials_dir} | source:{args.source_records}"
    )

    exclude_qids = None
    n_drop_excluded_q = 0
    if args.exclude_question_ids_from is not None:
        exclude_qids = set()
        with open(args.exclude_question_ids_from, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                qid = json.loads(line).get("meta", {}).get("question_id")
                if qid is not None:
                    exclude_qids.add(qid)
        print(f"  excluding {len(exclude_qids)} question_ids from "
              f"{args.exclude_question_ids_from}", flush=True)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n_seen = 0
    n_selected = 0
    n_kept = 0
    n_drop_not_selected = 0
    n_drop_no_source = 0
    n_drop_no_pair = 0
    n_drop_chosen_not_shorter = 0
    n_drop_short_saved = 0
    n_drop_short_ratio = 0
    n_drop_too_long_resp = 0
    n_drop_too_long_prompt = 0
    n_drop_per_question_cap = 0
    n_drop_gap_too_large = 0

    candidates_by_qid: dict[object, list[dict]] = defaultdict(list)

    for task in source_iter:
        n_seen += 1
        if n_seen % 5000 == 0:
            total_so_far = sum(len(v) for v in candidates_by_qid.values())
            print(f"  processed {n_seen}, candidates so far: {total_so_far}", flush=True)

        if task.get("status") != "selected":
            n_drop_not_selected += 1
            continue
        n_selected += 1

        if exclude_qids is not None and task.get("question_id") in exclude_qids:
            n_drop_excluded_q += 1
            continue

        key = (task.get("question_id"), task.get("run_id"))
        source_record = source_index.get(key)
        if source_record is None:
            n_drop_no_source += 1
            continue

        delta = task.get("selected_token_delta_vs_source")
        saved = -delta if delta is not None else None

        if saved is not None and args.min_token_saved is not None \
           and saved < args.min_token_saved:
            n_drop_short_saved += 1
            continue

        if args.min_token_delta_ratio is not None and saved is not None:
            src_tok = task.get("source_cot_tokens") or 0
            ratio = saved / src_tok if src_tok else 0
            if ratio < args.min_token_delta_ratio:
                n_drop_short_ratio += 1
                continue

        row = build_row(task, source_record, args.include_meta)
        if row is None:
            n_drop_no_pair += 1
            continue

        if args.max_prompt_tokens is not None:
            prompt_text = (
                row["question"] + row["prompt_suffix"] + row["prefix_cot"]
            )
            p_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            if p_tokens > args.max_prompt_tokens:
                n_drop_too_long_prompt += 1
                continue

        ch_tokens_n = None
        rj_tokens_n = None
        if (args.max_response_tokens is not None
                or args.min_pair_token_gap is not None
                or args.max_pairs_per_question is not None):
            if tokenizer is None:
                raise SystemExit(
                    "Need --tokenizer (max-response-tokens, min-pair-token-gap, "
                    "and max-pairs-per-question top-K all need it)."
                )
            ch_tokens_n = len(tokenizer.encode(row["chosen"], add_special_tokens=False))
            rj_tokens_n = len(tokenizer.encode(row["rejected"], add_special_tokens=False))

        if args.max_response_tokens is not None:
            if ch_tokens_n > args.max_response_tokens or rj_tokens_n > args.max_response_tokens:
                n_drop_too_long_resp += 1
                continue

        if args.min_pair_token_gap is not None and ch_tokens_n is not None:
            gap = rj_tokens_n - ch_tokens_n
            if gap < args.min_pair_token_gap:
                n_drop_chosen_not_shorter += 1
                continue

        if args.max_pair_token_gap is not None and ch_tokens_n is not None:
            gap = rj_tokens_n - ch_tokens_n
            if gap >= args.max_pair_token_gap:
                n_drop_gap_too_large += 1
                continue

        if args.include_meta and ch_tokens_n is not None:
            row["meta"]["chosen_tokens"] = ch_tokens_n
            row["meta"]["rejected_tokens"] = rj_tokens_n
            row["meta"]["pair_token_gap"] = rj_tokens_n - ch_tokens_n

        gap_for_rank = (
            (rj_tokens_n - ch_tokens_n)
            if ch_tokens_n is not None
            else (len(row["rejected"]) - len(row["chosen"]))
        )
        candidates_by_qid[task.get("question_id")].append((gap_for_rank, row))

    cap = args.max_pairs_per_question
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for qid in sorted(candidates_by_qid, key=lambda k: (k is None, k)):
            group = candidates_by_qid[qid]
            group.sort(
                key=lambda gr: (-gr[0],
                                gr[1].get("meta", {}).get("chosen_tokens", 0)),
            )
            if cap is not None and len(group) > cap:
                n_drop_per_question_cap += len(group) - cap
                group = group[:cap]
            for _gap, row in group:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_kept += 1
                if args.max_rows is not None and n_kept >= args.max_rows:
                    break
            if args.max_rows is not None and n_kept >= args.max_rows:
                print(f"  reached --max-rows {args.max_rows}, stopping early")
                break

    per_question_count = {qid: min(len(g), cap or len(g))
                          for qid, g in candidates_by_qid.items()
                          if min(len(g), cap or len(g)) > 0}

    summary = {
        "source": source_label,
        "tokenizer": args.tokenizer,
        "min_token_saved": args.min_token_saved,
        "min_token_delta_ratio": args.min_token_delta_ratio,
        "max_pairs_per_question": args.max_pairs_per_question,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_response_tokens": args.max_response_tokens,
        "min_pair_token_gap": args.min_pair_token_gap,
        "include_meta": args.include_meta,
        "prompt_suffix": PROMPT_SUFFIX,
        "n_seen": n_seen,
        "n_selected": n_selected,
        "n_kept": n_kept,
        "n_drop_not_selected": n_drop_not_selected,
        "n_drop_no_source": n_drop_no_source,
        "n_drop_no_pair": n_drop_no_pair,
        "n_drop_chosen_not_shorter": n_drop_chosen_not_shorter,
        "n_drop_short_saved": n_drop_short_saved,
        "n_drop_short_ratio": n_drop_short_ratio,
        "n_drop_too_long_prompt": n_drop_too_long_prompt,
        "n_drop_too_long_resp": n_drop_too_long_resp,
        "n_drop_per_question_cap": n_drop_per_question_cap,
        "n_drop_gap_too_large": n_drop_gap_too_large,
        "n_drop_excluded_question": (
            n_drop_excluded_q if exclude_qids is not None else 0
        ),
        "max_pair_token_gap": args.max_pair_token_gap,
        "exclude_question_ids_from": args.exclude_question_ids_from,
        "n_unique_questions": len(per_question_count),
        "output_jsonl": str(output_jsonl),
    }
    summary_path = output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print()
    print("=== build_dpo summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
