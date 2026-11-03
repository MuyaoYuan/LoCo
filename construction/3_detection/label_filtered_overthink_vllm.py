import argparse
import json
import math as _m
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    parser.add_argument("--prompt_logprobs_topk", type=int, default=1)
    parser.add_argument(
        "--system_prompt",
        default="Please reason step by step, and put your final answer within \\boxed{}.",
    )
    return parser.parse_args()


def split_steps(solution: str):
    return [step for step in solution.split("\n\n") if step.strip()]


def build_prompt_inputs(tokenizer, question: str, solution: str, system_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    sol_start_idx = len(prefix_ids)
    steps = split_steps(solution)
    if not steps:
        return None

    full_text = prefix + "\n\n".join(steps)
    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    prompt_token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    step_char_ends = []
    cursor = len(prefix)
    for idx, step in enumerate(steps):
        cursor += len(step)
        step_char_ends.append(cursor)
        if idx < len(steps) - 1:
            cursor += 2

    step_end_idx = []
    j = 0
    for end_char in step_char_ends:
        target = end_char if end_char == step_char_ends[-1] else end_char + 2
        last = None
        while j < len(offsets) and offsets[j][1] <= target:
            last = j
            j += 1
        if last is None:
            return None
        step_end_idx.append(last)

    return {
        "prompt_token_ids": prompt_token_ids,
        "sol_start_idx": sol_start_idx,
        "step_end_idx": step_end_idx,
        "step_texts": steps,
    }


def _read_logprob_value(value):
    if value is None:
        return None
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def extract_prompt_top1_probs(prompt_logprobs, expected_len: int):
    probs = [None] * expected_len
    if prompt_logprobs is None:
        return probs

    entries = list(prompt_logprobs)
    if len(entries) == expected_len - 1:
        entries = [None] + entries
    elif len(entries) < expected_len:
        entries = entries + [None] * (expected_len - len(entries))
    elif len(entries) > expected_len:
        entries = entries[:expected_len]

    for idx, entry in enumerate(entries):
        if not entry:
            continue
        best_logprob = None
        for value in entry.values():
            logprob = _read_logprob_value(value)
            if logprob is None:
                continue
            if best_logprob is None or logprob > best_logprob:
                best_logprob = logprob
        if best_logprob is None:
            continue
        probs[idx] = float(_m.exp(best_logprob))

    return probs


def compute_engine_max_model_len(prepared, requested_max_seq_len: int, margin: int = 8):
    if not prepared:
        return min(requested_max_seq_len, 32)

    max_prompt_len = max(len(entry["build_info"]["prompt_token_ids"]) for entry in prepared)
    return min(requested_max_seq_len, max_prompt_len + margin)


def summarize_prompt_lengths(prepared):
    lengths = sorted(len(entry["build_info"]["prompt_token_ids"]) for entry in prepared)
    if not lengths:
        return None

    def pick(frac: float):
        idx = min(len(lengths) - 1, max(0, int((len(lengths) - 1) * frac)))
        return lengths[idx]

    return {
        "min": lengths[0],
        "p50": pick(0.50),
        "p90": pick(0.90),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": lengths[-1],
    }


def compute_step_stats(build_info, prompt_top1_probs):
    step_ranges = []
    start = build_info["sol_start_idx"]
    for end in build_info["step_end_idx"]:
        step_ranges.append((start, end))
        start = end + 1

    confs = []
    for s_idx, e_idx in step_ranges:
        window = [prob for prob in prompt_top1_probs[s_idx:e_idx] if prob is not None]
        confs.append(float(sum(window) / len(window)) if window else None)

    vols = []
    for i in range(len(confs)):
        if i == 0 or confs[i] is None or confs[i - 1] is None:
            vols.append(0.0)
        else:
            vols.append(((confs[i] - confs[i - 1]) ** 2) / 4.0)

    steps = []
    for idx, (s_idx, e_idx) in enumerate(step_ranges):
        steps.append(
            {
                "step_idx": idx,
                "text": build_info["step_texts"][idx],
                "label": "NORMAL",
                "mean_conf": confs[idx],
                "var_conf": vols[idx],
            }
        )
    return steps, confs, vols


def percentile(values, frac):
    vals = sorted(values)
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, int((len(vals) - 1) * frac)))
    return vals[idx]


def label_overthink(steps, confs, vols, q25_conf, q75_vol):
    n_overthink = 0
    for idx in range(1, len(steps)):
        c = confs[idx - 1]
        v = vols[idx - 1]
        if c is not None and c <= q25_conf and v >= q75_vol:
            steps[idx]["label"] = "OVERTHINK"
            n_overthink += 1
    return n_overthink


def main():
    args = parse_args()
    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if args.max_samples is not None:
        items = []
        import ijson
        with open(input_json, "r", encoding="utf-8") as f:
            for obj in ijson.items(f, "item"):
                items.append(obj)
                if len(items) >= args.max_samples:
                    break
        print(f"Stream-loaded {len(items)} items (max_samples={args.max_samples})")
    else:
        items = json.loads(input_json.read_text())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    prepared = []
    skipped = []
    for item in items:
        build_info = build_prompt_inputs(
            tokenizer=tokenizer,
            question=item["question"],
            solution=item["cot_response"],
            system_prompt=args.system_prompt,
        )
        if build_info is None:
            skipped.append(
                {
                    "question_id": item.get("question_id"),
                    "run_id": item.get("run_id"),
                    "skip_reason": "empty_steps_or_bad_offsets",
                }
            )
            continue
        if len(build_info["prompt_token_ids"]) > args.max_seq_len:
            skipped.append(
                {
                    "question_id": item.get("question_id"),
                    "run_id": item.get("run_id"),
                    "skip_reason": f"seq_len={len(build_info['prompt_token_ids'])} > max_seq_len",
                }
            )
            continue
        prepared.append({"item": item, "build_info": build_info})

    engine_max_model_len = compute_engine_max_model_len(prepared, args.max_seq_len)
    print("engine max_model_len:", engine_max_model_len)
    prompt_stats = summarize_prompt_lengths(prepared)
    if prompt_stats is not None:
        print("prompt lens:", ", ".join(f"{key}={value}" for key, value in prompt_stats.items()))
    print("max_num_batched_tokens:", args.max_num_batched_tokens)

    llm_kwargs = {
        "model": args.model_name_or_path,
        "tokenizer": args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "max_model_len": engine_max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    try:
        llm = LLM(**llm_kwargs)
    except TypeError as exc:
        if args.max_num_batched_tokens is None or "max_num_batched_tokens" not in str(exc):
            raise
        print("WARNING: current vLLM LLM(...) does not accept max_num_batched_tokens; retrying without it.")
        llm_kwargs.pop("max_num_batched_tokens", None)
        llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=args.prompt_logprobs_topk,
    )

    scored = []
    for start in tqdm(range(0, len(prepared), args.batch_size), desc="label_overthink_vllm"):
        batch = prepared[start:start + args.batch_size]
        outputs = llm.generate(
            [{"prompt_token_ids": entry["build_info"]["prompt_token_ids"]} for entry in batch],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        for entry, output in zip(batch, outputs):
            prompt_top1_probs = extract_prompt_top1_probs(
                output.prompt_logprobs,
                expected_len=len(entry["build_info"]["prompt_token_ids"]),
            )
            steps, confs, vols = compute_step_stats(entry["build_info"], prompt_top1_probs)
            item = entry["item"]
            scored.append(
                {
                    "question_id": item["question_id"],
                    "run_id": item.get("run_id"),
                    "question": item["question"],
                    "target": item["target"],
                    "cot_answer": item.get("cot_answer"),
                    "final_answer": item.get("final_answer"),
                    "cot_response": item["cot_response"],
                    "n_steps": len(steps),
                    "n_overthink": 0,
                    "steps": steps,
                    "_confs": confs,
                    "_vols": vols,
                }
            )

    all_confs = [c for rec in scored for c in rec["_confs"] if c is not None]
    all_vols = [v for rec in scored for v in rec["_vols"]]
    q25_conf = percentile(all_confs, 0.25)
    q75_vol = percentile(all_vols, 0.75)
    print(f"thresholds: q25_conf={q25_conf} q75_vol={q75_vol} (n_steps={len(all_confs)})")

    results = []
    for rec in scored:
        n_overthink = label_overthink(rec["steps"], rec["_confs"], rec["_vols"], q25_conf, q75_vol)
        rec["n_overthink"] = n_overthink
        del rec["_confs"]
        del rec["_vols"]
        results.append(rec)

    payload = {
        "input_json": str(input_json),
        "model_name_or_path": args.model_name_or_path,
        "backend": "vllm_prompt_logprobs",
        "engine_max_model_len": engine_max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "q25_conf": q25_conf,
        "q75_conf": percentile(all_confs, 0.75),
        "q25_vol": percentile(all_vols, 0.25),
        "q75_vol": q75_vol,
        "num_items": len(items),
        "num_labeled": len(results),
        "num_skipped": len(skipped),
        "records": results,
        "skipped": skipped,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print("backend:", payload["backend"])
    print("num items:", payload["num_items"])
    print("num labeled:", payload["num_labeled"])
    print("num skipped:", payload["num_skipped"])
    print("output:", output_json)


if __name__ == "__main__":
    main()
