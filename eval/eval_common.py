import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


_THINK_CLOSE = "</think>"


def add_common_args(p: argparse.ArgumentParser, default_data_path: str,
                    default_batch_size: int):
    p.add_argument(
        "--base-model", required=True,
        help="Model path (merged full weights) to evaluate.",
    )
    p.add_argument("--data-path", default=default_data_path)
    p.add_argument("--output", required=True, help="Output JSON path.")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=36864)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=753)
    p.add_argument("--batch-size", type=int, default=default_batch_size)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap eval to first N examples (smoke test).")
    return p


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def split_cot_final(response: str):
    response = response or ""
    idx = response.rfind(_THINK_CLOSE)
    if idx == -1:
        return response, ""
    return response[:idx], response[idx + len(_THINK_CLOSE):]


def _first(*xs):
    for x in xs:
        if x is not None and x != "":
            return x
    return None


def judge_response(response, gt, extract_boxed, extract_markers, extract_bare, judge):
    pred = _first(extract_boxed(response), extract_markers(response),
                  extract_bare(response))
    correct = bool(judge(pred, gt))
    _, final = split_cot_final(response)
    return correct, pred, bool(final.strip())


def run_bench(dataset_name, prompt_suffix, extract_boxed, extract_markers,
              extract_bare, judge, args, *, question_key="problem"):
    test_examples = read_jsonl(args.data_path)
    if args.limit is not None:
        test_examples = test_examples[: args.limit]
    n = len(test_examples)

    llm = LLM(
        model=args.base_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    prompts, gts = [], []
    for ex in test_examples:
        messages = [{"role": "user", "content": ex[question_key] + prompt_suffix}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
        gts.append(ex["answer"])

    n_correct = 0
    total_tokens = 0

    out_path = Path(args.output)
    sidecar_path = (out_path.with_suffix(".responses.jsonl")
                    if out_path.suffix == ".json"
                    else Path(str(out_path) + ".responses.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    with sidecar_path.open("w", encoding="utf-8") as sf:
        for i in range(0, n, args.batch_size):
            sampling_params = SamplingParams(
                n=1,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed + i,
            )
            batch_prompts = prompts[i : i + args.batch_size]
            batch_gts = gts[i : i + args.batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)

            for j, output in enumerate(outputs):
                gt_answer = batch_gts[j]
                idx = i + j
                response = output.outputs[0].text
                num_new_tokens = len(tokenizer.encode(response))
                correct, pred, has_final = judge_response(
                    response, gt_answer,
                    extract_boxed, extract_markers, extract_bare, judge,
                )
                n_correct += int(correct)
                total_tokens += num_new_tokens

                sf.write(json.dumps({
                    "idx": idx,
                    "question": test_examples[idx][question_key],
                    "gt_answer": gt_answer,
                    "tokens": num_new_tokens,
                    "has_final": has_final,
                    "pred": pred,
                    "correct": correct,
                    "llm_response": response,
                }, ensure_ascii=False) + "\n")

    accuracy = n_correct / n
    avg_tokens = total_tokens / n

    print("#" * 80)
    print(dataset_name)
    print(f"  model:        {args.base_model}")
    print(f"  num samples:  {n}   seed: {args.seed}")
    print(f"  accuracy      : {accuracy:.4f}")
    print(f"  avg_tokens    : {avg_tokens:.1f}")

    summary = {
        "base_model": args.base_model,
        "data_path": args.data_path,
        "num_samples": n,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "prompt_suffix": prompt_suffix,
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "responses_file": sidecar_path.name,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  saved -> {out_path}")
    print(f"  responses -> {sidecar_path}")
    return summary
