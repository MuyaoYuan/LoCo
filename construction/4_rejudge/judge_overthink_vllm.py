import argparse
import json
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


JUDGE_STEP_FIELDS = (
    "judge_is_correct",
    "judge_gold_label",
    "judge_reason",
    "judge_api_error",
    "judge_parse_error",
    "judge_attempts",
    "judge_raw",
    "judge_raw_response",
)

DEFAULT_MODEL_NAME_OR_PATH = "model/Qwen3-30B-A3B-Instruct-2507"
JUDGE_PROMPT = """You are evaluating whether an existing per-step overthink label is correct.

You will see:
1. a math problem
2. the full chain of thought split into steps
3. the current step
4. the predicted label for the current step

Your job is to judge whether the predicted label is correct.

## Problem
{question}

## Previous Reasoning Steps
{prev_context}

## Current Step (Step {step_idx})
{step_text}

## Predicted Label
{predicted_label}

## Label Meaning
- OVERTHINK: this step is wasteful, redundant, or unnecessary; removing it would not break the reasoning path to the answer.
- NORMAL: this step is necessary reasoning; removing it would break or significantly weaken the reasoning path.
- FINAL_ANSWER_WRAPUP: this step mainly wraps up or restates the final answer without adding new reasoning.

## Your Task
Decide whether the predicted label is correct.

Respond in this exact JSON format and nothing else:
{{"gold_label": "OVERTHINK|NORMAL|FINAL_ANSWER_WRAPUP", "reason": "short reason"}}
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_steps_per_sample", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_model_len", type=int, default=12288)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.82)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--checkpoint_every_batches", type=int, default=1000)
    parser.add_argument(
        "--linear_backend",
        choices=[
            "auto",
            "cutlass",
            "flashinfer_cutlass",
            "flashinfer_trtllm",
            "flashinfer_cudnn",
            "marlin",
            "triton",
            "deep_gemm",
            "torch",
            "machete",
            "fbgemm",
            "conch",
            "exllama",
            "emulation",
        ],
        default=None,
    )
    parser.add_argument(
        "--gdn_prefill_backend",
        choices=["flashinfer", "triton", "cutedsl"],
        default=None,
    )
    return parser.parse_args()


def truncate(text: str, n: int = 4000):
    if len(text) <= n:
        return text
    return text[: n // 2] + "\n...[truncated]...\n" + text[-n // 2 :]


def build_prev_context(steps: list[dict], step_idx: int, max_chars: int = 4000):
    if step_idx == 0:
        return "(this is the first step - no previous steps)"
    prev = "\n\n".join(step["text"] for step in steps[:step_idx])
    return truncate(prev, max_chars)


def parse_judge_response(text: str):
    if not text:
        return {"gold_label": None, "reason": None, "parse_error": "empty response"}
    try:
        obj = json.loads(text.strip())
        return {
            "gold_label": obj.get("gold_label"),
            "reason": obj.get("reason"),
            "parse_error": None,
        }
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
                return {
                    "gold_label": obj.get("gold_label"),
                    "reason": obj.get("reason"),
                    "parse_error": None,
                }
            except json.JSONDecodeError as exc:
                return {"gold_label": None, "reason": None, "parse_error": str(exc)}
    return {"gold_label": None, "reason": text[:200], "parse_error": "no JSON found"}


def build_step_prompt(record: dict, step: dict, step_idx: int):
    return JUDGE_PROMPT.format(
        question=truncate(record["question"], 2000),
        prev_context=build_prev_context(record["steps"], step_idx),
        step_idx=step_idx,
        step_text=truncate(step["text"], 2000),
        predicted_label=step["label"],
    )


def build_generation_prompt(tokenizer, messages: list[dict]) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    apply_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if isinstance(chat_template, str) and "enable_thinking" in chat_template:
        apply_kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **apply_kwargs)


def is_step_judged(step: dict) -> bool:
    return step.get("judge_is_correct") in (True, False)


def step_attempt_count(step: dict) -> int:
    attempts = step.get("judge_attempts")
    return attempts if isinstance(attempts, int) and attempts > 0 else 0


def step_progress_state(step: dict) -> str:
    if step.get("label") != "OVERTHINK":
        return "non_target"
    if is_step_judged(step):
        return "judged"
    if step_attempt_count(step) > 0:
        return "failed"
    return "pending"


def records_match_for_resume(record: dict, resume_record: dict) -> bool:
    for key in ("question_id", "run_id"):
        value = record.get(key)
        resume_value = resume_record.get(key)
        if value is not None and resume_value is not None and value != resume_value:
            return False

    question = record.get("question")
    resume_question = resume_record.get("question")
    if question and resume_question and question != resume_question:
        return False
    return True


def copy_resume_step_fields(step: dict, resume_step: dict) -> bool:
    copied = False
    for field in JUDGE_STEP_FIELDS:
        if field in resume_step:
            step[field] = resume_step[field]
            copied = True
    return copied


def merge_resume_fields(records: list[dict], resume_records: list[dict]):
    copied_steps = 0
    resumed_judged_steps = 0
    resumed_failed_steps = 0
    mismatched_records = 0

    for record_idx, record in enumerate(records):
        if record_idx >= len(resume_records):
            break

        resume_record = resume_records[record_idx]
        if not records_match_for_resume(record, resume_record):
            mismatched_records += 1
            print(
                f"WARNING: resume record mismatch at index {record_idx}; skip existing output for this record.",
                flush=True,
            )
            continue

        resume_steps_by_idx = {
            step.get("step_idx", idx): step
            for idx, step in enumerate(resume_record.get("steps", []))
        }
        for step_idx, step in enumerate(record.get("steps", [])):
            if step.get("label") != "OVERTHINK":
                continue
            resume_step = resume_steps_by_idx.get(step.get("step_idx", step_idx))
            if resume_step is None:
                continue
            if not copy_resume_step_fields(step, resume_step):
                continue
            copied_steps += 1
            state = step_progress_state(step)
            if state == "judged":
                resumed_judged_steps += 1
            elif state == "failed":
                resumed_failed_steps += 1

    return {
        "copied_steps": copied_steps,
        "resumed_judged_steps": resumed_judged_steps,
        "resumed_failed_steps": resumed_failed_steps,
        "mismatched_records": mismatched_records,
    }


def summarize_judge_progress(records: list[dict]):
    total_steps = 0
    judged_steps = 0
    correct_steps = 0
    failed_steps = 0

    for record in records:
        for step in record.get("steps", []):
            if step.get("label") != "OVERTHINK":
                continue
            total_steps += 1
            state = step_progress_state(step)
            if state == "judged":
                judged_steps += 1
                if step.get("judge_is_correct") is True:
                    correct_steps += 1
            elif state == "failed":
                failed_steps += 1

    return total_steps, judged_steps, correct_steps, failed_steps


def update_record_judge_stats(record: dict):
    judged_overthink_steps = [step for step in record.get("steps", []) if step.get("label") == "OVERTHINK"]
    judged_count = sum(is_step_judged(step) for step in judged_overthink_steps)
    correct_count = sum(step.get("judge_is_correct") is True for step in judged_overthink_steps)
    failed_count = sum(step_progress_state(step) == "failed" for step in judged_overthink_steps)
    record["judge_step_accuracy"] = correct_count / judged_count if judged_count else None
    record["judge_correct_steps"] = correct_count
    record["judge_failed_steps"] = failed_count
    record["judge_target_steps"] = len(judged_overthink_steps)


def write_checkpoint(
    payload: dict,
    output_json: Path,
    *,
    judge_model: str,
    total_steps: int,
    judged_steps: int,
    correct_steps: int,
    failed_steps: int,
):
    payload["judge_backend"] = "vllm"
    payload["judge_model"] = judge_model
    payload["total_steps"] = total_steps
    payload["judged_steps"] = judged_steps
    payload["correct_steps"] = correct_steps
    payload["failed_steps"] = failed_steps
    payload["accuracy"] = correct_steps / judged_steps if judged_steps else None

    tmp_path = output_json.with_suffix(output_json.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp_path.replace(output_json)


def main():
    args = parse_args()
    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_json.read_text())
    records = payload.get("records", [])
    if args.max_samples is not None:
        records = records[: args.max_samples]
    payload["records"] = records

    for record in records:
        steps = record.get("steps", [])
        if args.max_steps_per_sample is not None:
            steps = steps[: args.max_steps_per_sample]
            record["steps"] = steps
        record["n_steps"] = len(record.get("steps", []))

    if output_json.exists():
        try:
            resume_payload = json.loads(output_json.read_text())
        except json.JSONDecodeError as exc:
            print(
                f"WARNING: failed to parse existing output_json for resume ({exc}); start from input_json.",
                flush=True,
            )
        else:
            resume_summary = merge_resume_fields(records, resume_payload.get("records", []))
            print(
                "resume checkpoint:",
                f"copied_steps={resume_summary['copied_steps']}",
                f"judged={resume_summary['resumed_judged_steps']}",
                f"failed={resume_summary['resumed_failed_steps']}",
                f"mismatched_records={resume_summary['mismatched_records']}",
                flush=True,
            )

    jobs = []
    for record_idx, record in enumerate(records):
        steps = record.get("steps", [])
        for step_idx, step in enumerate(steps):
            if step.get("label") != "OVERTHINK":
                continue
            if is_step_judged(step):
                continue
            jobs.append(
                {
                    "record_idx": record_idx,
                    "step_idx": step_idx,
                    "prompt": build_step_prompt(record, step, step_idx),
                }
            )

    for record in records:
        update_record_judge_stats(record)

    total_steps, judged_steps, correct_steps, failed_steps = summarize_judge_progress(records)
    pending_steps = total_steps - judged_steps - failed_steps
    print(
        f"judge progress before run: total={total_steps} judged={judged_steps} failed={failed_steps} pending={pending_steps}",
        flush=True,
    )

    write_checkpoint(
        payload,
        output_json,
        judge_model=args.model_name_or_path,
        total_steps=total_steps,
        judged_steps=judged_steps,
        correct_steps=correct_steps,
        failed_steps=failed_steps,
    )

    if not jobs:
        print("all OVERTHINK steps are already judged; nothing to do.", flush=True)
        print("judge_backend:", payload["judge_backend"])
        print("judge_model:", payload["judge_model"])
        print("num_records:", len(records))
        print("total_steps:", total_steps)
        print("judged_steps:", judged_steps)
        print("correct_steps:", correct_steps)
        print("failed_steps:", failed_steps)
        print("accuracy:", payload["accuracy"])
        print("output:", output_json)
        return

    llm_kwargs = {
        "model": args.model_name_or_path,
        "tokenizer": args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    if args.linear_backend is not None:
        llm_kwargs["linear_backend"] = args.linear_backend
    if args.gdn_prefill_backend is not None:
        llm_kwargs["gdn_prefill_backend"] = args.gdn_prefill_backend

    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    for batch_start in tqdm(range(0, len(jobs), args.batch_size), desc="judge_overthink_vllm"):
        batch_jobs = jobs[batch_start: batch_start + args.batch_size]
        prompts = []
        for job in batch_jobs:
            messages = [{"role": "user", "content": job["prompt"]}]
            prompts.append(build_generation_prompt(tokenizer, messages))

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        touched_record_indices = set()
        for job, output in zip(batch_jobs, outputs):
            generated = output.outputs[0] if output.outputs else None
            text = generated.text if generated else ""
            parsed = parse_judge_response(text)
            record = records[job["record_idx"]]
            step = record["steps"][job["step_idx"]]
            touched_record_indices.add(job["record_idx"])

            previous_state = step_progress_state(step)
            previous_attempts = step_attempt_count(step)
            if previous_state == "failed":
                failed_steps -= 1

            gold_label = parsed.get("gold_label")
            if gold_label is None:
                failed_steps += 1
                judge_is_correct = None
            else:
                judged_steps += 1
                judge_is_correct = gold_label == step["label"]
                if judge_is_correct:
                    correct_steps += 1

            step["judge_is_correct"] = judge_is_correct
            step["judge_gold_label"] = gold_label
            step["judge_reason"] = parsed.get("reason")
            step["judge_api_error"] = None
            step["judge_parse_error"] = parsed.get("parse_error")
            step["judge_attempts"] = previous_attempts + 1
            step["judge_raw"] = text
            step["judge_raw_response"] = {
                "text": text,
                "finish_reason": getattr(generated, "finish_reason", None),
                "stop_reason": getattr(generated, "stop_reason", None),
            }

            print(
                f"record {job['record_idx'] + 1}/{len(records)} step {job['step_idx'] + 1}/{record['n_steps']} "
                f"pred={step['label']} gold={gold_label} correct={judge_is_correct} attempts={step['judge_attempts']}",
                flush=True,
            )

        for record_idx in touched_record_indices:
            update_record_judge_stats(records[record_idx])

        if (batch_start // args.batch_size + 1) % args.checkpoint_every_batches == 0:
            write_checkpoint(
                payload,
                output_json,
                judge_model=args.model_name_or_path,
                total_steps=total_steps,
                judged_steps=judged_steps,
                correct_steps=correct_steps,
                failed_steps=failed_steps,
            )

    write_checkpoint(
        payload,
        output_json,
        judge_model=args.model_name_or_path,
        total_steps=total_steps,
        judged_steps=judged_steps,
        correct_steps=correct_steps,
        failed_steps=failed_steps,
    )

    print("judge_backend:", payload["judge_backend"])
    print("judge_model:", payload["judge_model"])
    print("num_records:", len(records))
    print("total_steps:", total_steps)
    print("judged_steps:", judged_steps)
    print("correct_steps:", correct_steps)
    print("failed_steps:", failed_steps)
    print("accuracy:", payload["accuracy"])
    print("output:", output_json)


if __name__ == "__main__":
    main()
