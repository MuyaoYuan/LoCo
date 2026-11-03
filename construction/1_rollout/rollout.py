import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.reasoning import build_prompt, count_tokens, evaluate_prediction


DATASET_NAME = "data/math14k"
DATASET_SPLIT = "train"
DATASET_TAG = "math14k"
DATASET_QUESTION_FIELD = "question"
DATASET_ANSWER_FIELD = "answer"


def _default_base_model() -> str:
    env = os.environ.get("MODEL_NAME") or os.environ.get("BASE_MODEL")
    if env:
        return env
    return "model/DeepSeek-R1-Distill-Qwen-7B"


MODEL_NAME = _default_base_model()
MODEL_TAG = "deepseek_r1_distill_qwen_7b"
OUTPUT_ROOT = "out/1_rollout"
DEFAULT_RUN_ID_PREFIX = "run"
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_ROLLOUTS = 8
DEFAULT_RESPONSE_MODE = "full"
DEFAULT_CHECKPOINT_EVERY_BATCHES = 10
DEFAULT_MAX_MODEL_LEN = 18000
DEFAULT_MAX_TOKENS = 16000
HF_CACHE_DIR = os.environ.get("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id-prefix", default=DEFAULT_RUN_ID_PREFIX,
                        help="Prefix for per-rollout run_id directories (e.g. 'run' -> run_00, run_01, ...).")
    parser.add_argument("--num-rollouts", type=int, default=DEFAULT_NUM_ROLLOUTS,
                        help="vLLM SamplingParams.n -- number of rollouts per prompt in a single pass.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Prompts per llm.generate call. Active sequence count = batch_size * num_rollouts.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dataset-name", default=DATASET_NAME,
                        help="Dataset path (local dir or hub id).")
    parser.add_argument("--dataset-split", default=DATASET_SPLIT)
    parser.add_argument("--dataset-tag", default=DATASET_TAG,
                        help="Dataset tag recorded in run metadata.")
    parser.add_argument("--dataset-question-field", default=DATASET_QUESTION_FIELD,
                        help="Column holding the user-facing question text.")
    parser.add_argument("--dataset-answer-field", default=DATASET_ANSWER_FIELD,
                        help="Column holding the gold answer (target/answer/etc.).")
    parser.add_argument(
        "--response-mode",
        choices=["full", "cot"],
        default=DEFAULT_RESPONSE_MODE,
    )
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_BATCHES,
        help="Per-worker mid-run checkpoint cadence. Each worker atomically "
             "rewrites its _partials/worker_<i>.json every N batches so a "
             "crash doesn't waste hours of generation. Set to 0 to disable.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help="vLLM LLM.max_model_len. Bump for long-CoT competition problems.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="SamplingParams.max_tokens (per-rollout generation cap). "
             "Should be < max_model_len minus typical prompt length.",
    )
    return parser.parse_args()


def build_run_paths(output_root: str, run_id: str, dataset_tag: str = DATASET_TAG):
    base_dir = Path(output_root) / run_id
    return {
        "base_dir": base_dir,
        "answers_path": base_dir / "answers.json",
        "summary_path": base_dir / "summary.json",
    }


def build_run_root(output_root: str, dataset_tag: str = DATASET_TAG):
    return Path(output_root)


def build_partials_dir(output_root: str, dataset_tag: str = DATASET_TAG):
    return build_run_root(output_root, dataset_tag) / "_partials"


def get_worker_paths(partials_dir: Path, worker_id: int):
    return {
        "output": partials_dir / f"worker_{worker_id}.json",
        "summary": partials_dir / f"worker_{worker_id}_summary.json",
    }


def worker_outputs_ready(partials_dir: Path, worker_id: int) -> bool:
    worker_paths = get_worker_paths(partials_dir, worker_id)
    return worker_paths["output"].exists() and worker_paths["summary"].exists()


def atomic_write_json(path: Path, payload) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    os.replace(tmp_path, path)


def load_existing_answers(answers_path: Path) -> list[dict]:
    if not answers_path.exists():
        return []
    try:
        return json.loads(answers_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def recompute_rollout_counters(answers: list[dict], num_rollouts: int, tokenizer):
    correct_per_rollout = [0] * num_rollouts
    tokens_per_rollout = [0] * num_rollouts
    for ans in answers:
        ridx = ans.get("rollout_idx")
        if ridx is None or ridx >= num_rollouts:
            continue
        if ans.get("accept"):
            correct_per_rollout[ridx] += 1
        tokens_per_rollout[ridx] += int(ans.get("tokens", 0) or 0)
    return correct_per_rollout, tokens_per_rollout


def get_visible_gpus() -> list[str]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        visible_gpus = [gpu.strip() for gpu in cuda_visible_devices.split(",") if gpu.strip()]
    else:
        nvidia_smi = os.popen("nvidia-smi -L | wc -l").read().strip()
        visible_gpus = [str(i) for i in range(int(nvidia_smi or 0))]

    if not visible_gpus:
        raise RuntimeError("No visible CUDA devices found.")

    return visible_gpus








def format_saved_response(response: str, response_mode: str) -> str:
    if response_mode == "full":
        return response

    if "</think>" in response:
        return response.split("</think>", 1)[0] + "</think>"

    return response










def compute_shard_ranges(total_size: int, num_shards: int) -> list[tuple[int, int]]:
    shard_size = math.ceil(total_size / num_shards)
    ranges = []
    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, total_size)
        if start < end:
            ranges.append((start, end))
    return ranges


def run_worker(*args, **kwargs):
    try:
        _run_worker_body(*args, **kwargs)
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
    os._exit(0)


def _run_worker_body(
    worker_id: int,
    gpu_id: str,
    start: int,
    end: int,
    seed: int,
    batch_size: int,
    num_rollouts: int,
    run_id_prefix: str,
    partials_dir: str,
    max_samples: int | None,
    response_mode: str,
    checkpoint_every_batches: int,
    dataset_name: str,
    dataset_split: str,
    dataset_question_field: str,
    dataset_answer_field: str,
    max_model_len: int,
    max_tokens: int,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    dataset = load_dataset(
        dataset_name,
        split=dataset_split,
        cache_dir=HF_CACHE_DIR,
    ).select(range(start, end))
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    partials_path = Path(partials_dir)
    worker_paths = get_worker_paths(partials_path, worker_id)
    partial_output_path = worker_paths["output"]
    partial_summary_path = worker_paths["summary"]

    answers = load_existing_answers(partial_output_path)
    num_done_prompts = len(answers) // num_rollouts if num_rollouts > 0 else 0
    resume_batch_idx = num_done_prompts // batch_size
    aligned_prompts = resume_batch_idx * batch_size
    if num_done_prompts != aligned_prompts:
        keep = aligned_prompts * num_rollouts
        answers = answers[:keep]
        num_done_prompts = aligned_prompts
    expected_len = num_done_prompts * num_rollouts
    if len(answers) > expected_len:
        answers = answers[:expected_len]
    if num_done_prompts > len(dataset):
        answers = []
        num_done_prompts = 0

    if num_done_prompts:
        print(
            f"worker {worker_id} gpu {gpu_id} resuming from partial output: "
            f"{num_done_prompts}/{len(dataset)} prompts already done "
            f"({len(answers)} flat answer rows)",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = max(max_model_len, 1_000_000)
    correct_per_rollout, tokens_per_rollout = recompute_rollout_counters(
        answers, num_rollouts, tokenizer
    )

    if num_done_prompts >= len(dataset):
        atomic_write_json(partial_output_path, answers)
        worker_summary = {
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "start": start,
            "end": end,
            "num_samples": len(dataset),
            "num_rollouts": num_rollouts,
            "correct_per_rollout": correct_per_rollout,
            "tokens_per_rollout": tokens_per_rollout,
            "partial_output_path": str(partial_output_path),
        }
        atomic_write_json(partial_summary_path, worker_summary)
        return

    llm = LLM(model=MODEL_NAME, tensor_parallel_size=1, max_model_len=max_model_len)

    batches_since_checkpoint = 0
    for batch_start in range(num_done_prompts, len(dataset), batch_size):
        batch_examples = dataset[batch_start : batch_start + batch_size]
        batch_questions = batch_examples[dataset_question_field]
        batch_answers = batch_examples[dataset_answer_field]
        sampling_params = SamplingParams(
            n=num_rollouts,
            temperature=0.6,
            top_p=0.95,
            max_tokens=max_tokens,
            seed=seed + start + batch_start,
        )
        batch_prompts = [build_prompt(tokenizer, question) for question in batch_questions]
        outputs = llm.generate(batch_prompts, sampling_params)

        for offset, output in enumerate(outputs):
            sample_idx = start + batch_start + offset
            example = dataset[sample_idx - start]
            gt_answer = batch_answers[offset]

            for rollout_idx, generated in enumerate(output.outputs):
                response = generated.text
                result = evaluate_prediction(response, gt_answer)
                cot_tokens = count_tokens(tokenizer, result["cot_response"])
                final_tokens = count_tokens(tokenizer, result["final_response"])
                num_new_tokens = len(tokenizer.encode(response))
                finish_reason = getattr(generated, "finish_reason", None)
                run_id = f"{run_id_prefix}_{rollout_idx:02d}"

                if result["accept"]:
                    correct_per_rollout[rollout_idx] += 1
                tokens_per_rollout[rollout_idx] += num_new_tokens

                answers.append(
                    {
                        "index": sample_idx,
                        "rollout_idx": rollout_idx,
                        "run_id": run_id,
                        "question_id": example.get("__index_level_0__", sample_idx),
                        "question": batch_questions[offset],
                        "target": gt_answer,
                        "llm_answer": result["prediction_cot"],
                        "cot_answer": result["prediction_cot"],
                        "final_answer": result["prediction_final"],
                        "accept": result["accept"],
                        "is_complete": result["is_complete"],
                        "has_boxed_answer": result["has_boxed_answer_cot"] and result["has_boxed_answer_final"],
                        "has_boxed_answer_cot": result["has_boxed_answer_cot"],
                        "has_boxed_answer_final": result["has_boxed_answer_final"],
                        "final_matches_cot": result["final_matches_cot"],
                        "finish_reason": finish_reason,
                        "llm_response": format_saved_response(response, response_mode),
                        "cot_response": result["cot_response"],
                        "final_response": result["final_response"],
                        "cot_tokens": cot_tokens,
                        "final_tokens": final_tokens,
                        "tokens": num_new_tokens,
                        "total_tokens": num_new_tokens,
                        "raw_prediction": result["prediction_cot"],
                    }
                )

            seen_so_far = batch_start + offset + 1
            print(
                f"worker {worker_id} gpu {gpu_id} sample {sample_idx + 1} "
                f"rollout_acc=" + ",".join(
                    f"{correct_per_rollout[i] / seen_so_far:.4f}" for i in range(num_rollouts)
                )
            )

        batches_since_checkpoint += 1
        if (
            checkpoint_every_batches > 0
            and batches_since_checkpoint >= checkpoint_every_batches
        ):
            atomic_write_json(partial_output_path, answers)
            done = min(batch_start + batch_size, len(dataset))
            print(
                f"worker {worker_id} gpu {gpu_id} checkpoint: "
                f"{done}/{len(dataset)} prompts written to {partial_output_path}",
                flush=True,
            )
            batches_since_checkpoint = 0

    atomic_write_json(partial_output_path, answers)

    worker_summary = {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "start": start,
        "end": end,
        "num_samples": len(dataset),
        "num_rollouts": num_rollouts,
        "correct_per_rollout": correct_per_rollout,
        "tokens_per_rollout": tokens_per_rollout,
        "partial_output_path": str(partial_output_path),
    }
    atomic_write_json(partial_summary_path, worker_summary)


def merge_results(
    num_workers: int,
    visible_gpus: list[str],
    total_size: int,
    partials_dir: Path,
    seed: int,
    batch_size: int,
    num_rollouts: int,
    response_mode: str,
):
    answers_by_run: dict[str, list[dict]] = {}
    correct_by_run: dict[str, int] = {}
    tokens_by_run: dict[str, int] = {}
    worker_summaries = []

    for worker_id in range(num_workers):
        worker_paths = get_worker_paths(partials_dir, worker_id)
        with open(worker_paths["output"], "r", encoding="utf-8") as f:
            shard_answers = json.load(f)
        with open(worker_paths["summary"], "r", encoding="utf-8") as f:
            worker_summary = json.load(f)
        worker_summaries.append(worker_summary)

        for ans in shard_answers:
            answers_by_run.setdefault(ans["run_id"], []).append(ans)

        for rollout_idx, c in enumerate(worker_summary.get("correct_per_rollout", [])):
            run_id = f"{shard_answers[0]['run_id'].rsplit('_', 1)[0]}_{rollout_idx:02d}" if shard_answers else f"run_{rollout_idx:02d}"
            correct_by_run[run_id] = correct_by_run.get(run_id, 0) + c
        for rollout_idx, t in enumerate(worker_summary.get("tokens_per_rollout", [])):
            run_id = f"{shard_answers[0]['run_id'].rsplit('_', 1)[0]}_{rollout_idx:02d}" if shard_answers else f"run_{rollout_idx:02d}"
            tokens_by_run[run_id] = tokens_by_run.get(run_id, 0) + t

    for run_id, lst in answers_by_run.items():
        lst.sort(key=lambda x: x["index"])
        for a in lst:
            a.pop("index", None)

    return answers_by_run, correct_by_run, tokens_by_run, worker_summaries


def run_once(
    seed: int,
    batch_size: int,
    num_rollouts: int,
    output_root: str,
    run_id_prefix: str = DEFAULT_RUN_ID_PREFIX,
    max_samples: int | None = None,
    response_mode: str = DEFAULT_RESPONSE_MODE,
    checkpoint_every_batches: int = DEFAULT_CHECKPOINT_EVERY_BATCHES,
    dataset_name: str = DATASET_NAME,
    dataset_split: str = DATASET_SPLIT,
    dataset_tag: str = DATASET_TAG,
    dataset_question_field: str = DATASET_QUESTION_FIELD,
    dataset_answer_field: str = DATASET_ANSWER_FIELD,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    run_root = build_run_root(output_root, dataset_tag)
    run_root.mkdir(parents=True, exist_ok=True)
    partials_dir = build_partials_dir(output_root, dataset_tag)
    partials_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        dataset_name,
        split=dataset_split,
        cache_dir=HF_CACHE_DIR,
    )
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    total_size = len(dataset)
    visible_gpus = get_visible_gpus()
    shard_ranges = compute_shard_ranges(total_size, len(visible_gpus))

    ctx = mp.get_context("spawn")
    processes = []
    for worker_id, ((start, end), gpu_id) in enumerate(zip(shard_ranges, visible_gpus)):
        if worker_outputs_ready(partials_dir, worker_id):
            print(f"reusing existing partial outputs for worker {worker_id}")
            continue

        process = ctx.Process(
            target=run_worker,
            args=(
                worker_id,
                gpu_id,
                start,
                end,
                seed,
                batch_size,
                num_rollouts,
                run_id_prefix,
                str(partials_dir),
                max_samples,
                response_mode,
                checkpoint_every_batches,
                dataset_name,
                dataset_split,
                dataset_question_field,
                dataset_answer_field,
                max_model_len,
                max_tokens,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {process.exitcode}")

    answers_by_run, correct_by_run, tokens_by_run, worker_summaries = merge_results(
        num_workers=len(shard_ranges),
        visible_gpus=visible_gpus,
        total_size=total_size,
        partials_dir=partials_dir,
        seed=seed,
        batch_size=batch_size,
        num_rollouts=num_rollouts,
        response_mode=response_mode,
    )

    run_summaries = []
    for rollout_idx in range(num_rollouts):
        run_id = f"{run_id_prefix}_{rollout_idx:02d}"
        paths = build_run_paths(output_root, run_id, dataset_tag)
        paths["base_dir"].mkdir(parents=True, exist_ok=True)

        run_answers = answers_by_run.get(run_id, [])
        num_correct = correct_by_run.get(run_id, 0)
        total_tokens = tokens_by_run.get(run_id, 0)
        avg_tokens = total_tokens / total_size if total_size else 0.0
        avg_accuracy = num_correct / total_size if total_size else 0.0
        run_summary = {
            "model_name": MODEL_NAME,
            "model_tag": MODEL_TAG,
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "dataset_tag": dataset_tag,
            "run_id": run_id,
            "rollout_idx": rollout_idx,
            "num_samples": total_size,
            "num_correct": num_correct,
            "avg_tokens": avg_tokens,
            "avg_accuracy": avg_accuracy,
            "seed": seed,
            "batch_size": batch_size,
            "num_rollouts": num_rollouts,
            "response_mode": response_mode,
            "max_model_len": max_model_len,
            "max_tokens": max_tokens,
            "answers_path": str(paths["answers_path"]),
            "summary_path": str(paths["summary_path"]),
        }
        with open(paths["answers_path"], "w", encoding="utf-8") as f:
            json.dump(run_answers, f, ensure_ascii=False, indent=2)
        with open(paths["summary_path"], "w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)

        print(
            f"#### {MODEL_TAG.upper()}_{dataset_tag.upper()}_{run_id.upper()} ####"
        )
        print(f"  num_samples: {total_size}  num_correct: {num_correct}  "
              f"avg_acc: {avg_accuracy:.4f}  avg_tokens: {avg_tokens:.1f}")
        run_summaries.append(run_summary)

    accuracies = [r["avg_accuracy"] for r in run_summaries]
    aggregate_dir = build_run_root(output_root, dataset_tag) / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate_summary_path = aggregate_dir / "multi_run_summary.json"
    aggregate_summary = {
        "model_tag": MODEL_TAG,
        "dataset_tag": dataset_tag,
        "num_runs": num_rollouts,
        "base_seed": seed,
        "batch_size": batch_size,
        "max_samples": max_samples,
        "response_mode": response_mode,
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "run_mode": "vllm_n_rollouts_single_pass",
        "average_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "min_accuracy": min(accuracies) if accuracies else 0.0,
        "max_accuracy": max(accuracies) if accuracies else 0.0,
        "accuracies": accuracies,
        "visible_gpu_count": len(visible_gpus),
        "visible_gpus": visible_gpus,
        "data_parallel_size": len(shard_ranges),
        "partials_dir": str(partials_dir),
        "worker_summaries": worker_summaries,
        "run_summaries": run_summaries,
    }
    with open(aggregate_summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, ensure_ascii=False, indent=2)

    print(f"aggregate summary: {aggregate_summary_path}")
    print(f"avg accuracy across {num_rollouts} rollouts: {aggregate_summary['average_accuracy']:.4f}")

    return aggregate_summary


def main():
    args = parse_args()
    run_once(
        seed=args.seed,
        batch_size=args.batch_size,
        num_rollouts=args.num_rollouts,
        output_root=args.output_root,
        run_id_prefix=args.run_id_prefix,
        max_samples=args.max_samples,
        response_mode=args.response_mode,
        checkpoint_every_batches=args.checkpoint_every_batches,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_tag=args.dataset_tag,
        dataset_question_field=args.dataset_question_field,
        dataset_answer_field=args.dataset_answer_field,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
