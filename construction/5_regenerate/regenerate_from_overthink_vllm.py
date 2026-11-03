import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path


AutoTokenizer = None
LLM = None
SamplingParams = None
build_prompt = None
count_tokens = None
evaluate_prediction = None


def _load_runtime_deps():
    global AutoTokenizer, LLM, SamplingParams
    global build_prompt, count_tokens, evaluate_prediction
    if AutoTokenizer is not None:
        return
    from transformers import AutoTokenizer as _AutoTokenizer
    from vllm import LLM as _LLM, SamplingParams as _SamplingParams
    from utils.reasoning import (
        build_prompt as _build_prompt,
        count_tokens as _count_tokens,
        evaluate_prediction as _evaluate_prediction,
    )
    AutoTokenizer = _AutoTokenizer
    LLM = _LLM
    SamplingParams = _SamplingParams
    build_prompt = _build_prompt
    count_tokens = _count_tokens
    evaluate_prediction = _evaluate_prediction


def _default_model_name() -> str:
    return "model/DeepSeek-R1-Distill-Qwen-7B"


MODEL_NAME = _default_model_name()

PROMPT_LEN_SAFETY_MARGIN = 32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--model_name_or_path", default=MODEL_NAME)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=18000)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument(
        "--checkpoint_every_batches",
        type=int,
        default=10,
        help="Atomically rewrite worker_{i}_results.json every K batches; <=0 disables.",
    )
    parser.add_argument(
        "--heartbeat_sec",
        type=int,
        default=300,
        help="While a batch is generating, a background thread logs an 'still running, elapsed Xs' "
             "line every this many seconds. <=0 disables the heartbeat.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log per-task detail lines (one line per task). Off by default; batch-level "
             "summaries and heartbeats are always logged to the worker log file.",
    )
    parser.add_argument(
        "--step_label_source",
        choices=["auto", "model", "judge"],
        default="judge",
        help="judge: only use judge_gold_label; auto: prefer judge_gold_label when present, else fallback to label",
    )
    return parser.parse_args()


def resolve_step_label(step: dict, step_label_source: str):
    judge_label = step.get("judge_gold_label")
    model_label = step.get("label")

    if step_label_source == "judge":
        return judge_label, "judge_gold_label"
    if step_label_source == "model":
        return model_label, "label"
    if judge_label:
        return judge_label, "judge_gold_label"
    return model_label, "label"


def build_regenerate_tasks(
    payload: dict,
    max_samples: int | None,
    max_tasks: int | None,
    step_label_source: str,
):
    records = payload.get("records", [])
    if max_samples is not None:
        records = records[:max_samples]

    tasks = []
    for record in records:
        steps = record.get("steps", [])
        for step in steps:
            selected_label, label_field = resolve_step_label(step, step_label_source)
            if selected_label != "OVERTHINK":
                continue

            overthink_step_idx = step.get("step_idx")
            prefix_steps = steps[:overthink_step_idx]
            prefix_cot_response = "\n\n".join(item.get("text", "") for item in prefix_steps).strip()
            original_suffix_steps = steps[overthink_step_idx:]
            original_suffix_cot_response = "\n\n".join(item.get("text", "") for item in original_suffix_steps).strip()
            source_cot_response = record.get("cot_response", "")

            tasks.append(
                {
                    "question_id": record.get("question_id"),
                    "run_id": record.get("run_id"),
                    "question": record.get("question"),
                    "target": record.get("target"),
                    "cot_answer": record.get("cot_answer"),
                    "final_answer": record.get("final_answer"),
                    "source_cot_response": source_cot_response,
                    "source_cot_tokens": None,
                    "source_n_steps": record.get("n_steps"),
                    "overthink_step_idx": overthink_step_idx,
                    "overthink_step_text": step.get("text", ""),
                    "overthink_coef": step.get("coef"),
                    "source_step_label": selected_label,
                    "source_step_label_field": label_field,
                    "prefix_cot_response": prefix_cot_response,
                    "prefix_steps": [item.get("text", "") for item in prefix_steps],
                    "original_suffix_cot_response": original_suffix_cot_response,
                    "original_suffix_steps": [item.get("text", "") for item in original_suffix_steps],
                }
            )
            if max_tasks is not None and len(tasks) >= max_tasks:
                return records, tasks

    return records, tasks


def fill_source_cot_tokens(tasks: list[dict], tokenizer) -> None:
    cache: dict[tuple, int] = {}
    for task in tasks:
        if task.get("source_cot_tokens") is not None:
            continue
        key = (task.get("question_id"), task.get("run_id"))
        if key not in cache:
            cache[key] = count_tokens(tokenizer, task.get("source_cot_response", ""))
        task["source_cot_tokens"] = cache[key]



def build_rollout_record(tokenizer, rollout_idx: int, generated, prefix_cot_response: str, target: str):
    suffix_text = generated.text
    llm_response = f"{prefix_cot_response}{suffix_text}" if prefix_cot_response else suffix_text
    result = evaluate_prediction(llm_response, target)
    cot_tokens = count_tokens(tokenizer, result["cot_response"])
    final_tokens = count_tokens(tokenizer, result["final_response"])
    total_tokens = len(tokenizer.encode(llm_response))

    return {
        "rollout_idx": rollout_idx,
        "finish_reason": getattr(generated, "finish_reason", None),
        "llm_response": llm_response,
        "cot_response": result["cot_response"],
        "final_response": result["final_response"],
        "cot_answer": result["prediction_cot"],
        "final_answer": result["prediction_final"],
        "accept": result["accept"],
        "is_complete": result["is_complete"],
        "has_boxed_answer_cot": result["has_boxed_answer_cot"],
        "has_boxed_answer_final": result["has_boxed_answer_final"],
        "final_matches_cot": result["final_matches_cot"],
        "cot_tokens": cot_tokens,
        "final_tokens": final_tokens,
        "total_tokens": total_tokens,
    }


def is_valid_candidate(candidate: dict):
    return (
        candidate["accept"]
        and candidate["is_complete"]
        and candidate["final_matches_cot"]
    )


def choose_best_candidate(candidates: list[dict]):
    valid_candidates = [item for item in candidates if is_valid_candidate(item)]
    if not valid_candidates:
        return None, "no_valid_rollout"

    ordered = sorted(
        valid_candidates,
        key=lambda item: (
            item["cot_tokens"],
            item["total_tokens"],
            item["rollout_idx"],
        ),
    )
    return ordered[0], "shortest_valid_rollout"


def summarize_selection(task: dict, best: dict | None, selection_reason: str, candidates: list[dict]):
    summary = {
        **task,
        "num_rollouts": len(candidates),
        "num_valid_rollouts": sum(is_valid_candidate(item) for item in candidates),
        "selection_reason": selection_reason,
        "status": "abandoned" if best is None else "selected",
        "rollouts": candidates,
    }

    if best is None:
        summary.update(
            {
                "selected_rollout_idx": None,
                "selected_llm_response": None,
                "selected_cot_response": None,
                "selected_final_response": None,
                "selected_cot_answer": None,
                "selected_final_answer": None,
                "selected_accept": False,
                "selected_is_complete": False,
                "selected_has_boxed_answer_cot": False,
                "selected_has_boxed_answer_final": False,
                "selected_final_matches_cot": False,
                "selected_finish_reason": None,
                "selected_cot_tokens": None,
                "selected_final_tokens": None,
                "selected_total_tokens": None,
                "selected_token_delta_vs_source": None,
            }
        )
        return summary

    summary.update(
        {
            "selected_rollout_idx": best["rollout_idx"],
            "selected_llm_response": best["llm_response"],
            "selected_cot_response": best["cot_response"],
            "selected_final_response": best["final_response"],
            "selected_cot_answer": best["cot_answer"],
            "selected_final_answer": best["final_answer"],
            "selected_accept": best["accept"],
            "selected_is_complete": best["is_complete"],
            "selected_has_boxed_answer_cot": best["has_boxed_answer_cot"],
            "selected_has_boxed_answer_final": best["has_boxed_answer_final"],
            "selected_final_matches_cot": best["final_matches_cot"],
            "selected_finish_reason": best["finish_reason"],
            "selected_cot_tokens": best["cot_tokens"],
            "selected_final_tokens": best["final_tokens"],
            "selected_total_tokens": best["total_tokens"],
            "selected_token_delta_vs_source": best["cot_tokens"] - task["source_cot_tokens"],
        }
    )
    return summary


def build_sample_outputs(records: list[dict], task_results: list[dict]):
    grouped = defaultdict(list)
    for task in task_results:
        grouped[(task.get("question_id"), task.get("run_id"))].append(task)

    sample_outputs = []
    training_samples = []
    for record in records:
        key = (record.get("question_id"), record.get("run_id"))
        regen_tasks = sorted(grouped.get(key, []), key=lambda item: item["overthink_step_idx"])

        record_source_cot_tokens = regen_tasks[0]["source_cot_tokens"] if regen_tasks else None

        selected_regenerations = []
        for task in regen_tasks:
            if task["status"] != "selected":
                continue
            selected_regenerations.append(
                {
                    "overthink_step_idx": task["overthink_step_idx"],
                    "overthink_step_text": task["overthink_step_text"],
                    "prefix_cot_response": task["prefix_cot_response"],
                    "original_suffix_cot_response": task["original_suffix_cot_response"],
                    "selection_reason": task["selection_reason"],
                    "source_step_label": task["source_step_label"],
                    "source_step_label_field": task["source_step_label_field"],
                    "regenerated_cot_response": task["selected_cot_response"],
                    "regenerated_final_response": task["selected_final_response"],
                    "regenerated_llm_response": task["selected_llm_response"],
                    "regenerated_cot_answer": task["selected_cot_answer"],
                    "regenerated_final_answer": task["selected_final_answer"],
                    "regenerated_cot_tokens": task["selected_cot_tokens"],
                    "regenerated_total_tokens": task["selected_total_tokens"],
                    "token_delta_vs_source": task["selected_token_delta_vs_source"],
                }
            )
            training_samples.append(
                {
                    "question_id": task["question_id"],
                    "run_id": task["run_id"],
                    "question": task["question"],
                    "target": task["target"],
                    "source_cot_response": task["source_cot_response"],
                    "source_cot_tokens": task["source_cot_tokens"],
                    "overthink_step_idx": task["overthink_step_idx"],
                    "overthink_step_text": task["overthink_step_text"],
                    "prefix_cot_response": task["prefix_cot_response"],
                    "original_suffix_cot_response": task["original_suffix_cot_response"],
                    "source_step_label": task["source_step_label"],
                    "source_step_label_field": task["source_step_label_field"],
                    "regenerated_cot_response": task["selected_cot_response"],
                    "regenerated_final_response": task["selected_final_response"],
                    "regenerated_llm_response": task["selected_llm_response"],
                    "regenerated_cot_answer": task["selected_cot_answer"],
                    "regenerated_final_answer": task["selected_final_answer"],
                    "regenerated_cot_tokens": task["selected_cot_tokens"],
                    "regenerated_total_tokens": task["selected_total_tokens"],
                    "token_delta_vs_source": task["selected_token_delta_vs_source"],
                }
            )

        sample_outputs.append(
            {
                "question_id": record.get("question_id"),
                "run_id": record.get("run_id"),
                "question": record.get("question"),
                "target": record.get("target"),
                "cot_answer": record.get("cot_answer"),
                "final_answer": record.get("final_answer"),
                "source_cot_response": record.get("cot_response"),
                "source_n_steps": record.get("n_steps"),
                "source_cot_tokens": record_source_cot_tokens,
                "num_regenerate_tasks": len(regen_tasks),
                "num_selected_regenerations": len(selected_regenerations),
                "regenerations": regen_tasks,
                "selected_regenerations": selected_regenerations,
            }
        )

    return sample_outputs, training_samples


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


def compute_shard_ranges(total_size: int, num_shards: int) -> list[tuple[int, int]]:
    if total_size == 0 or num_shards == 0:
        return []
    shard_size = math.ceil(total_size / num_shards)
    ranges = []
    for shard_id in range(num_shards):
        start = shard_id * shard_size
        end = min(start + shard_size, total_size)
        if start < end:
            ranges.append((start, end))
    return ranges


def make_task_key(task: dict) -> str:
    return json.dumps(
        [
            "regenerate_from_overthink_v1",
            task.get("question_id"),
            task.get("run_id"),
            task.get("overthink_step_idx"),
            task.get("question"),
            task.get("target"),
            task.get("prefix_cot_response"),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def get_worker_paths(partials_dir: Path, worker_id: int):
    return {
        "tasks": partials_dir / f"worker_{worker_id}_tasks.json",
        "results": partials_dir / f"worker_{worker_id}_results.json",
        "summary": partials_dir / f"worker_{worker_id}_summary.json",
    }


def worker_outputs_ready(partials_dir: Path, worker_id: int) -> bool:
    worker_paths = get_worker_paths(partials_dir, worker_id)
    return worker_paths["results"].exists() and worker_paths["summary"].exists()


def atomic_write_json(path: Path, payload) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    os.replace(tmp_path, path)


def load_existing_results(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    try:
        return json.loads(results_path.read_text())
    except json.JSONDecodeError:
        return []


def collect_existing_results(partials_dir: Path) -> tuple[dict[str, dict], int, list[Path]]:
    results_by_key: dict[str, dict] = {}
    duplicate_count = 0
    result_files = sorted(partials_dir.glob("worker_*_results.json"))
    for results_path in result_files:
        for item in load_existing_results(results_path):
            key = make_task_key(item)
            if key in results_by_key:
                duplicate_count += 1
                continue
            results_by_key[key] = item
    return results_by_key, duplicate_count, result_files


def recompute_status_counts(results: list[dict]):
    num_selected = 0
    num_abandoned_no_valid_rollout = 0
    num_abandoned_not_shorter = 0
    num_abandoned_prompt_too_long = 0
    num_abandoned_generate_error = 0
    for item in results:
        if item.get("status") == "selected":
            num_selected += 1
        elif item.get("selection_reason") == "selected_longer_than_source":
            num_abandoned_not_shorter += 1
        elif item.get("selection_reason") == "prompt_too_long":
            num_abandoned_prompt_too_long += 1
        elif item.get("selection_reason") == "generate_error":
            num_abandoned_generate_error += 1
        else:
            num_abandoned_no_valid_rollout += 1
    return (
        num_selected,
        num_abandoned_no_valid_rollout,
        num_abandoned_not_shorter,
        num_abandoned_prompt_too_long,
        num_abandoned_generate_error,
    )


def process_task_outputs(task: dict, output, tokenizer):
    candidates = [
        build_rollout_record(
            tokenizer=tokenizer,
            rollout_idx=rollout_idx,
            generated=generated,
            prefix_cot_response=task["prefix_cot_response"],
            target=task["target"],
        )
        for rollout_idx, generated in enumerate(output.outputs)
    ]

    best, selection_reason = choose_best_candidate(candidates)
    status_counts = {"selected": 0, "abandoned_no_valid_rollout": 0, "abandoned_not_shorter": 0}

    if best is None:
        result = summarize_selection(task, None, selection_reason, candidates)
        status_counts["abandoned_no_valid_rollout"] += 1
    elif best["cot_tokens"] > task["source_cot_tokens"]:
        result = summarize_selection(task, None, "selected_longer_than_source", candidates)
        result["proposed_rollout_idx"] = best["rollout_idx"]
        result["proposed_cot_tokens"] = best["cot_tokens"]
        result["proposed_total_tokens"] = best["total_tokens"]
        status_counts["abandoned_not_shorter"] += 1
    else:
        result = summarize_selection(task, best, selection_reason, candidates)
        status_counts["selected"] += 1

    return result, status_counts


def setup_worker_logger(worker_id: int, partials_dir: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger(f"regen.worker.{worker_id}")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    log_path = partials_dir / f"worker_{worker_id}.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)
    return logger


class Heartbeat:

    def __init__(self, logger: logging.Logger, label: str, interval_sec: int):
        self.logger = logger
        self.label = label
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.interval and self.interval > 0:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        start = time.monotonic()
        while not self._stop.wait(self.interval):
            elapsed = time.monotonic() - start
            self.logger.info(f"{self.label} still running, elapsed {elapsed:.0f}s")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return False


def run_worker(
    worker_id: int,
    gpu_id: str,
    partials_dir: str,
    model_name_or_path: str,
    num_rollouts: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    batch_size: int,
    max_seq_len: int,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    checkpoint_every_batches: int,
    heartbeat_sec: int = 300,
    verbose: bool = False,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    partials_path = Path(partials_dir)
    logger = setup_worker_logger(worker_id, partials_path, verbose)
    try:
        _run_worker_impl(
            logger=logger,
            worker_id=worker_id,
            gpu_id=gpu_id,
            partials_path=partials_path,
            model_name_or_path=model_name_or_path,
            num_rollouts=num_rollouts,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            checkpoint_every_batches=checkpoint_every_batches,
            heartbeat_sec=heartbeat_sec,
        )
    except Exception:
        logger.error("worker crashed:\n%s", traceback.format_exc())
        raise


def _run_worker_impl(
    logger: logging.Logger,
    worker_id: int,
    gpu_id: str,
    partials_path: Path,
    model_name_or_path: str,
    num_rollouts: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    batch_size: int,
    max_seq_len: int,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    checkpoint_every_batches: int,
    heartbeat_sec: int,
):
    _load_runtime_deps()

    worker_paths = get_worker_paths(partials_path, worker_id)
    tasks: list[dict] = json.loads(worker_paths["tasks"].read_text())
    task_keys = [make_task_key(task) for task in tasks]
    task_key_set = set(task_keys)

    results_by_key: dict[str, dict] = {}
    duplicate_results = 0
    stale_results = 0
    for item in load_existing_results(worker_paths["results"]):
        key = make_task_key(item)
        if key not in task_key_set:
            stale_results += 1
            continue
        if key in results_by_key:
            duplicate_results += 1
            continue
        results_by_key[key] = item
    results = [results_by_key[key] for key in task_keys if key in results_by_key]
    already_done = len(results)
    (
        num_selected,
        num_abandoned_no_valid_rollout,
        num_abandoned_not_shorter,
        num_abandoned_prompt_too_long,
        num_abandoned_generate_error,
    ) = recompute_status_counts(results)

    if already_done:
        logger.info(
            "resuming from partial results: %d/%d tasks already done",
            already_done, len(tasks),
        )
    if stale_results or duplicate_results:
        logger.info(
            "normalized partial cache: ignored_stale=%d ignored_duplicates=%d",
            stale_results, duplicate_results,
        )

    if already_done >= len(tasks):
        atomic_write_json(worker_paths["results"], results)
        worker_summary = {
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "num_tasks": len(tasks),
            "num_completed_tasks": len(results),
            "num_selected": num_selected,
            "num_abandoned_no_valid_rollout": num_abandoned_no_valid_rollout,
            "num_abandoned_not_shorter": num_abandoned_not_shorter,
            "num_abandoned_prompt_too_long": num_abandoned_prompt_too_long,
            "num_abandoned_generate_error": num_abandoned_generate_error,
            "results_path": str(worker_paths["results"]),
        }
        atomic_write_json(worker_paths["summary"], worker_summary)
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    fill_source_cot_tokens(tasks, tokenizer)

    pending_tasks = [task for task, key in zip(tasks, task_keys) if key not in results_by_key]
    num_batches = math.ceil(len(pending_tasks) / batch_size) if pending_tasks else 0
    logger.info(
        "START gpu=%s tasks=%d done=%d pending=%d batch_size=%d num_batches=%d "
        "max_tokens=%d max_seq_len=%d heartbeat_sec=%d",
        gpu_id, len(tasks), already_done, len(pending_tasks), batch_size, num_batches,
        max_tokens, max_seq_len, heartbeat_sec,
    )

    logger.info("loading vLLM engine (model=%s dtype=%s tp=%d) ...", model_name_or_path, dtype, tensor_parallel_size)
    load_t0 = time.monotonic()
    with Heartbeat(logger, f"worker {worker_id} engine load", heartbeat_sec):
        llm = LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            trust_remote_code=True,
            dtype=dtype,
            max_model_len=max_seq_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    logger.info("vLLM engine ready in %.1fs", time.monotonic() - load_t0)

    sampling_params = SamplingParams(
        n=num_rollouts,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )

    def _process_one(task, output):
        nonlocal num_selected, num_abandoned_no_valid_rollout, num_abandoned_not_shorter
        result, status_counts = process_task_outputs(task, output, tokenizer)
        num_selected += status_counts["selected"]
        num_abandoned_no_valid_rollout += status_counts["abandoned_no_valid_rollout"]
        num_abandoned_not_shorter += status_counts["abandoned_not_shorter"]
        results_by_key[make_task_key(task)] = result
        logger.debug(
            "question_id=%s run_id=%s overthink_step=%s status=%s selection=%s "
            "selected_cot_tokens=%s source_cot_tokens=%s",
            task["question_id"], task["run_id"], task["overthink_step_idx"],
            result["status"], result["selection_reason"],
            result["selected_cot_tokens"], task["source_cot_tokens"],
        )
        return status_counts

    batches_since_checkpoint = 0
    for batch_idx, batch_start in enumerate(range(0, len(pending_tasks), batch_size), start=1):
        batch_tasks = pending_tasks[batch_start: batch_start + batch_size]

        valid_pairs = []
        batch_prompt_too_long = 0
        for task in batch_tasks:
            prompt = build_prompt(tokenizer, task["question"])
            if task["prefix_cot_response"]:
                prompt = f"{prompt}{task['prefix_cot_response']}"
            prompt_tokens = count_tokens(tokenizer, prompt)
            if prompt_tokens + PROMPT_LEN_SAFETY_MARGIN > max_seq_len:
                result = summarize_selection(task, None, "prompt_too_long", [])
                result["prompt_tokens"] = prompt_tokens
                results_by_key[make_task_key(task)] = result
                num_abandoned_prompt_too_long += 1
                batch_prompt_too_long += 1
                logger.debug(
                    "question_id=%s run_id=%s overthink_step=%s status=abandoned "
                    "selection=prompt_too_long prompt_tokens=%d max_seq_len=%d",
                    task["question_id"], task["run_id"], task["overthink_step_idx"],
                    prompt_tokens, max_seq_len,
                )
                continue
            valid_pairs.append((task, prompt))

        prompts = [prompt for _, prompt in valid_pairs]
        valid_tasks = [task for task, _ in valid_pairs]
        logger.info(
            "batch %d/%d start, %d prompts (prompt_too_long=%d)",
            batch_idx, num_batches, len(prompts), batch_prompt_too_long,
        )

        batch_sel = batch_no_valid = batch_not_shorter = batch_gen_err = 0
        batch_dt = 0.0
        if prompts:
            batch_t0 = time.monotonic()
            batch_gen_exc = None
            outputs = None
            with Heartbeat(logger, f"worker {worker_id} batch {batch_idx}/{num_batches}", heartbeat_sec):
                try:
                    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                except ValueError as exc:
                    batch_gen_exc = exc
            batch_dt = time.monotonic() - batch_t0

            if outputs is not None:
                for task, output in zip(valid_tasks, outputs):
                    sc = _process_one(task, output)
                    batch_sel += sc["selected"]
                    batch_no_valid += sc["abandoned_no_valid_rollout"]
                    batch_not_shorter += sc["abandoned_not_shorter"]
            else:
                logger.error(
                    "batch %d/%d generate raised ValueError; retrying prompts "
                    "individually to isolate the offender: %s",
                    batch_idx, num_batches, batch_gen_exc,
                )
                for task, prompt in zip(valid_tasks, prompts):
                    try:
                        single = llm.generate([prompt], sampling_params, use_tqdm=False)
                    except ValueError as exc2:
                        result = summarize_selection(task, None, "generate_error", [])
                        result["generate_error"] = str(exc2)
                        results_by_key[make_task_key(task)] = result
                        num_abandoned_generate_error += 1
                        batch_gen_err += 1
                        logger.error(
                            "prompt skipped (generate_error): "
                            "question_id=%s run_id=%s overthink_step=%s : %s",
                            task["question_id"], task["run_id"],
                            task["overthink_step_idx"], exc2,
                        )
                        continue
                    sc = _process_one(task, single[0])
                    batch_sel += sc["selected"]
                    batch_no_valid += sc["abandoned_no_valid_rollout"]
                    batch_not_shorter += sc["abandoned_not_shorter"]

        done_now = len(results_by_key)
        logger.info(
            "batch %d/%d done in %.1fs, sel=%d no_valid=%d not_shorter=%d "
            "prompt_too_long=%d generate_error=%d | cum %d/%d",
            batch_idx, num_batches, batch_dt,
            batch_sel, batch_no_valid, batch_not_shorter,
            batch_prompt_too_long, batch_gen_err,
            done_now, len(tasks),
        )

        batches_since_checkpoint += 1
        if (
            checkpoint_every_batches > 0
            and batches_since_checkpoint >= checkpoint_every_batches
        ):
            results = [results_by_key[key] for key in task_keys if key in results_by_key]
            atomic_write_json(worker_paths["results"], results)
            logger.info(
                "checkpoint: %d/%d tasks written to %s",
                len(results), len(tasks), worker_paths["results"],
            )
            batches_since_checkpoint = 0

    results = [results_by_key[key] for key in task_keys if key in results_by_key]
    atomic_write_json(worker_paths["results"], results)

    worker_summary = {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "num_tasks": len(tasks),
        "num_completed_tasks": len(results),
        "num_selected": num_selected,
        "num_abandoned_no_valid_rollout": num_abandoned_no_valid_rollout,
        "num_abandoned_not_shorter": num_abandoned_not_shorter,
        "num_abandoned_prompt_too_long": num_abandoned_prompt_too_long,
        "num_abandoned_generate_error": num_abandoned_generate_error,
        "results_path": str(worker_paths["results"]),
    }
    atomic_write_json(worker_paths["summary"], worker_summary)
    logger.info(
        "DONE completed=%d/%d selected=%d no_valid=%d not_shorter=%d "
        "prompt_too_long=%d generate_error=%d",
        len(results), len(tasks), num_selected,
        num_abandoned_no_valid_rollout, num_abandoned_not_shorter,
        num_abandoned_prompt_too_long, num_abandoned_generate_error,
    )


def merge_worker_outputs(partials_dir: Path, num_workers: int):
    results = []
    worker_summaries = []
    num_selected = 0
    num_abandoned_no_valid_rollout = 0
    num_abandoned_not_shorter = 0
    num_abandoned_prompt_too_long = 0
    num_abandoned_generate_error = 0

    for worker_id in range(num_workers):
        worker_paths = get_worker_paths(partials_dir, worker_id)
        worker_results = json.loads(worker_paths["results"].read_text())
        worker_summary = json.loads(worker_paths["summary"].read_text())

        results.extend(worker_results)
        worker_summaries.append(worker_summary)
        num_selected += worker_summary.get("num_selected", 0)
        num_abandoned_no_valid_rollout += worker_summary.get("num_abandoned_no_valid_rollout", 0)
        num_abandoned_not_shorter += worker_summary.get("num_abandoned_not_shorter", 0)
        num_abandoned_prompt_too_long += worker_summary.get("num_abandoned_prompt_too_long", 0)
        num_abandoned_generate_error += worker_summary.get("num_abandoned_generate_error", 0)

    return (
        results,
        worker_summaries,
        num_selected,
        num_abandoned_no_valid_rollout,
        num_abandoned_not_shorter,
        num_abandoned_prompt_too_long,
        num_abandoned_generate_error,
    )


def main():
    args = parse_args()
    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    partials_dir = output_json.parent / f"{output_json.stem}_partials"
    partials_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_json.read_text())
    records, tasks = build_regenerate_tasks(
        payload=payload,
        max_samples=args.max_samples,
        max_tasks=args.max_tasks,
        step_label_source=args.step_label_source,
    )

    visible_gpus = get_visible_gpus()
    if args.tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be >= 1")
    num_workers = max(1, len(visible_gpus) // args.tensor_parallel_size)
    shard_ranges = compute_shard_ranges(len(tasks), num_workers)
    num_workers = len(shard_ranges)

    print(f"num_input_records: {len(records)}")
    print(f"num_tasks: {len(tasks)}")
    print(f"visible_gpus: {visible_gpus}")
    print(f"tensor_parallel_size: {args.tensor_parallel_size}")
    print(f"num_workers (data parallel): {num_workers}")
    print(f"partials_dir: {partials_dir}")

    if not tasks:
        results = []
        worker_summaries = []
        num_selected = 0
        num_abandoned_no_valid_rollout = 0
        num_abandoned_not_shorter = 0
        num_abandoned_prompt_too_long = 0
        num_abandoned_generate_error = 0
    else:
        existing_results_by_key, duplicate_existing_results, existing_result_files = collect_existing_results(partials_dir)
        current_task_keys = {make_task_key(task) for task in tasks}
        reusable_results_by_key = {
            key: value
            for key, value in existing_results_by_key.items()
            if key in current_task_keys
        }
        stale_existing_results = len(existing_results_by_key) - len(reusable_results_by_key)

        print(
            "resume cache scan: "
            f"result_files={len(existing_result_files)} "
            f"reusable_results={len(reusable_results_by_key)}/{len(tasks)} "
            f"stale_results={stale_existing_results} "
            f"duplicate_results={duplicate_existing_results}"
        )

        num_reusable = len(reusable_results_by_key)
        if num_reusable > 0:
            pending = [t for t in tasks if make_task_key(t) not in reusable_results_by_key]
            done = [t for t in tasks if make_task_key(t) in reusable_results_by_key]
            groups = [[] for _ in range(num_workers)]
            for i, task in enumerate(pending):
                groups[i % num_workers].append(task)
            for i, task in enumerate(done):
                groups[i % num_workers].append(task)
            tasks = [task for group in groups for task in group]
            print(
                f"balanced resume: {len(pending)} pending + {len(done)} done "
                f"reordered into {num_workers} groups "
                f"(~{len(pending) // num_workers} pending/worker)"
            )

        try:
            input_stat = input_json.stat()
            input_sig = {"size": input_stat.st_size, "mtime_ns": input_stat.st_mtime_ns}
        except OSError:
            input_sig = None
        current_shard_meta = {
            "input_json": str(input_json),
            "input_sig": input_sig,
            "num_workers": num_workers,
            "num_tasks": len(tasks),
            "step_label_source": args.step_label_source,
            "max_samples": args.max_samples,
            "max_tasks": args.max_tasks,
            "shard_ranges": [list(r) for r in shard_ranges],
            "balanced_resume_count": num_reusable,
        }
        shard_meta_path = partials_dir / "shard_meta.json"
        prev_shard_meta = None
        if shard_meta_path.exists():
            try:
                prev_shard_meta = json.loads(shard_meta_path.read_text())
            except json.JSONDecodeError:
                prev_shard_meta = None
        tasks_layout_unchanged = prev_shard_meta == current_shard_meta

        wrote_tasks = 0
        skipped_tasks = 0
        for worker_id, (start, end) in enumerate(shard_ranges):
            worker_paths = get_worker_paths(partials_dir, worker_id)
            shard_tasks = tasks[start:end]
            if tasks_layout_unchanged and worker_paths["tasks"].exists():
                skipped_tasks += 1
            else:
                worker_paths["tasks"].write_text(
                    json.dumps(shard_tasks, ensure_ascii=False, separators=(",", ":"))
                )
                wrote_tasks += 1

            seeded_results = []
            for task in shard_tasks:
                key = make_task_key(task)
                if key in reusable_results_by_key:
                    seeded_results.append(reusable_results_by_key[key])
            atomic_write_json(worker_paths["results"], seeded_results)

            if len(seeded_results) >= len(shard_tasks):
                (
                    seeded_num_selected,
                    seeded_num_abandoned_no_valid_rollout,
                    seeded_num_abandoned_not_shorter,
                    seeded_num_abandoned_prompt_too_long,
                    seeded_num_abandoned_generate_error,
                ) = recompute_status_counts(seeded_results)
                worker_summary = {
                    "worker_id": worker_id,
                    "gpu_id": None,
                    "num_tasks": len(shard_tasks),
                    "num_completed_tasks": len(seeded_results),
                    "num_selected": seeded_num_selected,
                    "num_abandoned_no_valid_rollout": seeded_num_abandoned_no_valid_rollout,
                    "num_abandoned_not_shorter": seeded_num_abandoned_not_shorter,
                    "num_abandoned_prompt_too_long": seeded_num_abandoned_prompt_too_long,
                    "num_abandoned_generate_error": seeded_num_abandoned_generate_error,
                    "results_path": str(worker_paths["results"]),
                }
                atomic_write_json(worker_paths["summary"], worker_summary)
            elif worker_paths["summary"].exists():
                worker_paths["summary"].unlink()

        atomic_write_json(shard_meta_path, current_shard_meta)
        print(f"tasks shard files: wrote={wrote_tasks} skipped_unchanged={skipped_tasks}")

        ctx = mp.get_context("spawn")
        processes = []
        for worker_id, (start, end) in enumerate(shard_ranges):
            if worker_outputs_ready(partials_dir, worker_id):
                print(f"reusing existing partial outputs for worker {worker_id} (range {start}:{end})")
                continue

            gpu_block = visible_gpus[
                worker_id * args.tensor_parallel_size:
                (worker_id + 1) * args.tensor_parallel_size
            ]
            gpu_id = ",".join(gpu_block)

            print(f"spawning worker {worker_id} on gpu {gpu_id} for tasks [{start}:{end})")
            process = ctx.Process(
                target=run_worker,
                args=(
                    worker_id,
                    gpu_id,
                    str(partials_dir),
                    args.model_name_or_path,
                    args.num_rollouts,
                    args.temperature,
                    args.top_p,
                    args.max_tokens,
                    args.seed + worker_id,
                    args.batch_size,
                    args.max_seq_len,
                    args.dtype,
                    args.tensor_parallel_size,
                    args.gpu_memory_utilization,
                    args.checkpoint_every_batches,
                    args.heartbeat_sec,
                    args.verbose,
                ),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Worker process failed with exit code {process.exitcode}")

        (
            results,
            worker_summaries,
            num_selected,
            num_abandoned_no_valid_rollout,
            num_abandoned_not_shorter,
            num_abandoned_prompt_too_long,
            num_abandoned_generate_error,
        ) = merge_worker_outputs(partials_dir, num_workers)

    sample_outputs, training_samples = build_sample_outputs(records, results)
    output_payload = {
        "input_json": str(input_json),
        "model_name_or_path": args.model_name_or_path,
        "step_label_source": args.step_label_source,
        "num_input_records": len(records),
        "num_tasks": len(tasks),
        "num_completed_tasks": len(results),
        "num_rollouts": args.num_rollouts,
        "num_selected": num_selected,
        "num_abandoned_no_valid_rollout": num_abandoned_no_valid_rollout,
        "num_abandoned_not_shorter": num_abandoned_not_shorter,
        "num_abandoned_prompt_too_long": num_abandoned_prompt_too_long,
        "num_abandoned_generate_error": num_abandoned_generate_error,
        "selection_rate": num_selected / len(results) if results else 0.0,
        "num_training_samples": len(training_samples),
        "visible_gpu_count": len(visible_gpus),
        "visible_gpus": visible_gpus,
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": num_workers,
        "partials_dir": str(partials_dir),
        "worker_summaries": worker_summaries,
        "tasks": results,
        "samples": sample_outputs,
        "training_samples": training_samples,
    }
    output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2))

    print("num_input_records:", output_payload["num_input_records"])
    print("num_tasks:", output_payload["num_tasks"])
    print("num_completed_tasks:", output_payload["num_completed_tasks"])
    print("num_selected:", output_payload["num_selected"])
    print("num_abandoned_no_valid_rollout:", output_payload["num_abandoned_no_valid_rollout"])
    print("num_abandoned_not_shorter:", output_payload["num_abandoned_not_shorter"])
    print("num_abandoned_prompt_too_long:", output_payload["num_abandoned_prompt_too_long"])
    print("num_abandoned_generate_error:", output_payload["num_abandoned_generate_error"])
    print("num_training_samples:", output_payload["num_training_samples"])
    print("selection_rate:", output_payload["selection_rate"])
    print("visible_gpu_count:", output_payload["visible_gpu_count"])
    print("data_parallel_size:", output_payload["data_parallel_size"])
    print("output:", output_json)


if __name__ == "__main__":
    main()
