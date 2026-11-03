import argparse

from rollout import (
    DATASET_ANSWER_FIELD,
    DATASET_NAME,
    DATASET_QUESTION_FIELD,
    DATASET_SPLIT,
    DATASET_TAG,
    DEFAULT_CHECKPOINT_EVERY_BATCHES,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NUM_ROLLOUTS,
    DEFAULT_RESPONSE_MODE,
    DEFAULT_RUN_ID_PREFIX,
    OUTPUT_ROOT,
    run_once,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_ROLLOUTS,
                        help="Alias for --num-rollouts; total rollouts per prompt.")
    parser.add_argument("--num-rollouts", type=int, default=None,
                        help="Overrides --num-runs if given.")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Single seed passed to vLLM SamplingParams.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Prompts per vLLM call; n*batch_size sequences are in flight.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--run-id-prefix", default=DEFAULT_RUN_ID_PREFIX)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-split", default=DATASET_SPLIT)
    parser.add_argument("--dataset-tag", default=DATASET_TAG)
    parser.add_argument("--dataset-question-field", default=DATASET_QUESTION_FIELD)
    parser.add_argument("--dataset-answer-field", default=DATASET_ANSWER_FIELD)
    parser.add_argument(
        "--response-mode",
        choices=["full", "cot"],
        default=DEFAULT_RESPONSE_MODE,
    )
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_BATCHES,
        help="Per-worker mid-run checkpoint cadence; 0 disables.",
    )
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser.parse_args()


def main():
    args = parse_args()
    num_rollouts = args.num_rollouts if args.num_rollouts is not None else args.num_runs

    run_once(
        seed=args.base_seed,
        batch_size=args.batch_size,
        num_rollouts=num_rollouts,
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
