import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_run_answers(input_dir: Path):
    run_files = sorted(input_dir.glob("run_*/answers.json"))
    if not run_files:
        raise FileNotFoundError(f"No run_*/answers.json found under {input_dir}")

    grouped = {}
    for run_file in run_files:
        run_id = run_file.parent.name
        answers = json.loads(run_file.read_text())
        for answer in answers:
            question_id = answer["question_id"]
            grouped.setdefault(question_id, []).append(
                {
                    "run_id": run_id,
                    **answer,
                }
            )

    return grouped, run_files


def is_valid_record(record):
    return (
        record.get("accept") is True
        and record.get("is_complete") is True
        and record.get("has_boxed_answer_cot") is True
        and record.get("has_boxed_answer_final") is True
        and record.get("final_matches_cot") is True
    )


def build_filtered_entry(record):
    return {
        "question_id": record["question_id"],
        "run_id": record["run_id"],
        "question": record["question"],
        "target": record["target"],
        "cot_answer": record["cot_answer"],
        "final_answer": record["final_answer"],
        "llm_response": record["llm_response"],
        "cot_response": record["cot_response"],
        "final_response": record["final_response"],
        "cot_tokens": record["cot_tokens"],
        "final_tokens": record["final_tokens"],
        "total_tokens": record["total_tokens"],
        "finish_reason": record["finish_reason"],
    }


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "filtered"
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped, run_files = load_run_answers(input_dir)

    shortest = []
    longest = []
    all_valid = []
    discarded = []

    for question_id, records in sorted(grouped.items()):
        valid_records = [record for record in records if is_valid_record(record)]
        if not valid_records:
            discarded.append(
                {
                    "question_id": question_id,
                    "num_runs": len(records),
                    "discard_reason": "no_valid_correct_complete_record",
                }
            )
            continue

        all_valid.extend(build_filtered_entry(record) for record in valid_records)

        valid_records.sort(key=lambda item: (item["cot_tokens"], item["total_tokens"], item["run_id"]))
        shortest.append(build_filtered_entry(valid_records[0]))
        longest.append(build_filtered_entry(valid_records[-1]))

    summary = {
        "input_dir": str(input_dir),
        "num_runs": len(run_files),
        "num_questions": len(grouped),
        "num_all_valid": len(all_valid),
        "num_shortest": len(shortest),
        "num_longest": len(longest),
        "num_discarded": len(discarded),
        "all_valid_path": str(output_dir / "all_valid.json"),
        "shortest_path": str(output_dir / "shortest.json"),
        "longest_path": str(output_dir / "longest.json"),
        "discarded_path": str(output_dir / "discarded.json"),
    }

    (output_dir / "all_valid.json").write_text(json.dumps(all_valid, ensure_ascii=False, indent=2))
    (output_dir / "shortest.json").write_text(json.dumps(shortest, ensure_ascii=False, indent=2))
    (output_dir / "longest.json").write_text(json.dumps(longest, ensure_ascii=False, indent=2))
    (output_dir / "discarded.json").write_text(json.dumps(discarded, ensure_ascii=False, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("num questions:", summary["num_questions"])
    print("num all valid:", summary["num_all_valid"])
    print("num shortest:", summary["num_shortest"])
    print("num longest:", summary["num_longest"])
    print("num discarded:", summary["num_discarded"])
    print("output dir:", output_dir)


if __name__ == "__main__":
    main()
