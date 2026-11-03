from __future__ import annotations

import argparse
import dataclasses as _dc
import os
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, PeftModel, get_peft_model
from trl import DPOConfig, DPOTrainer



def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_deepspeed_config(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    stage = int(os.environ.get("ZERO_STAGE", "3"))
    zero: Dict[str, Any] = {
        "stage": stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
    }
    if stage == 3:
        zero.update({
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1_000_000_000,
            "stage3_max_reuse_distance": 1_000_000_000,
            "stage3_gather_16bit_weights_on_model_save": True,
        })

    hf_sched = ((config or {}).get("training") or {}).get("lr_scheduler_type", "cosine")
    if hf_sched in ("constant", "constant_with_warmup"):
        scheduler = {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        }
    else:
        scheduler = {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "total_num_steps": "auto",
            },
        }

    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": "auto"},
        "zero_optimization": zero,
        "scheduler": scheduler,
    }



def _resolve_attn_impl(name: str) -> str:
    aliases = {
        "fa2": "flash_attention_2",
        "flash2": "flash_attention_2",
        "flash_attn_2": "flash_attention_2",
        "flash-attention-2": "flash_attention_2",
        "fa3": "flash_attention_3",
        "flash3": "flash_attention_3",
        "flash_attn_3": "flash_attention_3",
        "flash-attention-3": "flash_attention_3",
    }
    return aliases.get(name, name)


def prepare_model(config: Dict[str, Any], init_adapter: str | None = None):
    torch_dtype = getattr(torch, config["model"]["torch_dtype"])
    attn_impl = _resolve_attn_impl(
        config["model"].get("attn_implementation", "flash_attention_2")
    )
    model_kwargs: Dict[str, Any] = dict(
        trust_remote_code=config["model"]["trust_remote_code"],
        use_cache=False,
        attn_implementation=attn_impl,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"], dtype=torch_dtype, **model_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"], torch_dtype=torch_dtype, **model_kwargs,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = config["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        use_rslora=lora_cfg.get("use_rslora", False),
    )
    if init_adapter:
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        print(f"[init_adapter] loaded from {init_adapter}")
    else:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer



def _resolve_jsonl_files(dataset_path: str) -> list[str]:
    path = Path(dataset_path)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        files = sorted(str(p) for p in path.glob("*.jsonl"))
        if files:
            return files
    raise FileNotFoundError(f"No JSONL files found at: {dataset_path}")


def load_training_dataset(config: Dict[str, Any]):
    dataset_cfg = config["dataset"]
    source_type = dataset_cfg.get("source_type", "local_jsonl")
    split = dataset_cfg.get("split", "train")
    cache_dir = dataset_cfg.get("cache_dir")

    if source_type == "local_jsonl":
        files = {split: _resolve_jsonl_files(dataset_cfg["path"])}
        dataset = load_dataset("json", data_files=files, split=split, cache_dir=cache_dir)
    elif source_type == "huggingface":
        dataset = load_dataset(dataset_cfg["path"], split=split, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unsupported dataset.source_type: {source_type}")

    if dataset_cfg.get("shuffle_before_select", False):
        dataset = dataset.shuffle(seed=dataset_cfg.get("seed", 42))

    max_samples = dataset_cfg.get("max_samples")
    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        if dataset_cfg.get("longest_first", False):
            ps = dataset_cfg.get("default_prompt_suffix", "") or ""
            lengths = [
                len((r.get("question") or "") + ps
                    + (r.get("prefix_cot") or "")
                    + max(r.get("chosen") or "", r.get("rejected") or "", key=len))
                for r in dataset
            ]
            top_idx = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)[:max_samples]
            top_idx.sort()
            dataset = dataset.select(top_idx)
        else:
            dataset = dataset.select(range(max_samples))

    return dataset


def _build_formatter(tokenizer, dataset_cfg: Dict[str, Any]):
    question_col = dataset_cfg.get("question_column", "question")
    prompt_suffix_col = dataset_cfg.get("prompt_suffix_column", "prompt_suffix")
    prefix_cot_col = dataset_cfg.get("prefix_cot_column", "prefix_cot")
    chosen_col = dataset_cfg.get("chosen_column", "chosen")
    rejected_col = dataset_cfg.get("rejected_column", "rejected")
    fallback_prompt_suffix = dataset_cfg.get("default_prompt_suffix", "")
    eos_token = tokenizer.eos_token or ""

    def format_example(example):
        prompts: list[str] = []
        chosens: list[str] = []
        rejecteds: list[str] = []
        questions = example[question_col]
        prefix_cots = example[prefix_cot_col]
        chosens_in = example[chosen_col]
        rejecteds_in = example[rejected_col]
        prompt_suffixes = example.get(
            prompt_suffix_col, [fallback_prompt_suffix] * len(questions)
        )
        for question, ps, prefix_cot, chosen, rejected in zip(
            questions, prompt_suffixes, prefix_cots, chosens_in, rejecteds_in
        ):
            ps = ps if ps is not None else fallback_prompt_suffix
            messages = [{"role": "user", "content": question + ps}]
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            prompt_text = chat_prompt + (prefix_cot or "")
            prompts.append(prompt_text)
            chosens.append((chosen or "") + eos_token)
            rejecteds.append((rejected or "") + eos_token)
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    return format_example



class EpochSaveCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True
        return control






def _build_dpo_config(config: Dict[str, Any]) -> DPOConfig:
    dpo_cfg = config.get("dpo", {})
    supported = {f.name for f in _dc.fields(DPOConfig)}

    kwargs: Dict[str, Any] = dict(config["training"])
    kwargs["deepspeed"] = get_deepspeed_config(config)
    kwargs["ddp_find_unused_parameters"] = False
    kwargs["max_length"] = config.get("max_length", 12000)
    if "max_prompt_length" in supported:
        kwargs["max_prompt_length"] = config.get("max_prompt_length", 4096)
    if "max_completion_length" in supported:
        kwargs["max_completion_length"] = config.get("max_completion_length", 7168)

    loss_type = dpo_cfg.get("loss_type", "sigmoid")
    loss_field = next((f for f in _dc.fields(DPOConfig) if f.name == "loss_type"), None)
    if loss_field is not None:
        want_list = "list" in str(loss_field.type).lower()
        if want_list and isinstance(loss_type, str):
            loss_type = [loss_type]
        elif (not want_list) and isinstance(loss_type, (list, tuple)):
            loss_type = loss_type[0] if loss_type else "sigmoid"
    kwargs["loss_type"] = loss_type

    kwargs["beta"] = dpo_cfg.get("beta", 0.1)
    if "label_smoothing" in supported:
        kwargs["label_smoothing"] = dpo_cfg.get("label_smoothing", 0.0)
    if "precompute_ref_log_probs" in supported:
        kwargs["precompute_ref_log_probs"] = dpo_cfg.get("precompute_ref_log_probs", False)

    for legacy in ("reference_free", "rpo_alpha", "sync_ref_model"):
        if legacy in dpo_cfg and legacy in supported:
            kwargs[legacy] = dpo_cfg[legacy]

    return DPOConfig(**kwargs)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="train/configs/lora_dpo_r1_7b.yaml",
        help="Path to the YAML training config.",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in output_dir (auto-detected by Trainer).",
    )
    parser.add_argument(
        "--init_adapter",
        default=None,
        help="Path to a pretrained LoRA adapter checkpoint to initialize from "
             "(for SFT→DPO two-stage: load a finished SFT adapter instead of a "
             "fresh random one). LoRA config in YAML must match the SFT run. "
             "Incompatible with --resume.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.init_adapter and args.resume:
        raise SystemExit("--init_adapter and --resume are mutually exclusive: "
                         "resume continues its own adapter in output_dir, "
                         "init_adapter starts from an external SFT adapter.")
    model, tokenizer = prepare_model(config, init_adapter=args.init_adapter)

    dataset = load_training_dataset(config)
    format_example = _build_formatter(tokenizer, config["dataset"])
    drop_columns = [c for c in dataset.column_names
                    if c not in {"prompt", "chosen", "rejected"}]
    dataset = dataset.map(format_example, batched=True, remove_columns=drop_columns)

    training_args = _build_dpo_config(config)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.add_callback(EpochSaveCallback())

    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model()


if __name__ == "__main__":
    main()
