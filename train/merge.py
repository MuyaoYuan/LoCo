import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def find_tensor(root, key):
    for p in sorted(Path(root).glob("*.safetensors")):
        with safe_open(p, framework="pt") as f:
            if key in f.keys():
                return f.get_tensor(key)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--torch-dtype", default="bfloat16")
    args = parser.parse_args()

    dtype = getattr(torch, args.torch_dtype)
    device_map = {"": 0} if torch.cuda.is_available() else "cpu"
    print(f"device: {device_map}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map=device_map)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(output)
    for name in ("generation_config.json", "chat_template.jinja"):
        src = Path(args.base_model) / name
        if src.exists():
            shutil.copy(src, output / name)

    key = "model.layers.3.self_attn.q_proj.weight"
    w_base = find_tensor(args.base_model, key)
    w_merged = find_tensor(output, key)
    diff = (w_merged.float() - w_base.float()).abs()
    changed = (diff > 0).float().mean().item() * 100
    print(f"{key}: changed {changed:.1f}% elements, max|delta|={diff.max().item():.6f}")
    assert diff.max().item() > 0

    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    if "dtype" in config:
        config.setdefault("torch_dtype", config["dtype"])
    if "torch_dtype" in config:
        config.setdefault("dtype", config["torch_dtype"])
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))

    print(f"merged {args.adapter} into {args.base_model} -> {output}")


if __name__ == "__main__":
    main()
