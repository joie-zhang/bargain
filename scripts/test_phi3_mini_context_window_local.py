#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import torch
from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "/scratch/gpfs/DANQIC/models/Phi-3-mini-128k-instruct"
DEFAULT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 98304, 122880, 126976, 130048, 131071]


def load_local_phi3_modules(model_path: Path):
    package_name = "phi3mini_local_probe"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_path)]
    sys.modules[package_name] = package

    modules = {}
    for module_name in ["configuration_phi3", "modeling_phi3"]:
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{module_name}",
            model_path / f"{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        modules[module_name] = module

    return modules["configuration_phi3"], modules["modeling_phi3"]


def patch_modeling_for_sdpa(modeling_module):
    modeling_module.Phi3PreTrainedModel._supports_sdpa = True
    original_prepare = modeling_module._prepare_4d_causal_attention_mask

    def patched_prepare(attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window=None):
        # For these tests we use single, unpadded sequences, so SDPA can use
        # its internal causal masking without materializing a giant 4D mask.
        if attention_mask is None:
            return None
        return original_prepare(
            attention_mask,
            input_shape,
            inputs_embeds,
            past_key_values_length,
            sliding_window,
        )

    modeling_module._prepare_4d_causal_attention_mask = patched_prepare


def load_model_and_tokenizer(model_path: Path):
    config_module, modeling_module = load_local_phi3_modules(model_path)
    patch_modeling_for_sdpa(modeling_module)

    config = config_module.Phi3Config.from_pretrained(str(model_path))
    config._attn_implementation = "sdpa"

    model = modeling_module.Phi3ForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    return model, tokenizer


def build_prompt_for_target_tokens(tokenizer, target_tokens: int) -> tuple[str, int]:
    prefix = "<|system|>\nYou are a precise assistant.<|end|>\n<|user|>\n"
    suffix = (
        "\nThis is a long context smoke test. Ignore the filler above and reply with exactly OK."
        "<|end|>\n<|assistant|>\n"
    )

    def make_prompt(repeats: int) -> str:
        return prefix + ("token " * repeats) + suffix

    low, high = 0, max(target_tokens, 1)
    best_prompt = make_prompt(0)
    best_length = len(tokenizer(best_prompt, add_special_tokens=False).input_ids)

    while low <= high:
        mid = (low + high) // 2
        prompt = make_prompt(mid)
        length = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if length <= target_tokens:
            best_prompt = prompt
            best_length = length
            low = mid + 1
        else:
            high = mid - 1

    return best_prompt, best_length


def probe_length(model, tokenizer, target_tokens: int) -> dict:
    prompt, prompt_tokens = build_prompt_for_target_tokens(tokenizer, target_tokens)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to("cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    started = time.time()
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False)
    elapsed = time.time() - started

    next_token_id = int(outputs.logits[0, -1].argmax().item())
    next_token_text = tokenizer.decode([next_token_id])

    return {
        "target_tokens": target_tokens,
        "prompt_tokens": int(prompt_tokens),
        "predicted_next_token_id": next_token_id,
        "predicted_next_token_piece": tokenizer.convert_ids_to_tokens(next_token_id),
        "predicted_next_token": next_token_text,
        "elapsed_s": round(elapsed, 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress-test Phi-3-mini local context length.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--length",
        action="append",
        dest="lengths",
        type=int,
        help="Prompt token lengths to test. Repeat the flag to supply multiple lengths.",
    )
    parser.add_argument("--json", action="store_true", help="Print results as JSON.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    model_path = Path(args.model_path)
    model, tokenizer = load_model_and_tokenizer(model_path)

    results = []
    for length in (args.lengths or DEFAULT_LENGTHS):
        try:
            result = probe_length(model, tokenizer, length)
            result["ok"] = True
        except Exception as exc:
            result = {
                "target_tokens": length,
                "ok": False,
                "error": str(exc),
            }
        results.append(result)

        if args.json:
            continue

        print(json.dumps(result, ensure_ascii=True))

    if args.json:
        print(json.dumps(results, ensure_ascii=True, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
