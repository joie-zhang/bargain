#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


DEFAULT_MODEL_PATH = "/scratch/gpfs/DANQIC/models/Phi-3-mini-128k-instruct"
QUESTIONS = [
    "What is 12 plus 35?",
    "What is the color of the sky?",
    "What is your name? What is the name of the model that you are. Are you a reasoning model? What is your context length? What is your name?",
]


# Phi-3's released remote code expects an older transformers cache API.
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
if not hasattr(DynamicCache, "get_max_length"):
    def _get_max_length(self):
        value = self.get_max_cache_shape()
        return None if value is None or value < 0 else value
    DynamicCache.get_max_length = _get_max_length
if not hasattr(DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length, layer_idx=0):
        del new_seq_length
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = _get_usable_length


def load_generator(model_path: str):
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_dir}")

    torch.random.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )


def ask(generator, question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Answer briefly and directly."},
        {"role": "user", "content": question},
    ]
    output = generator(
        messages,
        max_new_tokens=128,
        return_full_text=False,
        temperature=0.0,
        do_sample=False,
    )
    return output[0]["generated_text"].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local Phi-3-mini-128k-instruct smoke test.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--json", action="store_true", help="Print results as JSON.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; local Phi-3 inference needs a GPU.")

    generator = load_generator(args.model_path)
    results = [{"question": question, "answer": ask(generator, question)} for question in QUESTIONS]

    if args.json:
        print(json.dumps(results, ensure_ascii=True, indent=2))
        return 0

    for idx, item in enumerate(results, start=1):
        print(f"Q{idx}: {item['question']}")
        print(f"A{idx}: {item['answer']}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
