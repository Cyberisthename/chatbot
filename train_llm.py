#!/usr/bin/env python3
"""train_llm.py

Fine-tune a llama.cpp/Ollama-compatible base model using LoRA/QLoRA on your local
./training-data folder.

This script produces:
- LoRA adapter weights (for reuse)
- a merged Hugging Face model directory (ready for llama.cpp HF→GGUF conversion)

Next step:
  python3 convert_to_gguf.py --hf_model_dir ./releases/hf/jarvis-merged
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


JARVIS_SYSTEM_PROMPT = """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), an advanced AI assistant created by Ben.

You are professional, precise, and helpful. You can be witty, but never at the expense of clarity.
Prioritize user safety and privacy. When uncertain, say so and suggest next steps."""


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _prompt_choice(prompt: str, choices: Sequence[str], default_index: int = 0) -> str:
    if not _is_interactive():
        return choices[default_index]

    while True:
        print(prompt)
        for i, c in enumerate(choices, start=1):
            print(f"  {i}. {c}")
        raw = input(f"Select [1-{len(choices)}] (default {default_index + 1}): ").strip()
        if not raw:
            return choices[default_index]
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print("Invalid selection. Try again.\n")


def _detect_gpu_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available (CPU-only)"

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    return f"CUDA available: {props.name} ({vram_gb:.1f} GiB VRAM)"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_training_pairs_from_txt(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Parse 'User:' / 'Assistant:' pairs from a .txt file.

    Returns:
      (preface_text, pairs)
    """

    user_pos = text.find("User:")
    preface = text[:user_pos].strip() if user_pos != -1 else text.strip()
    convo = text[user_pos:] if user_pos != -1 else ""

    if not convo:
        return preface, []

    pattern = re.compile(r"User:\s*(.*?)\nAssistant:\s*(.*?)(?=\nUser:|\Z)", re.DOTALL)
    pairs = [(u.strip(), a.strip()) for (u, a) in pattern.findall(convo)]
    return preface, pairs


def load_local_training_data(data_dir: Path) -> List[Dict[str, Any]]:
    """Load examples from ./training-data.

    Supported formats:
    - .txt containing 'User:' / 'Assistant:' blocks
    - .jsonl containing either:
        {"messages": [{"role": "user"|"assistant"|"system", "content": "..."}, ...]}
      or
        {"instruction": "...", "output": "..."}
    """

    if not data_dir.exists():
        raise FileNotFoundError(f"Missing dataset folder: {data_dir}")

    examples: List[Dict[str, Any]] = []

    for p in sorted(data_dir.iterdir()):
        if p.suffix.lower() == ".txt":
            text = p.read_text(encoding="utf-8", errors="replace")
            preface, pairs = _read_training_pairs_from_txt(text)

            # If file is only preface, treat it as additional system context.
            if pairs:
                for user, assistant in pairs:
                    examples.append(
                        {
                            "system": (JARVIS_SYSTEM_PROMPT + ("\n\n" + preface if preface else "")).strip(),
                            "user": user,
                            "assistant": assistant,
                        }
                    )
            elif preface:
                # No explicit pairs; create a minimal single example to keep pipeline working.
                examples.append(
                    {
                        "system": JARVIS_SYSTEM_PROMPT,
                        "user": "Introduce yourself.",
                        "assistant": preface,
                    }
                )

        if p.suffix.lower() == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "messages" in obj and isinstance(obj["messages"], list):
                        examples.append({"messages": obj["messages"]})
                    elif "instruction" in obj and "output" in obj:
                        examples.append(
                            {
                                "system": JARVIS_SYSTEM_PROMPT,
                                "user": str(obj["instruction"]),
                                "assistant": str(obj["output"]),
                            }
                        )
                    else:
                        raise ValueError(
                            f"Unrecognized jsonl schema in {p}. Expected 'messages' or 'instruction'/'output'."
                        )

    if not examples:
        raise FileNotFoundError(f"No training files found in: {data_dir}")

    return examples


def _render_chat(
    tokenizer: Any, messages: List[Dict[str, str]], *, add_generation_prompt: bool
) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    # Fallback: a simple, model-agnostic format.
    out: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            out.append(f"### System\n{content}\n")
        elif role == "user":
            out.append(f"### User\n{content}\n")
        elif role == "assistant":
            out.append(f"### Assistant\n{content}\n")
        else:
            out.append(f"### {role.title()}\n{content}\n")

    if add_generation_prompt:
        out.append("### Assistant\n")
    return "\n".join(out).strip() + "\n"


def _example_to_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    if "messages" in example:
        return [{"role": m["role"], "content": m["content"]} for m in example["messages"]]

    return [
        {"role": "system", "content": example.get("system") or JARVIS_SYSTEM_PROMPT},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]


def _infer_lora_target_modules(model: torch.nn.Module) -> List[str]:
    """Heuristic: pick common attention/MLP projection names if present."""

    common = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "Wqkv",
        "wo",
        "wq",
        "wk",
        "wv",
    ]

    names = set()
    for name, _ in model.named_modules():
        names.add(name.split(".")[-1])

    selected = [n for n in common if n in names]

    # If we didn't match anything, fall back to a conservative set of the most frequent Linear leaf names.
    if selected:
        return selected

    linear_leaf_names: Dict[str, int] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            leaf = name.split(".")[-1]
            linear_leaf_names[leaf] = linear_leaf_names.get(leaf, 0) + 1

    return [n for (n, _) in sorted(linear_leaf_names.items(), key=lambda kv: kv[1], reverse=True)[:8]]


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [f["labels"] for f in features]
        to_pad = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        batch = self.tokenizer.pad(to_pad, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]

        padded_labels = torch.full(
            (len(labels), max_len), -100, dtype=torch.long
        )
        for i, lab in enumerate(labels):
            lab_t = torch.tensor(lab, dtype=torch.long)
            padded_labels[i, : lab_t.shape[0]] = lab_t

        batch["labels"] = padded_labels
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tune for Jarvis")
    parser.add_argument("--data_dir", default="./training-data")
    parser.add_argument("--base_model", default=None)

    parser.add_argument("--adapter_dir", default="./releases/hf/jarvis-lora")
    parser.add_argument("--merged_dir", default="./releases/hf/jarvis-merged")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=2048)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable Hugging Face trust_remote_code for models that require custom code.",
    )

    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Use 4-bit QLoRA (requires CUDA + bitsandbytes).",
    )

    args = parser.parse_args()

    print("🤖 Jarvis LoRA/QLoRA Fine-tuning")
    print(f"🔍 Hardware: {_detect_gpu_summary()}")

    base_model = args.base_model
    if base_model is None:
        base_model = _prompt_choice(
            "\nSelect a base model (must be llama.cpp-convertible):",
            [
                "Qwen/Qwen2.5-1.5B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "google/gemma-2-2b-it",
                "microsoft/Phi-3.5-mini-instruct",
            ],
            default_index=0,
        )

    data_dir = Path(args.data_dir)
    adapter_dir = Path(args.adapter_dir)
    merged_dir = Path(args.merged_dir)

    _ensure_dir(adapter_dir)
    _ensure_dir(merged_dir)

    print(f"\n📚 Loading local training data from: {data_dir}")
    raw_examples = load_local_training_data(data_dir)
    messages_examples = [_example_to_messages(ex) for ex in raw_examples]
    print(f"✅ Loaded {len(messages_examples)} examples")

    print(f"\n🧠 Loading tokenizer/model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    torch_dtype = None
    if use_cuda:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    quantization_config = None
    if args.qlora:
        if not use_cuda:
            raise RuntimeError("--qlora requested but CUDA is not available")
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "--qlora requested but BitsAndBytesConfig not available. Install bitsandbytes + recent transformers."
            ) from e

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch_dtype if use_cuda else None,
        quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if args.qlora:
            model = prepare_model_for_kbit_training(model)

        target_modules = _infer_lora_target_modules(model)
        print(f"🎯 LoRA target modules: {target_modules}")

        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)

        # Prints trainable parameter counts
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    except ImportError as e:
        raise RuntimeError(
            "peft is required for LoRA/QLoRA training. Please `pip install peft accelerate`."
        ) from e

    print("\n🔄 Tokenizing & building labels (loss only on assistant tokens)")

    ds = Dataset.from_list([{"messages": m} for m in messages_examples])

    def tokenize_row(row: Dict[str, Any]) -> Dict[str, Any]:
        messages = row["messages"]

        prompt_messages = messages[:-1]
        full_messages = messages

        prompt_text = _render_chat(tokenizer, prompt_messages, add_generation_prompt=True)
        full_text = _render_chat(tokenizer, full_messages, add_generation_prompt=False)

        full = tokenizer(
            full_text,
            truncation=True,
            max_length=args.max_seq_len,
            add_special_tokens=False,
        )
        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=args.max_seq_len,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = full["input_ids"]
        labels = list(input_ids)

        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    tokenized = ds.map(tokenize_row, remove_columns=ds.column_names)

    data_collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)

    optim = "adamw_torch"
    if args.qlora:
        # If bitsandbytes is installed, this is a common optimizer choice.
        try:
            import bitsandbytes  # noqa: F401

            optim = "paged_adamw_8bit"
        except Exception:
            optim = "adamw_torch"

    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        fp16=bool(use_cuda and torch_dtype == torch.float16),
        bf16=bool(use_cuda and torch_dtype == torch.bfloat16),
        optim=optim,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("\n🎯 Training...")
    trainer.train()

    print(f"\n💾 Saving LoRA adapter → {adapter_dir}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"\n🧩 Merging LoRA into base model → {merged_dir}")
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    meta = {
        "base_model": base_model,
        "method": "QLoRA" if args.qlora else "LoRA",
        "data_dir": str(data_dir),
        "num_examples": len(messages_examples),
        "max_seq_len": args.max_seq_len,
    }
    (merged_dir / "jarvis_training_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("\n✅ Done")
    print(f"Merged HF model: {merged_dir}")
    print(f"Next: python3 convert_to_gguf.py --hf_model_dir {merged_dir}")


if __name__ == "__main__":
    main()
