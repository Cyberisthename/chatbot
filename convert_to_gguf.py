#!/usr/bin/env python3
"""convert_to_gguf.py

Convert a merged Hugging Face CausalLM model directory (Llama/Mistral/Qwen2/Gemma/Phi)
into an Ollama-runnable GGUF using llama.cpp's *official* conversion script,
then quantize with llama.cpp's quantize binary.

Typical usage:
  # 1) Train LoRA/QLoRA and merge into HF format
  python3 train_llm.py

  # 2) Convert + quantize into ./releases/
  python3 convert_to_gguf.py --hf_model_dir ./releases/hf/jarvis-merged --quant Q4_K_M

  # 3) Import into Ollama
  ollama create jarvis -f Modelfile
  ollama run jarvis
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


JARVIS_SYSTEM_PROMPT = """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), an advanced AI assistant created by Ben.

Be professional, precise, and helpful. You may be witty, but never at the expense of clarity.
Prioritize user safety and privacy. When uncertain, say so and suggest next steps."""


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _run(cmd: list[str], *, cwd: Optional[Path] = None) -> None:
    cmd_str = " ".join([shlex_quote(c) for c in cmd])
    print(f"\n$ {cmd_str}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def shlex_quote(s: str) -> str:
    return subprocess.list2cmdline([s])


def _ensure_llama_cpp_repo(llama_cpp_dir: Path) -> None:
    if (llama_cpp_dir / "convert_hf_to_gguf.py").exists():
        return

    if llama_cpp_dir.exists() and any(llama_cpp_dir.iterdir()):
        raise RuntimeError(
            f"{llama_cpp_dir} exists but doesn't look like llama.cpp (missing convert_hf_to_gguf.py)."
        )

    print(f"📥 Cloning llama.cpp into {llama_cpp_dir} (local only)")
    llama_cpp_dir.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/ggerganov/llama.cpp.git",
        str(llama_cpp_dir),
    ])


def _find_quantize_binary(llama_cpp_dir: Path) -> Optional[Path]:
    candidates = [
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "quantize",
        llama_cpp_dir / "build" / "bin" / "llama_quantize",
        llama_cpp_dir / "llama-quantize",
        llama_cpp_dir / "quantize",
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    return None


def _build_llama_cpp(llama_cpp_dir: Path) -> Path:
    print("🛠️  Building llama.cpp (CPU build) ...")

    build_dir = llama_cpp_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake = shutil.which("cmake")
    if not cmake:
        raise RuntimeError("cmake not found. Please install cmake to build llama.cpp tools.")

    _run([cmake, "-S", str(llama_cpp_dir), "-B", str(build_dir)])
    _run([cmake, "--build", str(build_dir), "-j"])

    quant = _find_quantize_binary(llama_cpp_dir)
    if not quant:
        raise RuntimeError(
            "Built llama.cpp but could not find quantize binary. Looked in build/bin."
        )
    return quant


def _choose_quant(default: str = "Q4_K_M") -> str:
    choices = [
        "Q2_K",
        "Q3_K_S",
        "Q3_K_M",
        "Q4_0",
        "Q4_K_M",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
    ]

    if not _is_interactive():
        return default

    print("\nSelect GGUF quantization (depends on your RAM/VRAM + speed needs):")
    for i, c in enumerate(choices, start=1):
        suffix = " (default)" if c == default else ""
        print(f"  {i}. {c}{suffix}")

    raw = input(f"Quant (default {default}): ").strip()
    if not raw:
        return default

    if raw in choices:
        return raw

    if raw.isdigit() and 1 <= int(raw) <= len(choices):
        return choices[int(raw) - 1]

    print(f"Unrecognized selection '{raw}', using default: {default}")
    return default


def _convert_hf_to_gguf(
    *,
    llama_cpp_dir: Path,
    hf_model_dir: Path,
    out_f16: Path,
) -> None:
    script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing llama.cpp script: {script}")

    out_f16.parent.mkdir(parents=True, exist_ok=True)

    # llama.cpp script flags changed slightly across versions. We probe help output.
    help_out = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout

    if "--outfile" in help_out:
        cmd = [
            sys.executable,
            str(script),
            str(hf_model_dir),
            "--outtype",
            "f16",
            "--outfile",
            str(out_f16),
        ]
    elif "--output" in help_out:
        cmd = [
            sys.executable,
            str(script),
            str(hf_model_dir),
            "--outtype",
            "f16",
            "--output",
            str(out_f16),
        ]
    else:
        # Best-effort fallback (older versions used positional output).
        cmd = [
            sys.executable,
            str(script),
            str(hf_model_dir),
            "--outtype",
            "f16",
        ]

    _run(cmd, cwd=llama_cpp_dir)

    if not out_f16.exists():
        raise RuntimeError(
            f"HF→GGUF conversion finished, but output file not found: {out_f16}"
        )


def _quantize(
    *,
    quant_bin: Path,
    in_gguf: Path,
    out_gguf: Path,
    quant: str,
    threads: int,
) -> None:
    out_gguf.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(quant_bin),
        str(in_gguf),
        str(out_gguf),
        quant,
        str(threads),
    ]

    # Some versions do not accept threads arg; retry without it.
    try:
        _run(cmd)
    except subprocess.CalledProcessError:
        _run([str(quant_bin), str(in_gguf), str(out_gguf), quant])

    if not out_gguf.exists():
        raise RuntimeError(f"Quantization finished, but output file not found: {out_gguf}")


def _write_modelfile(project_root: Path, gguf_path: Path) -> None:
    modelfile = project_root / "Modelfile"
    rel = os.path.relpath(gguf_path, start=project_root)

    content = "\n".join(
        [
            f"FROM ./{rel}",
            "",
            f"SYSTEM \"\"\"{JARVIS_SYSTEM_PROMPT}\"\"\"",
            "",
            "PARAMETER temperature 0.7",
            "PARAMETER top_p 0.9",
            "PARAMETER num_ctx 4096",
            "",
        ]
    )

    modelfile.write_text(content, encoding="utf-8")
    print(f"📝 Wrote {modelfile} → FROM ./{rel}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert merged HF model → GGUF (llama.cpp) → quantize")
    parser.add_argument("--hf_model_dir", default="./releases/hf/jarvis-merged")

    parser.add_argument("--releases_dir", default="./releases")
    parser.add_argument("--model_name", default="jarvis")

    parser.add_argument(
        "--llama_cpp_dir",
        default="./.llama.cpp",
        help="Local clone of llama.cpp (will be cloned if missing).",
    )

    parser.add_argument(
        "--quant",
        default=None,
        help="Quantization type (e.g. Q4_K_M). If omitted, you'll be prompted.",
    )
    parser.add_argument("--threads", type=int, default=max(os.cpu_count() or 4, 4))

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    hf_model_dir = (project_root / args.hf_model_dir).resolve()
    releases_dir = (project_root / args.releases_dir).resolve()
    llama_cpp_dir = (project_root / args.llama_cpp_dir).resolve()

    if not hf_model_dir.exists():
        raise FileNotFoundError(
            f"HF model dir not found: {hf_model_dir}\nDid you run: python3 train_llm.py ?"
        )

    quant = args.quant or _choose_quant()

    out_f16 = releases_dir / f"{args.model_name}.f16.gguf"
    out_quant = releases_dir / f"{args.model_name}.{quant}.gguf"

    _ensure_llama_cpp_repo(llama_cpp_dir)

    quant_bin = _find_quantize_binary(llama_cpp_dir)
    if not quant_bin:
        quant_bin = _build_llama_cpp(llama_cpp_dir)

    print(f"\n🔄 Converting HF → GGUF (f16): {out_f16}")
    _convert_hf_to_gguf(llama_cpp_dir=llama_cpp_dir, hf_model_dir=hf_model_dir, out_f16=out_f16)

    print(f"\n🧮 Quantizing → {out_quant} ({quant})")
    _quantize(
        quant_bin=quant_bin,
        in_gguf=out_f16,
        out_gguf=out_quant,
        quant=quant,
        threads=args.threads,
    )

    _write_modelfile(project_root, out_quant)

    print("\n✅ GGUF ready for Ollama")
    print(f"Final GGUF: {out_quant}")
    print("Next:")
    print("  ollama create jarvis -f Modelfile")
    print("  ollama run jarvis")


if __name__ == "__main__":
    main()
