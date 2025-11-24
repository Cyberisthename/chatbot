from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
JARVIS_LAB_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "llama3.1"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class OllamaError(RuntimeError):
    pass


def call_ollama(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    if response.status_code != 200:
        raise OllamaError(f"Ollama returned status {response.status_code}: {response.text}")

    data = response.json()
    message = data.get("message")
    if not message or "content" not in message:
        raise OllamaError(f"Unexpected Ollama response payload: {data}")

    return message["content"].strip()


def call_lab_run_phase(params: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{JARVIS_LAB_URL}/run_phase_experiment", json=params, timeout=120)
    response.raise_for_status()
    return response.json()


def call_lab_tri(params: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{JARVIS_LAB_URL}/tri", json=params, timeout=120)
    response.raise_for_status()
    return response.json()


def call_lab_discovery(params: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{JARVIS_LAB_URL}/discovery", json=params, timeout=120)
    response.raise_for_status()
    return response.json()


def call_lab_replay_drift(params: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{JARVIS_LAB_URL}/replay_drift", json=params, timeout=120)
    response.raise_for_status()
    return response.json()


def ensure_lab_server_ready(retries: int = 5, delay: float = 1.5) -> None:
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(f"{JARVIS_LAB_URL}/health", timeout=5)
            if response.status_code == 200:
                return
        except Exception:  # noqa: BLE001 - best effort health check
            time.sleep(delay)
    raise RuntimeError("Jarvis Lab API is not reachable. Start jarvis_api.py first.")


def main() -> None:
    ensure_lab_server_ready()

    print("🤖 Connected to Jarvis Lab API at", JARVIS_LAB_URL)
    print("🧠 Using Ollama model:", DEFAULT_MODEL)
    print("Type 'exit' or 'quit' to stop.\n")

    system_prompt = (
        "You are Ben's Lab AI. Collaborate with the user to run Jarvis experiments.\n"
        "If you need to execute a lab experiment, respond with a single line formatted exactly as:\n"
        "TOOL: {\"name\": \"tool_name\", \"args\": {...}}\n"
        "Available tools: run_phase, tri, discovery, replay_drift.\n"
        "Only use TOOL responses when execution is required. Otherwise, reply normally."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    while True:
        try:
            user_input = input("Ben: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            reply = call_ollama(messages)
        except OllamaError as exc:
            print(f"[ERROR] Failed to reach Ollama: {exc}")
            messages.pop()  # remove last user message if Ollama fails
            continue

        print("AI:", reply)

        if reply.startswith("TOOL:"):
            try:
                tool_json = reply[len("TOOL:") :].strip()
                tool = json.loads(tool_json)
                name = tool.get("name")
                args = tool.get("args", {})
            except (json.JSONDecodeError, AttributeError) as exc:
                print(f"[ERROR] Invalid tool payload: {exc}")
                messages.append({"role": "assistant", "content": reply})
                continue

            lab_result: Dict[str, Any]
            try:
                if name == "run_phase":
                    lab_result = call_lab_run_phase(args)
                elif name == "tri":
                    lab_result = call_lab_tri(args)
                elif name == "discovery":
                    lab_result = call_lab_discovery(args)
                elif name == "replay_drift":
                    lab_result = call_lab_replay_drift(args)
                else:
                    print(f"[ERROR] Unknown tool name: {name}")
                    messages.append({"role": "assistant", "content": reply})
                    continue
            except requests.RequestException as exc:
                print(f"[ERROR] Lab request failed: {exc}")
                messages.append({"role": "assistant", "content": reply})
                continue

            lab_summary = json.dumps(lab_result)
            messages.append({
                "role": "user",
                "content": f"Experiment result: {lab_summary}",
            })

            try:
                follow_up = call_ollama(messages)
            except OllamaError as exc:
                print(f"[ERROR] Ollama follow-up failed: {exc}")
                continue

            print("AI (after lab):", follow_up)
            messages.append({"role": "assistant", "content": follow_up})
        else:
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
