import threading
import json as _json

MEMORY_FILE = "jarvis_memory.json"
MEMORY_LOCK = threading.Lock()

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {"facts": [], "chats": []}
    with open(MEMORY_FILE, "r") as f:
        return _json.load(f)

def save_memory(memory):
    with MEMORY_LOCK:
        with open(MEMORY_FILE, "w") as f:
            _json.dump(memory, f, indent=2)
from duckduckgo_search import DDGS
#!/usr/bin/env python3
"""
J.A.R.V.I.S. Python Inference Backend
Supports GGUF model loading and inference
"""

import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional

try:
    from llama_cpp import Llama
except ImportError:
    print("Warning: llama-cpp-python not installed. Using mock backend.")
    Llama = None

class JarvisInferenceBackend:
    def remember_fact(self, fact):
        memory = load_memory()
        memory["facts"].append(fact)
        save_memory(memory)
        return "[Fact remembered!]"

    def recall_facts(self):
        memory = load_memory()
        if not memory["facts"]:
            return "[No facts remembered yet.]"
        return "\n".join(f"- {fact}" for fact in memory["facts"])

    def log_chat(self, message):
        memory = load_memory()
        memory["chats"].append(message)
        save_memory(memory)
    def web_search(self, query, max_results=3):
        """Perform a web search and return top results as a string."""
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                snippets = [r["body"] for r in results if "body" in r]
            if not snippets:
                return "[No relevant web results found.]"
            return "\n".join(snippets)
        except Exception as e:
            return f"[Web search error: {e}]"
    def __init__(self, model_path: str, config: Dict[str, Any] = None):
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize the model"""
        try:
            if Llama is None:
                print("🤖 Using mock inference backend")
                self.is_initialized = True
                return True
                
            print(f"🚀 Loading model: {self.model_path}")
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.config.get('context_size', 2048),
                n_gpu_layers=self.config.get('gpu_layers', 0),
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                verbose=False
            )
            
            self.is_initialized = True
            print("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        """Generate text from prompt"""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")
            
        start_time = time.time()
        
        if Llama is None:
            # Mock generation
            response = "I am J.A.R.V.I.S., your advanced AI assistant. I'm here to help you with any questions or tasks you may have."
            tokens_used = len(response.split())
        else:
            # Real generation
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                stop=["</s>", "Human:", "User:"],
                echo=False
            )
            
            response = output['choices'][0]['text'].strip()
            tokens_used = output['usage']['total_tokens']
        
        generation_time = time.time() - start_time
        
        return {
            'text': response,
            'tokens_used': tokens_used,
            'generation_time': generation_time,
            'tokens_per_second': tokens_used / generation_time if generation_time > 0 else 0
        }
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Chat with the model"""
        # Check for memory or web search commands
        last_user = messages[-1]["content"].lower() if messages else ""
        if last_user.startswith("remember:"):
            fact = messages[-1]["content"][9:].strip()
            reply = self.remember_fact(fact)
            self.log_chat(messages[-1])
            return {'message': {'role': 'assistant', 'content': reply}, 'usage': {}, 'performance': {}, 'timestamp': time.time()}
        if last_user.startswith("recall facts"):
            reply = self.recall_facts()
            self.log_chat(messages[-1])
            return {'message': {'role': 'assistant', 'content': reply}, 'usage': {}, 'performance': {}, 'timestamp': time.time()}
        if last_user.startswith("search:") or last_user.startswith("google ") or last_user.startswith("websearch:"):
            query = last_user.split(":",1)[-1] if ":" in last_user else last_user
            web_results = self.web_search(query.strip())
            reply = f"[Web search results for '{query.strip()}']\n{web_results}"
            self.log_chat(messages[-1])
            return {'message': {'role': 'assistant', 'content': reply}, 'usage': {}, 'performance': {}, 'timestamp': time.time()}
        # Log chat
        self.log_chat(messages[-1])
        # Gather extra context: memory facts and (optionally) web search
        memory_facts = self.recall_facts()
        extra_context = ""
        if memory_facts and not memory_facts.startswith("[No facts"):
            extra_context += "Known facts (memory):\n" + memory_facts + "\n\n"
        # Optionally, if last user message is a question, try web search too
        last_user = messages[-1]["content"].lower() if messages else ""
        if last_user.endswith("?"):
            web_results = self.web_search(messages[-1]["content"])
            if web_results and not web_results.startswith("[Web search error"):
                extra_context += f"Web search results:\n{web_results}\n\n"
        # Format messages with extra context
        prompt = self._format_messages(messages, extra_context=extra_context)
        # Generate response
        result = self.generate(prompt, **kwargs)
        return {
            'message': {
                'role': 'assistant',
                'content': result['text']
            },
            'usage': {
                'prompt_tokens': len(prompt.split()),
                'completion_tokens': result['tokens_used'],
                'total_tokens': len(prompt.split()) + result['tokens_used']
            },
            'performance': {
                'generation_time': result['generation_time'],
                'tokens_per_second': result['tokens_per_second']
            },
            'timestamp': time.time()
        }

@app.route("/chat", methods=["POST"])
def chat_api():
    global backend
    data = request.get_json()
    messages = data.get("messages", [])
    if not backend or not backend.is_initialized:
        return jsonify({"error": "Model not initialized"}), 503
    try:
        result = backend.chat(messages)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Flask API for local chat ---
from flask import Flask, request, jsonify

app = Flask(__name__)
backend = None

@app.route("/chat", methods=["POST"])
def chat_api():
    global backend
    data = request.get_json()
    messages = data.get("messages", [])
    if not backend or not backend.is_initialized:
        return jsonify({"error": "Model not initialized"}), 503
    try:
        result = backend.chat(messages)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    # Ensure J.A.R.V.I.S. knows Ben is his creator
    memory = load_memory()
    creator_fact = "Ben is my creator."
    if creator_fact not in memory["facts"]:
        memory["facts"].append(creator_fact)
        save_memory(memory)
    import argparse
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. Inference Backend")
    parser.add_argument("model_path", help="Path to GGUF model file")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server (default: 8000)")
    args = parser.parse_args()

    global backend
    backend = JarvisInferenceBackend(args.model_path)
    if not backend.initialize():
        print("Failed to initialize model")
        sys.exit(1)

    # Start Flask API server
    print(f"\n🌐 Starting J.A.R.V.I.S. API server on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()