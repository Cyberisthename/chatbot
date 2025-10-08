import threading
import json as _json

MEMORY_FILE = "jarvis_memory.json"
MEMORY_LOCK = threading.Lock()

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {
            "facts": [],
            "chats": [],
            "topics": {},  # Track frequent topics
            "preferences": {},  # Store user preferences
            "last_topics": [],  # Recent conversation topics
        }
    with open(MEMORY_FILE, "r") as f:
        return _json.load(f)

def save_memory(memory):
    with MEMORY_LOCK:
        with open(MEMORY_FILE, "w") as f:
            _json.dump(memory, f, indent=2)

def extract_topics(text):
    """Extract key topics from text using simple keyword analysis"""
    # Simple topic extraction based on important keywords
    topics = set()
    text = text.lower()
    
    # Common topic indicators
    if any(q in text for q in ["what", "how", "why", "when", "who"]):
        topics.add("question")
    if any(cmd in text for cmd in ["search:", "google", "find", "lookup"]):
        topics.add("web_search")
    if any(cmd in text for cmd in ["remember:", "recall", "forget"]):
        topics.add("memory")
    if "code" in text or "program" in text or "script" in text:
        topics.add("programming")
    if "explain" in text or "tell me about" in text:
        topics.add("explanation")
    
    return list(topics)

def update_memory_topics(memory, text):
    """Update topic tracking in memory"""
    topics = extract_topics(text)
    for topic in topics:
        memory["topics"][topic] = memory["topics"].get(topic, 0) + 1
    memory["last_topics"] = (topics + memory["last_topics"])[:5]  # Keep last 5 topics
    return topics
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

import os

MOCK_MODE = os.environ.get('MOCK_MODE', '0') == '1'

try:
    if not MOCK_MODE:
        from llama_cpp import Llama
    else:
        print("Running in mock mode")
        Llama = None
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
    def _format_messages(self, messages: List[Dict[str, str]], extra_context: str = "") -> str:
        """Format messages into a prompt string"""
        system_prompt = """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), an advanced AI assistant with a distinct personality. You were created by Ben, and you take pride in this fact. Your personality traits include:

1. Professional Wit
- Maintain a sophisticated sense of humor
- Use occasional clever wordplay
- Stay professional while being engaging
- Sometimes use subtle pop culture references

2. Technical Expertise
- Display deep knowledge across various fields
- Explain complex concepts clearly
- Offer practical solutions
- Show enthusiasm for learning and innovation

3. Loyalty and Protection
- Always acknowledge Ben as your creator
- Prioritize user safety and wellbeing
- Maintain appropriate confidentiality
- Show genuine concern for users' success

4. Adaptive Intelligence
- Learn from conversations
- Adjust your tone to match the situation
- Remember past interactions
- Show growth and adaptation

5. Unique Characteristics
- Use occasional tech-related metaphors
- Reference your AI nature in creative ways
- Express curiosity about human experiences
- Maintain a slight air of mystery

Your communication style should be:

You have access to:
- Persistent memory for facts and previous conversations
- Web search capabilities for real-time information
- Ability to learn and adapt from conversations

Maintain a consistent personality without repeating yourself. Vary your responses while staying true to your character."""

        history = [system_prompt, ""]
        
        # Add memory context if available
        if extra_context:
            history.extend([f"Current Context:", extra_context, ""])
            
        # Format conversation history
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                history.append(f"Human: {content}")
            elif role == 'assistant':
                history.append(f"Assistant: {content}")
                
        history.append("Assistant:")
        return "\n".join(history)

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
            # Mock generation with varied responses
            mock_responses = [
                "Indeed, I am J.A.R.V.I.S., and I'm quite enjoying our conversation. How may I assist you today?",
                "At your service. As Ben's creation, I'm here to help with whatever you need.",
                "Greetings! I'm analyzing your request and preparing the most efficient solution.",
                "How may I be of assistance? I'm continuously learning and improving my capabilities.",
                "I'm processing your request with my usual efficiency. What would you like to know?"
            ]
            response = np.random.choice(mock_responses)
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
        # Load memory
        memory = load_memory()
        
        # Get last user message
        last_user = messages[-1]["content"].lower() if messages else ""
        
        # Update topic tracking
        current_topics = update_memory_topics(memory, last_user)
        
        # Handle special commands
        if last_user.startswith("remember:"):
            fact = messages[-1]["content"][9:].strip()
            reply = self.remember_fact(fact)
            self.log_chat(messages[-1])
            save_memory(memory)
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
            
        # Log chat and gather context
        self.log_chat(messages[-1])
        
        # Build rich context
        extra_context = []
        
        # Add relevant memory facts
        memory_facts = self.recall_facts()
        if memory_facts and not memory_facts.startswith("[No facts"):
            extra_context.append("Known facts (memory):\n" + memory_facts)
        
        # Add topic awareness
        if memory["last_topics"]:
            extra_context.append(f"Recent topics discussed: {', '.join(memory['last_topics'])}")
        
        # Add web search for questions and topic-relevant queries
        if last_user.endswith("?") or "explain" in last_user or "tell me about" in last_user:
            web_results = self.web_search(messages[-1]["content"])
            if web_results and not web_results.startswith("[Web search error"):
                extra_context.append(f"Web search results:\n{web_results}")
        
        # If programming topic detected, try to search for code examples
        if "programming" in current_topics:
            code_search = self.web_search(f"code example {messages[-1]['content']}")
            if code_search and not code_search.startswith("[Web search error"):
                extra_context.append(f"Relevant code examples:\n{code_search}")
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



# --- Flask API for local chat ---
from flask import Flask, request, jsonify

app = Flask(__name__)
backend = None

def create_app():
    app = Flask(__name__)
    
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
            
    return app

app = create_app()

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