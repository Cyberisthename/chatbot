"""J.A.R.V.I.S. Ollama Integration"""
import requests
import json
from typing import Optional

class OllamaJarvis:
    def __init__(self, model="jarvis", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/generate"
    
    def chat(self, prompt: str, stream: bool = False) -> str:
        try:
            response = requests.post(
                self.endpoint,
                json={"model": self.model, "prompt": prompt, "stream": stream},
                timeout=300
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

if __name__ == "__main__":
    jarvis = OllamaJarvis()
    
    print("🤖 J.A.R.V.I.S. - Ollama Chat")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        try:
            response = jarvis.chat(user_input)
            print(f"J.A.R.V.I.S.: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
