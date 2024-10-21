from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from pydantic import Field
import requests

class OllamaLLM(LLM):
    model: str = Field(default="llama3.1")
    base_url: str = Field(default='http://localhost:11434')

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = f"{self.base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "prompt": prompt
        }

        response = requests.post('http://localhost:11434/api/generate', headers=headers, json=payload, stream=True)

        if response.status_code != 200:
            raise Exception(f"Ollama API returned status code {response.status_code}")

        result = ""
        for line in response.iter_lines():
            if line:
                data = line.decode('utf-8')
                result += data

        return result
