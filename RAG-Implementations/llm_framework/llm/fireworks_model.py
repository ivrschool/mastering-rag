import os
from openai import OpenAI
from llm.base import BaseLLM
from utils.config_loader import get_env, load_config
from llm.registry import register_model

config = load_config()

@register_model("fireworks")
class FireworksModel(BaseLLM):
    def __init__(self):
        self.api_key = get_env("FIREWORKS_API_KEY")
        self.model_name = config["FIREWORKS"]["model_name"]
        self.temperature = float(config["FIREWORKS"].get("temperature", 0.0))
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.fireworks.ai/inference/v1"
        )

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"
