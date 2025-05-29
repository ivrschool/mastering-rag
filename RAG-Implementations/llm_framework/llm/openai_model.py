from openai import OpenAI
from llm_framework.llm.base import BaseLLM
from llm_framework.utils.config_loader import get_env, load_config
from llm_framework.llm.registry import register_model

config = load_config()

@register_model("openai")
class OpenAIModel(BaseLLM):
    def __init__(self):
        self.api_key = get_env("OPENAI_API_KEY")
        self.model_name = config["OPENAI"]["model_name"]
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
