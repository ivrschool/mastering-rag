from transformers import pipeline
from llm.base import BaseLLM
from utils.config_loader import get_env, load_config
from llm.registry import register_model

config = load_config()

@register_model("deepseek")
class DeepSeekModel(BaseLLM):
    def __init__(self):
        self.model_name = config["DEEPSEEK"]["model_name"]
        self.token = get_env("HF_TOKEN")
        self.pipeline = pipeline(
            "text-generation",  # or text2text-generation if appropriate
            model=self.model_name,
            token=self.token
        )

    def generate_response(self, prompt: str) -> str:
        return self.pipeline(prompt, max_length=100)[0]["generated_text"]
