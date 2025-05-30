from transformers import pipeline
from llm_framework.llm.base import BaseLLM
from llm_framework.utils.config_loader import get_env, load_config
from llm_framework.llm.registry import register_model

config = load_config()

@register_model("hf")
class HFModel(BaseLLM):
    def __init__(self):
        self.model_name = config["HF"]["model_name"]
        self.token = get_env("HF_TOKEN")
        self.pipeline = pipeline("text2text-generation", model=self.model_name, token=self.token, device=-1)

    def generate_response(self, prompt: str) -> str:
        return self.pipeline(prompt)[0]["generated_text"]
