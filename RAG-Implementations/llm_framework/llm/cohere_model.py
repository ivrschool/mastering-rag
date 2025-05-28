import cohere
from llm.base import BaseLLM
from utils.config_loader import get_env, load_config
from llm.registry import register_model

config = load_config()

@register_model("cohere")
class CohereModel(BaseLLM):
    def __init__(self):
        self.api_key = get_env("COHERE_API_KEY")
        self.model_name = config["COHERE"]["model_name"]
        self.client = cohere.ClientV2(self.api_key)

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.message.content[0].text.strip()
        except Exception as e:
            return f"Error: {str(e)}"

