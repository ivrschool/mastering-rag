from llm_framework.utils.config_loader import load_config
from llm_framework.llm.openai_model import OpenAIModel
from llm_framework.llm.hf_model import HFModel
from llm_framework.llm.deepseek_model import DeepSeekModel
from llm_framework.llm.fireworks_model import FireworksModel



def get_model():
    config = load_config()
    provider = config["MODEL"]["provider"].lower()

    if provider == "openai":
        return OpenAIModel()
    elif provider == "hf":
        return HFModel()
    elif provider == "fireworks":
        return FireworksModel()
    elif provider == "deepseek":
        return DeepSeekModel()
    else:
        raise ValueError(f"Unsupported model provider: {provider}")



