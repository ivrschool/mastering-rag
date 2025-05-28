from utils.config_loader import load_config
from llm import openai_model, hf_model, deepseek_model, fireworks_model, cohere_model
from llm.registry import get_model_class
# from llm.openai_model import OpenAIModel
# from llm.hf_model import HFModel
# from llm.deepseek_model import DeepSeekModel
# from llm.fireworks_model import FireworksModel



# def get_model():
#     config = load_config()
#     provider = config["MODEL"]["provider"].lower()

#     if provider == "openai":
#         return OpenAIModel()
#     elif provider == "hf":
#         return HFModel()
#     elif provider == "fireworks":
#         return FireworksModel()
#     elif provider == "deepseek":
#         return DeepSeekModel()
#     else:
#         raise ValueError(f"Unsupported model provider: {provider}")

def get_model():
    config = load_config()
    provider = config["MODEL"]["provider"]
    
    model_class = get_model_class(provider)
    return model_class()


