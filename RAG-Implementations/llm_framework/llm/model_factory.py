from utils.config_loader import load_config
import importlib
from llm.registry import get_model_class
import os

def auto_import_models():
    llm_dir = os.path.dirname(__file__)
    for filename in os.listdir(llm_dir):
        if (
            filename.endswith(".py") and
            filename not in ["__init__.py", "base.py", "registry.py", "model_factory.py"]
        ):
            module_name = f"llm.{filename[:-3]}"
            importlib.import_module(module_name)

# Call this to ensure models are registered
auto_import_models()


def get_model():
    config = load_config()
    provider = config["MODEL"]["provider"]
    model_class = get_model_class(provider)
    return model_class()


