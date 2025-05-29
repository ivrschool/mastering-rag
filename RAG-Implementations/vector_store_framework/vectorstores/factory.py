import os
import importlib
from vectorstores.registry import get_vector_store_class
from utils.config_loader import load_config

def auto_import_backends():
    dir_path = os.path.dirname(__file__)
    for file in os.listdir(dir_path):
        if file.endswith(".py") and file not in ["__init__.py", "base.py", "registry.py", "factory.py"]:
            importlib.import_module(f"vectorstores.{file[:-3]}")

auto_import_backends()

def get_vector_store():
    config = load_config()
    backend = config["VECTORSTORE"]["provider"]
    store_class = get_vector_store_class(backend)
    return store_class()
