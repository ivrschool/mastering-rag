import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

def load_config():
    # Locate .env file outside the project directory
    env_path = (Path(__file__).resolve().parent.parent.parent / ".env")
    load_dotenv(dotenv_path=env_path)

    # Load config.yaml inside project
    config_path = Path(__file__).parent.parent.parent / "rag_llama_index" / "config" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Inject env vars using keys from config
    if "api_key_env" in config["llm"]:
        config["llm"]["api_key"] = os.getenv(config["llm"]["api_key_env"])
    if "api_key_env" in config["vector_store"]:
        config["vector_store"]["api_key"] = os.getenv(config["vector_store"]["api_key_env"])

    return config
