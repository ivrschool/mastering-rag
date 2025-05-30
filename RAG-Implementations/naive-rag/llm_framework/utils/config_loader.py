from configparser import ConfigParser
from dotenv import load_dotenv
import os

load_dotenv()

def load_config():
    config = ConfigParser()
    config.read("llm_framework/config/config.ini")
    return config

def get_env(key):
    return os.getenv(key)
