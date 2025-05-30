from llama_index.llms.fireworks import Fireworks
from llama_index.llms.openai import OpenAI

class LLMFactory:
    @staticmethod
    def create_llm(config):
        if config['llm']['provider'] == 'fireworks':
            return Fireworks(api_key=config['llm']['api_key'], model=config['llm']['model'])
        elif config['llm']['provider'] == 'openai':
            return OpenAI(api_key=config['llm']['api_key'], model=config['llm']['model'])
        raise ValueError("Unsupported LLM provider")
