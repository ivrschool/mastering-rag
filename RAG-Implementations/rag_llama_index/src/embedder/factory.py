from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch

class EmbedderFactory:
    @staticmethod
    def create_embedder(config):
        provider = config['embedding']['provider']
        
        if provider == 'huggingface':
            model_name = config['embedding']['model_name']

            # Manually select device to avoid get_default_device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            return HuggingFaceEmbedding(
                model_name=model_name,
                device=device  # Ensures compatibility with sentence-transformers
            )
        
        elif provider == 'openai':
            return OpenAIEmbedding(model=config['embedding']['model_name'])
        
        raise ValueError(f"Unsupported embedding provider: {provider}")
