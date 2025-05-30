from llama_index.core import SimpleDirectoryReader
from pathlib import Path

def load_documents(data_path: str):
    reader = SimpleDirectoryReader(Path(data_path))
    return reader.load_data()
