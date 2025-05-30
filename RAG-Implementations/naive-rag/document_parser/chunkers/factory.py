from document_parser.chunkers.simple_chunker import SimpleChunker
from document_parser.chunkers.advanced_chunker import AdvancedChunker

def get_chunker(name: str):
    if name == "simple":
        return SimpleChunker()
    elif name == "advanced":
        return AdvancedChunker()
    else:
        raise ValueError(f"No chunker found for: {name}")
