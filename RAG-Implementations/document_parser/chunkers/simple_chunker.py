from document_parser.chunkers.base_chunker import BaseChunker
from typing import List
import configparser

config = configparser.ConfigParser()
config.read("document_parser/config/config.ini")
chunk_size = int(config["CHUNKING"]["chunk_size"])
overlap = int(config["CHUNKING"]["overlap"])

class SimpleChunker(BaseChunker):
    def chunk(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks
