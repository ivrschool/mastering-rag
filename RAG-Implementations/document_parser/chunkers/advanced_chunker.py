import nltk
nltk.download('punkt_tab')
from typing import List
from document_parser.chunkers.base_chunker import BaseChunker
import configparser

config = configparser.ConfigParser()
config.read("document_parser/config/config.ini")
chunk_size = int(config["CHUNKING"].get("chunk_size", 5))
overlap = int(config["CHUNKING"].get("overlap", 1))
unit = config["CHUNKING"].get("unit", "sentence")  # 'sentence' or 'word'

class AdvancedChunker(BaseChunker):
    def chunk(self, text: str) -> List[str]:
        if unit == "sentence":
            return self._sentence_chunks(text)
        elif unit == "word":
            return self._word_chunks(text)
        else:
            raise ValueError(f"[ERROR] Unsupported unit '{unit}'")

    def _sentence_chunks(self, text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk = " ".join(sentences[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _word_chunks(self, text: str) -> List[str]:
        words = nltk.word_tokenize(text)
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
