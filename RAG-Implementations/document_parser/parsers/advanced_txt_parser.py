import unicodedata
import re
from document_parser.parsers.base_parser import BaseParser

class AdvancedTxtParser(BaseParser):
    def parse(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Normalize Unicode
        text = unicodedata.normalize("NFKC", raw_text)

        # Replace multiple spaces and normalize newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Optionally: Remove non-printable/control characters
        text = ''.join(c for c in text if c.isprintable())

        # Additional filters could be added here (e.g., spellcheck, noise removal)
        return text
