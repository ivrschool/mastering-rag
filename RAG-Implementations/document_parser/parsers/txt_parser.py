from document_parser.parsers.base_parser import BaseParser

class TxtParser(BaseParser):
    def parse(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
