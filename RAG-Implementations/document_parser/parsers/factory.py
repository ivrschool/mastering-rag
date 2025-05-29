from document_parser.parsers.txt_parser import TxtParser
from document_parser.parsers.advanced_txt_parser import AdvancedTxtParser
from document_parser.parsers.base_parser import BaseParser

def get_parser(file_type: str, mode: str = "simple") -> BaseParser:
    if file_type == "txt":
        if mode == "simple":
            return TxtParser()
        elif mode == "advanced":
            return AdvancedTxtParser()
    raise NotImplementedError(f"No parser implemented for file type: {file_type}, mode: {mode}")
