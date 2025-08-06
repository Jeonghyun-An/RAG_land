# app/services/file_parser.py
from pdfminer.high_level import extract_text

def parse_pdf(file_path: str) -> str:
    text = extract_text(file_path)
    return text
