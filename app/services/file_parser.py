import fitz
from typing import Optional

def parse_pdf(path: str) -> Optional[str]:
    try:
        doc = fitz.open(path)
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))  # or "blocks"
        return "\n".join(texts)
    except Exception as e:
        print(f"[ERROR] PDF parse failed: {e}")
        return None
