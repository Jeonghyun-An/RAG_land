# app/services/chunker.py
def chunk_text(text: str, max_length: int = 500) -> list[str]:
    lines = text.split("\n")
    chunks = []
    chunk = ""
    for line in lines:
        if len(chunk) + len(line) < max_length:
            chunk += line + "\n"
        else:
            chunks.append(chunk.strip())
            chunk = line + "\n"
    if chunk:
        chunks.append(chunk.strip())
    return chunks
