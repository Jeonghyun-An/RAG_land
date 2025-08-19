# app/services/file_parser.py
from __future__ import annotations
from typing import List, Tuple, Union

def parse_pdf(path: str, by_page: bool = False) -> Union[str, List[Tuple[int, str]]]:
    # 1) 먼저 pdfminer 사용
    try:
        import pdfminer.high_level
        from pdfminer.layout import LAParams
        laparams = LAParams()
        if not by_page:
            return pdfminer.high_level.extract_text(path, laparams=laparams)
        pages = []
        for i, page in enumerate(pdfminer.high_level.extract_pages(path, laparams=laparams), start=1):
            text = "".join([elem.get_text() for elem in page if hasattr(elem, "get_text")]).strip()
            if text:
                pages.append((i, text))
        return pages
    except Exception:
        # 2) 폴백: PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(path)
            if not by_page:
                return "\n\n".join([p.extract_text() or "" for p in reader.pages]).strip()
            pages = []
            for i, p in enumerate(reader.pages, start=1):
                t = (p.extract_text() or "").strip()
                if t:
                    pages.append((i, t))
            return pages
        except Exception as e:
            raise RuntimeError(f"PDF 텍스트 추출 실패: {e}")
