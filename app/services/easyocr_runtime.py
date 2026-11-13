# app/services/easyocr_runtime.py
"""
EasyOCR 실행 공통부 — Reader 재사용, GPU 메모리 부족 시 CPU 로 다시 읽기.

배경 (2026-11-13 운영)
  같은 GPU(L40S 46GB)에 vLLM(Gemma 4 26B, 75%)·STT·임베딩이 함께 올라가 있어
  EasyOCR 글자 영역 검출이 GPU 메모리 1.11GiB 를 받지 못하고 CUDA out of memory 로 실패했다.
  Java 연계 업로드 경로는 이 오류를 조용히 삼켜, 스캔 문서가 글자 없이(자리표시만) 색인됐다.

동작
  - Reader 는 (언어, GPU 여부)별로 한 번만 만들어 재사용한다(문서마다 모델을 다시 올리지 않음).
  - GPU 로 읽다가 메모리가 모자라면 GPU 캐시를 비우고 같은 페이지를 CPU Reader 로 다시 읽는다.
    그 뒤 OCR_GPU_RETRY_SEC(기본 300초) 동안은 GPU 를 다시 시도하지 않고 CPU 로 읽는다.
  - 메모리 부족이 아닌 오류는 숨기지 않고 그대로 올린다.
  - 같은 Reader 를 여러 작업이 동시에 쓰지 않게 잠근다(동시 OCR 로 GPU 메모리가 몰리는 것도 줄어든다).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Iterable, Tuple

_GUARD = threading.Lock()
_READERS: Dict[Tuple[Tuple[str, ...], bool], Any] = {}
_LOCKS: Dict[Tuple[Tuple[str, ...], bool], threading.Lock] = {}
_cpu_until = 0.0          # 이 시각(monotonic)까지는 GPU 대신 CPU 로 읽는다


def get_reader(langs: Iterable[str], gpu: bool):
    """(Reader, 그 Reader 전용 잠금)을 돌려준다. 처음 한 번만 만든다."""
    key = (tuple(langs), bool(gpu))
    with _GUARD:
        reader = _READERS.get(key)
        if reader is None:
            import easyocr
            print(f"[OCR] EasyOCR Reader 생성: langs={list(key[0])}, gpu={key[1]}")
            reader = easyocr.Reader(list(key[0]), gpu=key[1])
            _READERS[key] = reader
            _LOCKS[key] = threading.Lock()
        return reader, _LOCKS[key]


def is_cuda_oom(exc: BaseException) -> bool:
    if type(exc).__name__ == "OutOfMemoryError":          # torch.cuda.OutOfMemoryError
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cublas_status_alloc_failed" in msg


def _free_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _gpu_retry_sec() -> float:
    try:
        return float(os.getenv("OCR_GPU_RETRY_SEC") or 300)
    except ValueError:
        return 300.0


def readtext(img, langs: Iterable[str], gpu: bool, **kwargs):
    """reader.readtext(img, **kwargs) — GPU 메모리가 모자라면 CPU 로 다시 읽는다."""
    global _cpu_until
    langs = tuple(langs)
    if gpu and time.monotonic() >= _cpu_until:
        oom = ""
        try:
            reader, lock = get_reader(langs, True)
            with lock:
                return reader.readtext(img, **kwargs)
        except Exception as e:
            if not is_cuda_oom(e):
                raise
            oom = (str(e).splitlines() or [""])[0][:160]
        # 예외 객체(와 그 traceback 이 붙잡은 GPU 메모리)를 놓은 뒤에 캐시를 비운다
        _free_cuda_cache()
        retry = _gpu_retry_sec()
        _cpu_until = time.monotonic() + retry
        print(f"[OCR] GPU 메모리 부족 → CPU 로 읽음 ({retry:.0f}초 동안 GPU 재시도 안 함): {oom}")
    reader, lock = get_reader(langs, False)
    with lock:
        return reader.readtext(img, **kwargs)
