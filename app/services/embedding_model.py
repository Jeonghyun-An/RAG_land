# app/services/embedding_model.py
from __future__ import annotations

import os
import traceback
from functools import lru_cache
from typing import List, Tuple

# -----------------------------
# 환경 변수
# -----------------------------
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")  # 범용 성능/품질 밸런스
EMBED_MAX_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "512"))  # 법령/규정에 권장 384~512
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))  # GPU 사용 전제
EMBEDDING_DEVICE_ENV = os.getenv("EMBEDDING_DEVICE", "auto").lower()  # auto|cuda|cpu
EMBED_DTYPE = os.getenv("EMBED_DTYPE", "auto").lower()  # auto|bf16|fp16|fp32

# 토크나이저 멀티스레딩 경고 억제 + CPU 점유 완화
os.environ.setdefault("TOKENIZERS_PARALLELISM", os.getenv("TOKENIZERS_PARALLELISM", "false"))
EMBED_NUM_THREADS = int(os.getenv("EMBED_NUM_THREADS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(EMBED_NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(EMBED_NUM_THREADS))


def _select_device() -> str:
    if EMBEDDING_DEVICE_ENV in ("cpu", "cuda"):
        if EMBEDDING_DEVICE_ENV == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
            print("[EMBED] Requested CUDA but not available → fallback to CPU")
            return "cpu"
        return "cpu"

    # auto
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _select_dtype(device: str):
    """
    GPU면 bf16>fp16>fp32 순으로 시도, CPU는 fp32.
    """
    try:
        import torch
    except Exception:
        return None

    if device != "cuda":
        return torch.float32

    # 사용자가 명시했으면 우선
    if EMBED_DTYPE == "bf16":
        return torch.bfloat16
    if EMBED_DTYPE == "fp16":
        return torch.float16
    if EMBED_DTYPE == "fp32":
        return torch.float32

    # auto
    if torch.cuda.is_available():
        # bf16 지원이면 bf16, 아니면 fp16
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def _limit_threads() -> None:
    try:
        import torch
        torch.set_num_threads(max(1, EMBED_NUM_THREADS))
    except Exception:
        pass


# -----------------------------
# 로더
# -----------------------------
@lru_cache
def _load_embedding_impl() -> Tuple[object, str]:
    """
    1) sentence-transformers (권장)
    2) FlagEmbedding(BGE-M3) (설치돼 있으면 시도)
    반환: (impl, kind)  kind in {"st","flag"}
    """
    device = _select_device()
    _limit_threads()

    st_err = None
    flag_err = None

    # 1) sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        import torch

        print(f"[EMBED] Loading (ST) model='{DEFAULT_MODEL}' device='{device}'")
        model = SentenceTransformer(DEFAULT_MODEL, device=device, trust_remote_code=True)
        # 길이 제한(문장 자르기 기준)
        try:
            model.max_seq_length = EMBED_MAX_TOKENS
        except Exception:
            pass

        # dtype 조정(가능할 때만)
        try:
            dtype = _select_dtype(device)
            if device == "cuda":
                if dtype == torch.bfloat16:
                    model = model.to(dtype=torch.bfloat16)
                elif dtype == torch.float16:
                    # 일부 ST 래퍼는 .half()가 더 안전
                    try:
                        model = model.half()
                    except Exception:
                        model = model.to(dtype=torch.float16)
            # CPU는 fp32 유지
        except Exception as _e:
            print(f"[EMBED] dtype adjust skipped: {_e}")

        return model, "st"
    except Exception as e:
        st_err = e
        print(f"[EMBED] ST load failed: {e}\n{traceback.format_exc()}")

    # 2) FlagEmbedding(BGE-M3)
    try:
        import importlib
        if importlib.util.find_spec("FlagEmbedding") is not None:
            from FlagEmbedding import BGEM3FlagModel
            use_fp16 = (device == "cuda" and EMBED_DTYPE in ("auto", "fp16", "bf16"))
            print(f"[EMBED] Loading (FlagEmbedding) model='{DEFAULT_MODEL}' device='{device}' fp16={use_fp16}")
            model = BGEM3FlagModel(DEFAULT_MODEL, use_fp16=use_fp16, device=device)
            return model, "flag"
        else:
            flag_err = ModuleNotFoundError("FlagEmbedding not installed")
    except Exception as e:
        flag_err = e
        print(f"[EMBED] FlagEmbedding load failed: {e}\n{traceback.format_exc()}")

    # 모두 실패
    raise RuntimeError(
        "임베딩 모델 로드 실패: sentence-transformers 또는 FlagEmbedding 중 하나가 필요합니다. "
        "requirements 및 CUDA/드라이버를 확인하세요."
        f"[ST 에러: {st_err!r}] [Flag 에러: {flag_err!r}]"
    )


def get_embedding_model():
    model, _ = _load_embedding_impl()
    return model


# -----------------------------
# API
# -----------------------------
def embed(texts: List[str]) -> List[List[float]]:
    """문장/청크 임베딩 반환. normalize=True 고정."""
    if not texts:
        return []
    model, kind = _load_embedding_impl()

    if kind == "st":
        # sentence-transformers
        try:
            if hasattr(model, "max_seq_length"):
                model.max_seq_length = EMBED_MAX_TOKENS
        except Exception:
            pass

        vecs = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
        )
        return vecs.tolist()

    # FlagEmbedding 경로 (BGEM3FlagModel)
    outs = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        max_length=EMBED_MAX_TOKENS,
    )
    vecs = outs.get("dense_vecs", outs)  # 일부 버전 호환
    try:
        return vecs.tolist()
    except Exception:
        return list(vecs)


def get_sentence_embedding_dimension() -> int:
    model, kind = _load_embedding_impl()
    # ST
    if kind == "st":
        try:
            return model.get_sentence_embedding_dimension()
        except Exception:
            pass
    # FlagEmbedding(BGE-M3)
    try:
        dim = getattr(model, "embedding_size", None)
        if isinstance(dim, int):
            return dim
    except Exception:
        pass
    # 안전 폴백(BGE-M3=1024)
    return 1024
