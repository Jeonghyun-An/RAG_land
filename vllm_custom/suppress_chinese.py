# /workspace/suppress_chinese.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Set

import torch
from transformers import AutoTokenizer

# vLLM 0.11.0(V1)에서는 이 interface의 __init__이 NotImplementedError를 던짐
from vllm.v1.sample.logits_processor.interface import LogitsProcessor


class ChineseTokenSuppressor(LogitsProcessor):
    """
    vLLM v0.11.0 (V1 engine) custom logits processor.
    - Suppresses CJK(Chinese) tokens by setting their logits to -inf.
    - IMPORTANT: Do NOT call super().__init__() (it raises NotImplementedError in v0.11.0).
    """

    CJK_RANGES = [
        (0x4E00, 0x9FFF),    # CJK Unified Ideographs
        (0x3400, 0x4DBF),    # Extension A
        (0xF900, 0xFAFF),    # Compatibility Ideographs
        (0x20000, 0x2CEAF),  # Extension B~E
    ]

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        # ✅ 절대 super().__init__() 호출하지 말 것 (0.11.0에서는 바로 NotImplementedError)
        self.vllm_config = vllm_config
        self.device = device
        self.is_pin_memory = is_pin_memory

        self.enabled = os.getenv("SUPPRESS_CHINESE_TOKENS", "1").strip() == "1"
        self.verbose = os.getenv("SUPPRESS_CHINESE_VERBOSE", "0").strip() == "1"

        # 모델 이름 추출 (환경변수 우선, 없으면 vllm_config에서 최대한 찾아봄)
        self.model_name = (
            os.getenv("MODEL_NAME")
            or os.getenv("MODEL_ID")
            or self._try_get_model_name_from_config(vllm_config)
        )

        self._ids_cpu: Optional[torch.Tensor] = None
        self._ids_cache: Dict[str, torch.Tensor] = {}

        if not self.enabled:
            print("[ChineseSuppressor] Disabled (SUPPRESS_CHINESE_TOKENS=0)")
            return

        if not self.model_name:
            print("[ChineseSuppressor] MODEL_NAME/MODEL_ID not found -> disabled")
            self.enabled = False
            return

        print(f"[ChineseSuppressor] Loading tokenizer for: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        vocab_size = len(tokenizer.get_vocab())
        if self.verbose:
            print(f"[ChineseSuppressor] Scanning vocab: {vocab_size} tokens...")

        ids = self._build_cjk_token_ids(tokenizer)
        self._ids_cpu = torch.tensor(sorted(ids), dtype=torch.long, device="cpu")

        suppressed = int(self._ids_cpu.numel())
        ratio = (suppressed / vocab_size * 100.0) if vocab_size else 0.0

        print("[ChineseSuppressor] Ready:")
        print(f"  - Total vocab: {vocab_size}")
        print(f"  - Suppressed CJK token ids: {suppressed}")
        print(f"  - Suppression ratio: {ratio:.2f}%")

    def _try_get_model_name_from_config(self, vllm_config) -> Optional[str]:
        # vLLM 내부 config 구조가 환경에 따라 달라서 안전하게 try-chain
        try:
            mc = getattr(vllm_config, "model_config", None)
            if mc is None:
                return None
            for key in ("model", "model_name", "hf_model_name", "tokenizer"):
                val = getattr(mc, key, None)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except Exception:
            return None
        return None

    def _is_cjk_char(self, ch: str) -> bool:
        if not ch or len(ch) != 1:
            return False
        code = ord(ch)
        for start, end in self.CJK_RANGES:
            if start <= code <= end:
                return True
        return False

    def _contains_cjk(self, s: str) -> bool:
        if not s:
            return False
        return any(self._is_cjk_char(c) for c in s)

    def _build_cjk_token_ids(self, tokenizer) -> Set[int]:
        vocab = tokenizer.get_vocab()
        ids: Set[int] = set()

        for token, tid in vocab.items():
            # decode 시도
            try:
                decoded = tokenizer.convert_tokens_to_string([token])
            except Exception:
                decoded = token

            if self._contains_cjk(token) or self._contains_cjk(decoded):
                ids.add(tid)

        return ids

    def _ids_on_device(self, device: torch.device) -> torch.Tensor:
        assert self._ids_cpu is not None
        key = str(device)
        if key in self._ids_cache:
            return self._ids_cache[key]
        ids_dev = self._ids_cpu.to(device=device, non_blocking=True)
        self._ids_cache[key] = ids_dev
        return ids_dev

    # vLLM V1 요구 메서드들
    def is_argmax_invariant(self) -> bool:
        # argmax 자체를 바꾸는 로직이므로 False
        return False

    def update_state(self, *args: Any, **kwargs: Any) -> None:
        # 상태 필요 없음
        return None

    # vLLM에서 호출할 수 있도록 __call__ 제공 (내부에서 apply 수행)
    def __call__(self, logits: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.apply(logits, **kwargs)

    def apply(self, logits: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if (not self.enabled) or (self._ids_cpu is None):
            return logits

        ids = self._ids_on_device(logits.device)
        neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)

        # logits shape: [vocab] or [batch, vocab]
        if logits.dim() == 1:
            logits.index_fill_(0, ids, neg_inf)
        elif logits.dim() == 2:
            logits.index_fill_(1, ids, neg_inf)

        return logits
