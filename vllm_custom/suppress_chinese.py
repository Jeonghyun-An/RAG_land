#!/usr/bin/env python3
"""
vLLM 서버용 중국어(CJK) 토큰 억제 LogitsProcessor (callable)

중요:
- vllm.sampling_params.LogitsProcessor 는 "typing alias"라서 subclass 불가
- 따라서 "callable 객체"로 구현해야 함
"""
import os
import torch
from typing import List, Set, Optional

# 상속 제거 (절대: from vllm.sampling_params import LogitsProcessor 하지 말기)


class ChineseTokenSuppressor:
    """
    중국어/한자(CJK) 토큰을 억제하는 callable

    vLLM 버전에 따라 logits_processor 시그니처가 2가지가 있을 수 있어서:
    - (token_ids, logits) 또는
    - (prompt_token_ids, token_ids, logits)
    둘 다 처리하도록 *args 로 받는다.
    """

    CHINESE_RANGES = [
        (0x4E00, 0x9FFF),    # CJK Unified Ideographs
        (0x3400, 0x4DBF),    # CJK Extension A
        (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
        (0x20000, 0x2CEAF),  # CJK Extension B~E 일부
    ]

    def __init__(self, tokenizer, enabled: bool = True, verbose: bool = False):
        self.enabled = enabled
        self.verbose = verbose
        self.tokenizer = tokenizer

        self._chinese_token_ids: Set[int] = set()
        self._ids_cpu: Optional[torch.Tensor] = None  # torch.long tensor (CPU)
        self._ids_device_cache = {}  # device(str) -> tensor(long) on that device

        if not self.enabled:
            print("[ChineseSuppressor] Disabled")
            return

        print("[ChineseSuppressor] Initializing...")
        self._chinese_token_ids = self._build_chinese_token_set()

        # CPU 텐서로 미리 만들어두고, 호출 시 logits device로 옮겨서 사용
        if self._chinese_token_ids:
            self._ids_cpu = torch.tensor(sorted(self._chinese_token_ids), dtype=torch.long)

        if self.verbose:
            vocab_size = len(tokenizer.get_vocab())
            suppressed = len(self._chinese_token_ids)
            ratio = (suppressed / vocab_size * 100) if vocab_size > 0 else 0
            print("[ChineseSuppressor] Ready:")
            print(f"  - Total vocab: {vocab_size}")
            print(f"  - Chinese tokens: {suppressed}")
            print(f"  - Suppression ratio: {ratio:.2f}%")

    def _is_chinese_char(self, char: str) -> bool:
        if not char or len(char) != 1:
            return False
        code = ord(char)
        return any(start <= code <= end for start, end in self.CHINESE_RANGES)

    def _contains_chinese(self, text: str) -> bool:
        if not text:
            return False
        return any(self._is_chinese_char(c) for c in text)

    def _build_chinese_token_set(self) -> Set[int]:
        chinese_ids = set()
        vocab = self.tokenizer.get_vocab()

        if self.verbose:
            print(f"[ChineseSuppressor] Scanning {len(vocab)} tokens...")

        for token, token_id in vocab.items():
            try:
                decoded = self.tokenizer.convert_tokens_to_string([token])
                if self._contains_chinese(decoded):
                    chinese_ids.add(token_id)
            except Exception:
                if self._contains_chinese(token):
                    chinese_ids.add(token_id)

        return chinese_ids

    def _get_ids_on_device(self, device: torch.device) -> Optional[torch.Tensor]:
        if self._ids_cpu is None or self._ids_cpu.numel() == 0:
            return None
        key = str(device)
        if key in self._ids_device_cache:
            return self._ids_device_cache[key]
        ids_dev = self._ids_cpu.to(device=device, non_blocking=True)
        self._ids_device_cache[key] = ids_dev
        return ids_dev

    def __call__(self, *args):
        """
        vLLM logits processor 호환:
          - (token_ids, logits)
          - (prompt_token_ids, token_ids, logits)
        """
        if not self.enabled or not self._chinese_token_ids:
            # args 중 logits 그대로 반환해야 함
            return args[-1]

        logits = args[-1]  # 마지막 인자가 logits
        # logits shape: [vocab] 또는 [batch, vocab] 등일 수 있음
        ids = self._get_ids_on_device(logits.device)
        if ids is None:
            return logits

        # ✅ Python for-loop 금지 (느림). 텐서 인덱싱으로 한 방에 마스킹
        # logits[..., ids] = -inf
        logits.index_fill_(-1, ids, float("-inf"))
        return logits


def create_chinese_suppressor(tokenizer, enabled: Optional[bool] = None):
    if enabled is None:
        enabled = os.getenv("SUPPRESS_CHINESE_TOKENS", "1").strip() == "1"
    verbose = os.getenv("SUPPRESS_CHINESE_VERBOSE", "0").strip() == "1"
    return ChineseTokenSuppressor(tokenizer, enabled=enabled, verbose=verbose)
