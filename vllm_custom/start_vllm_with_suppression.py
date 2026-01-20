#!/usr/bin/env python3
"""
vLLM OpenAI 서버를 중국어 토큰 억제 패치와 함께 실행.

핵심:
- vLLM 내부 argparse/cli API는 버전마다 깨지기 쉬움
- 특히 Python 3.12에서는 argparse가 deprecated= 키워드를 못 받아 터질 수 있음
- 따라서 vLLM을 "python -m vllm.entrypoints.openai.api_server ..." 로 CLI 그대로 실행한다.
- 우리는 그 전에 SamplingParams.__init__만 패치해두면 된다.
"""
import os
import sys
import subprocess

from transformers import AutoTokenizer
from vllm.sampling_params import SamplingParams

try:
    from suppress_chinese import create_chinese_suppressor
    SUPPRESSOR_AVAILABLE = True
except Exception as e:
    print(f"[WARN] suppress_chinese import failed, suppression disabled: {e}")
    SUPPRESSOR_AVAILABLE = False


def _resolve_model_name_from_env_or_argv() -> str:
    if os.getenv("MODEL_NAME"):
        return os.getenv("MODEL_NAME")
    if "--model" in sys.argv:
        i = sys.argv.index("--model")
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return ""


def patch_sampling_params():
    if not SUPPRESSOR_AVAILABLE:
        return

    suppress_enabled = os.getenv("SUPPRESS_CHINESE_TOKENS", "1").strip() == "1"
    if not suppress_enabled:
        print("[ChineseSuppressor] Disabled by SUPPRESS_CHINESE_TOKENS=0")
        return

    model_name = _resolve_model_name_from_env_or_argv()
    if not model_name:
        print("[ChineseSuppressor] MODEL_NAME/--model not found. Suppression disabled.")
        return

    print(f"[ChineseSuppressor] Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    suppressor = create_chinese_suppressor(tokenizer, enabled=True)

    original_init = SamplingParams.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.logits_processors is None:
            self.logits_processors = []
        self.logits_processors.insert(0, suppressor)

    SamplingParams.__init__ = patched_init
    print("[ChineseSuppressor] SamplingParams patched successfully")


def exec_vllm_cli():
    # 현재 프로세스의 argv(--model ... 등)를 그대로 넘겨서 vLLM CLI를 실행
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *sys.argv[1:]]
    print("[vLLM] exec:", " ".join(cmd))
    # 현재 프로세스를 교체(exec)해도 되지만, 로그/제어를 위해 subprocess로 실행
    p = subprocess.run(cmd)
    raise SystemExit(p.returncode)


def main():
    patch_sampling_params()
    exec_vllm_cli()


if __name__ == "__main__":
    main()
