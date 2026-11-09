"""관리자 경로 보호(admin_gate) 테스트 — 문서 업로드 포함."""
import pytest

from app.services.admin_gate import evaluate, is_gated_path


@pytest.mark.parametrize("path", [
    "/llama/upload",
    "/llama/upload/",
    "//llama//upload",             # 중복 슬래시
    "/llama/x/../upload",          # 점 경로
    "/llama/admin/rag-config",
    "/llama/admin/ask",
    "/llama/finetune/start",
    "/llama/debug/milvus/by-doc",
    "/llama/docs",
    "/llama/files/purge",
])
def test_gated_paths(path):
    assert is_gated_path(path)


@pytest.mark.parametrize("path", [
    "/llama/ask",
    "/llama/java/convert-and-index",   # 홈페이지 WAS 연계는 별도(내부 토큰·Gateway 규칙)
    "/llama/job/abc",
    "/llama/jobs",
    "/llama/rag/docs",
    "/llama/files",
    "/llama/stt/transcribe",
    "/llama/healthz",
    "/llama/uploads",                  # 접두사만 같은 다른 경로
])
def test_open_paths(path):
    assert not is_gated_path(path)


def test_evaluate_requires_matching_secret(monkeypatch):
    monkeypatch.delenv("ADMIN_GATE_ENFORCE", raising=False)
    monkeypatch.setenv("ADMIN_GATE_SECRET", "gate-secret-value")
    assert evaluate("gate-secret-value")[0] is True
    assert evaluate("wrong")[:2] == (False, 403)
    assert evaluate("")[:2] == (False, 403)


def test_evaluate_without_secret_fails_closed(monkeypatch):
    monkeypatch.delenv("ADMIN_GATE_ENFORCE", raising=False)
    monkeypatch.delenv("ADMIN_GATE_SECRET", raising=False)
    assert evaluate("anything")[:2] == (False, 503)
