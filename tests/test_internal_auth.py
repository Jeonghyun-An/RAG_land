"""홈페이지 연계 API 내부 토큰 확인 테스트."""
from app.services.internal_auth import verify_internal_token


def test_no_secret_allows_when_not_required(monkeypatch):
    monkeypatch.delenv("JAVA_TOKEN_REQUIRED", raising=False)
    assert verify_internal_token(None, secret="") is True


def test_no_secret_rejects_when_required(monkeypatch):
    monkeypatch.setenv("JAVA_TOKEN_REQUIRED", "1")
    assert verify_internal_token("anything", secret="") is False


def test_secret_set_checks_token(monkeypatch):
    monkeypatch.delenv("JAVA_TOKEN_REQUIRED", raising=False)
    assert verify_internal_token("s3cret-value", secret="s3cret-value") is True
    assert verify_internal_token("wrong", secret="s3cret-value") is False
    assert verify_internal_token(None, secret="s3cret-value") is False
    assert verify_internal_token("", secret="s3cret-value") is False


def test_reads_secret_from_env(monkeypatch):
    monkeypatch.setenv("JAVA_SHARED_SECRET", "from-env")
    assert verify_internal_token("from-env") is True
    assert verify_internal_token("other") is False
