"""모델 허용목록(model_registry) 테스트."""
from app.services import model_registry as mr


def test_unspecified_model_is_allowed():
    assert mr.model_request_error(None) is None
    assert mr.model_request_error("  ") is None
    assert mr.model_request_error(None, admin=True) is None


def test_ask_allows_only_default_model_by_default(monkeypatch):
    monkeypatch.delenv("ASK_MODEL_ALLOWLIST", raising=False)
    assert mr.model_request_error(mr.DEFAULT_ALIAS) is None
    other = next(n for n in mr.REGISTRY if n != mr.DEFAULT_ALIAS)
    assert mr.model_request_error(other) == "허용되지 않은 모델입니다."


def test_ask_allowlist_from_env(monkeypatch):
    monkeypatch.setenv("ASK_MODEL_ALLOWLIST", "gemma-3-12b, qwen3-14b")
    assert mr.model_request_error("gemma-3-12b") is None
    assert mr.model_request_error("qwen3-14b") is None
    assert mr.model_request_error("Qwen/Qwen2.5-7B-Instruct") is not None


def test_admin_allows_registered_but_not_arbitrary(monkeypatch):
    monkeypatch.delenv("OPENAI_ALIAS_URLS", raising=False)
    assert mr.model_request_error("Qwen/Qwen3-14B-FP8", admin=True) is None
    assert mr.model_request_error("attacker/huge-model", admin=True) == "등록되지 않은 모델입니다."


def test_alias_url_models_count_as_registered(monkeypatch):
    monkeypatch.setenv("OPENAI_ALIAS_URLS", '{"colleague-gemma": "http://host.docker.internal:8002/v1"}')
    assert mr.is_registered("colleague-gemma")


def test_malformed_alias_urls_do_not_crash(monkeypatch):
    monkeypatch.setenv("OPENAI_ALIAS_URLS", "{not json")
    assert mr.is_registered("attacker/model") is False
    monkeypatch.setenv("OPENAI_ALIAS_URLS", '["list", "not", "dict"]')
    assert mr.is_registered("list") is False
