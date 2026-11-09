"""오류 응답 내부 정보 비노출 테스트 (FastAPI 최소 앱)."""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.services.error_handlers import install_error_handlers

SECRET_DETAIL = "MinIO get_object failed: connect to minio:9000 user=admin path=/data/secret"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("ERROR_DETAIL_EXPOSE", raising=False)
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/llama/rag/meta/x")
    def http_500():
        raise HTTPException(500, SECRET_DETAIL)

    @app.get("/llama/rag/unavailable")
    def http_503():
        raise HTTPException(503, "Milvus 연결 대기/검색 실패: " + SECRET_DETAIL)

    @app.get("/llama/rag/crash")
    def crash():
        raise RuntimeError(SECRET_DETAIL)

    @app.get("/llama/rag/bad")
    def http_400():
        raise HTTPException(400, "질문을 입력해 주세요.")

    @app.get("/llama/admin/boom")
    def admin_500():
        raise HTTPException(500, SECRET_DETAIL)

    return TestClient(app, raise_server_exceptions=False)


def test_5xx_detail_is_hidden(client):
    r = client.get("/llama/rag/meta/x")
    assert r.status_code == 500
    body = r.json()
    assert "secret" not in r.text and "minio" not in r.text.lower()
    assert body["detail"].startswith("요청을 처리하는 중 오류")
    assert len(body["error_id"]) == 12


def test_503_uses_unavailable_message(client):
    r = client.get("/llama/rag/unavailable")
    assert r.status_code == 503
    assert "secret" not in r.text
    assert r.json()["detail"].startswith("서비스를 일시적으로")


def test_unhandled_exception_is_hidden(client):
    r = client.get("/llama/rag/crash")
    assert r.status_code == 500
    assert "secret" not in r.text
    assert "error_id" in r.json()


def test_4xx_message_is_kept(client):
    r = client.get("/llama/rag/bad")
    assert r.status_code == 400
    assert r.json() == {"detail": "질문을 입력해 주세요."}


def test_admin_path_keeps_detail(client):
    r = client.get("/llama/admin/boom")
    assert r.status_code == 500
    assert r.json() == {"detail": SECRET_DETAIL}


def test_expose_flag_keeps_detail(client, monkeypatch):
    monkeypatch.setenv("ERROR_DETAIL_EXPOSE", "1")
    r = client.get("/llama/rag/meta/x")
    assert r.json() == {"detail": SECRET_DETAIL}
