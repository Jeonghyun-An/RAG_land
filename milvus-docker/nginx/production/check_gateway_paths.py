#!/usr/bin/env python3
"""
새 app.conf 를 적용하기 전에, 지금 운영 중인 gateway 의 접속 로그를 새 규칙에 대입해
"막히거나 관리자 IP 전용이 되는 경로"를 미리 보여 준다. (표준 라이브러리만 사용)

사용법 (서버에서):
    docker logs gateway 2>&1 | python3 check_gateway_paths.py

읽는 법
  허용          새 설정에서도 그대로 통과
  관리자 전용   관리자 IP(admin_ips.inc)가 아니면 403  → 화면이 이 경로를 쓰면 관리자 IP 등록 필요
  차단          404/405/403  → 실제로 쓰이는 경로라면 규칙에 추가해야 함(적용하지 말고 알려 주세요)
  마지막 칸(X-Forwarded-For)이 '-' 이면 Apache 를 거치지 않은 직접 호출, IP 가 있으면 Apache 경유

app.conf 의 location 규칙을 그대로 옮긴 것이다. 규칙을 바꾸면 이 파일도 같이 고쳐야 한다.
"""
import collections
import posixpath
import re
import sys
from urllib.parse import unquote

# nginx 기본 로그: $remote_addr - $remote_user [time] "$request" $status $bytes "$referer" "$ua" "$xff"
LOG_DEFAULT = re.compile(r'^(\S+) - \S+ \[[^\]]+\] "(\S+) (\S+)[^"]*" (\d{3}) \S+ "[^"]*" "[^"]*" "([^"]*)"')
# 새 설정의 로그: $remote_addr [time] "$request_method $uri" $status ...
LOG_NEW = re.compile(r'^(\S+) \[[^\]]+\] "(\S+) (\S+)" (\d{3})')

ADMIN = re.compile(r"^/(rag/)?llama/(finetune|admin|dev|debug|milvus|db|docs|openapi\.json|generate|test|files/purge)(/|$)")
REGEXES = [  # app.conf 의 정규식 location 순서와 동일
    ("hidden", re.compile(r"/\."), "block"),
    ("admin", ADMIN, "admin"),
    ("ask", re.compile(r"^/rag/llama/ask(-stream)?(/|$)"), "allow"),
    ("job-stream", re.compile(r"^/rag/llama/job/[^/]+/stream$"), "allow"),
    ("upload", re.compile(r"^/rag/llama/upload(/|$)"), "admin"),
    ("stt", re.compile(r"^/rag/llama/stt(/|$)"), "allow"),
    ("service", re.compile(r"^/rag/llama/(jobs?|doc|files?|rag|status|view|download|healthz)(/|$)"), "allow"),
    ("java", re.compile(r"^/llama/java(/|$)"), "java"),
]
PREFIXES = [("/rag/minio/", True), ("/rag/stt/", True), ("/rag/", True),   # (경로, ^~ 여부)
            ("/rag/llama/", False), ("/llama/", False), ("/", False)]


def normalize(raw: str) -> str:
    """nginx 가 location 매칭에 쓰는 형태: 쿼리 제거, % 디코딩, 연속 슬래시 병합, . 와 .. 정리."""
    path = unquote(raw.split("?", 1)[0])
    path = re.sub(r"/{2,}", "/", path)
    parts = []
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    norm = "/" + "/".join(parts)
    if path.endswith("/") and parts:
        norm += "/"
    return norm


def classify(method: str, raw_path: str, xff: str):
    """(구분, 설명) 반환. 구분: allow / admin / block"""
    norm = normalize(raw_path)
    if norm in ("/healthz", "/", "/rag", "/llama", "/llama/healthz"):
        return "allow", "고정 경로"

    best = None
    for prefix, caret in PREFIXES:
        if norm.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, caret)
    prefix, caret = best

    if caret:                                  # ^~ 접두사는 정규식 검사를 건너뜀
        if prefix == "/rag/minio/":
            if method == "OPTIONS" or method in ("GET", "HEAD"):
                return "allow", "MinIO 조회"
            return "block", "MinIO 는 조회(GET/HEAD)만 허용 → 405"
        return "allow", "음성 변환" if prefix == "/rag/stt/" else "화면"

    for name, regex, kind in REGEXES:
        if regex.search(norm):
            if kind == "allow":
                return "allow", name
            if kind == "admin":
                return "admin", "관리자 IP 전용"
            if kind == "java":
                if xff not in ("", "-"):
                    return "block", "Apache 를 거친 /llama/java 호출 → 403"
                return "allow", "WAS 직접 호출"
            return "block", "숨김 파일 → 404"

    if prefix in ("/rag/llama/", "/llama/"):
        return "block", "허용목록 밖 → 404"
    return "allow", "화면"


def group_key(path: str) -> str:
    segs = normalize(path).strip("/").split("/")
    return "/" + "/".join(segs[:4]) + ("/…" if len(segs) > 4 else "")


def main():
    stats = collections.Counter()
    detail = {"admin": collections.Counter(), "block": collections.Counter()}
    notes = {}
    xff_seen = collections.Counter()
    admin_clients = collections.Counter()
    java_calls = collections.Counter()
    parsed = skipped = 0

    for line in sys.stdin:
        m = LOG_DEFAULT.match(line)
        if m:
            _, method, path, status, xff = m.groups()
        else:
            m = LOG_NEW.match(line)
            if not m:
                skipped += 1
                continue
            _, method, path, status = m.groups()
            xff = "?"
        parsed += 1
        kind, why = classify(method, path, xff)
        stats[kind] += 1
        if "/rag/llama/ask" in path:
            xff_seen["있음(Apache 가 전달)" if xff not in ("-", "", "?") else ("없음('-')" if xff != "?" else "알 수 없음")] += 1
        if "/llama/java/" in path and method == "POST":
            java_calls[("Apache 경유" if "/rag/llama/java" in path or xff not in ("-", "", "?") else "직접 호출") + f" {group_key(path)}"] += 1
        if kind == "admin" and xff not in ("-", "", "?"):
            admin_clients[xff.split(",")[-1].strip()] += 1   # Apache 가 맨 뒤에 붙인 실제 클라이언트 IP
        if kind in detail:
            key = (method, group_key(path))
            detail[kind][key] += 1
            notes[key] = (why, status)

    print(f"분석한 요청 {parsed}건 (형식이 달라 건너뜀 {skipped}건)\n")
    if not parsed:
        print("로그가 비어 있거나 형식이 다릅니다. `docker logs gateway 2>&1 | head` 로 확인하세요.")
        return
    labels = {"allow": "허용", "admin": "관리자 IP 전용", "block": "차단"}
    for k in ("allow", "admin", "block"):
        print(f"  {labels[k]:<14} {stats[k]:>7}건 ({stats[k] * 100 / parsed:.1f}%)")

    for k in ("admin", "block"):
        if detail[k]:
            print(f"\n[{labels[k]}] 경로 (많이 쓰인 순)")
            for (method, g), n in detail[k].most_common(25):
                why, status = notes[(method, g)]
                print(f"  {n:>6}건  {method:<6} {g}   ← {why} (기존 응답 {status})")

    if admin_clients:
        print("\n[관리자 경로를 사용한 클라이언트 IP] → admin_ips.inc 에 넣을 후보 (한 줄에 하나)")
        for ip, n in admin_clients.most_common(10):
            print(f"  {ip}  1;      # {n}건")

    if xff_seen:
        print("\n[질의(/rag/llama/ask) 요청의 X-Forwarded-For]")
        for k, n in xff_seen.most_common():
            print(f"  {k}: {n}건")
        if "없음('-')" in xff_seen:
            print("  ※ '없음' 이 있으면 Apache 가 클라이언트 IP 를 전달하지 않는 것입니다. 이 상태에서는 관리자 IP 판별이 안 되고")
            print("    모든 사용자가 한 IP 로 보여 속도 제한(초당 2회)을 함께 쓰게 됩니다. 적용 전에 Apache 설정을 확인하세요.")
    if java_calls:
        print("\n[자바(WAS)의 /llama/java POST 호출]")
        for k, n in java_calls.most_common():
            print(f"  {n:>6}건  {k}")
        if any(k.startswith("Apache") for k in java_calls):
            print("  ※ 'Apache 경유' 가 있으면 새 설정에서 막힙니다. 적용하지 말고 알려 주세요.")
    else:
        print("\n[자바(WAS)의 /llama/java POST 호출] 로그에 없음 → 자바 개발자에게 호출 주소를 확인하세요.")

    blocked = detail["block"]
    print("\n결론:", "차단되는 경로가 없습니다." if not blocked else
          f"차단되는 경로 {len(blocked)}종이 있습니다. 실제 화면·연계가 쓰는 경로인지 확인하기 전에는 적용하지 마세요.")


if __name__ == "__main__":
    main()
