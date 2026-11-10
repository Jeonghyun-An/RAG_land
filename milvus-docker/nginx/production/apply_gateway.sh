#!/usr/bin/env bash
# ============================================================================
# Gateway(Nginx) 설정 적용 스크립트  —  서버에서 한 번 실행
#
#   하는 일: 백업 → app.conf 복사 → 관리자 IP 추가 → 관리자 경로 인증값 준비
#            → nginx -t 검사 → reload → 동작 확인. 어느 단계든 실패하면 자동으로 원복한다.
#   하지 않는 일: 컨테이너 재시작, 이미지·스택 변경 (Nginx 설정은 reload 만으로 적용된다)
#
# 사용법 (이 스크립트가 있는 폴더 = app.conf, real_ip.inc, admin_ips.inc, admin_gate.map 이 함께 있는 폴더)
#   sudo ./apply_gateway.sh -a 192.168.12.77            # 관리자 PC IP 를 넣고 적용
#   sudo ./apply_gateway.sh -a 192.168.12.77 -a 10.1.2.0/24
#   sudo ./apply_gateway.sh                             # IP 는 나중에 넣고 설정만 적용
#   sudo ./apply_gateway.sh --denied                    # 관리자 경로에서 403 을 받은 IP 후보 보기(내 IP 확인용)
#
# 옵션
#   -a IP      관리자 IP 추가(여러 번 가능, IPv4 또는 192.168.12.0/24 형식, /16 보다 넓은 대역은 거부)
#   -d DIR     서버의 conf.d 폴더            (기본 /data/nuclear-rag/nginx/conf.d)
#   -c NAME    gateway 컨테이너 이름         (기본 gateway)
#   -n NET     Docker 네트워크 이름          (기본 ragnet, real_ip.inc 점검용)
#   -s SECRET  관리자 경로 인증값을 직접 지정(영문·숫자·._- 16자 이상). 생략하면 기존 값을 유지하고,
#              기존 값이 없으면 새로 만든다.
#   --denied   관리자 경로에서 403 을 받은 접속 IP 를 보여 주고 끝낸다(변경 없음)
#
# 다시 실행해도 안전하다: 서버에 이미 있는 real_ip.inc·관리자 IP·인증값은 그대로 두고 app.conf 만 다시 복사한다.
# 적용 전에 real_ip.inc(X-Forwarded-For 를 믿는 출발지)를 점검해, 위조 위험이 있으면 아무것도 바꾸지 않고 멈춘다.
# ============================================================================
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_DIR="/data/nuclear-rag/nginx/conf.d"
CONTAINER="gateway"
NETWORK="ragnet"
SECRET_ARG=""
ONLY_DENIED=0
ADMIN_IPS=()
FILES=(app.conf real_ip.inc admin_ips.inc admin_gate.map)

say()  { printf '%s\n' "$*"; }
warn() { printf '[주의] %s\n' "$*" >&2; }
die()  { printf '[중단] %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,/^# =====.*$/p' "${BASH_SOURCE[0]}" | sed -n '2,$p' | sed 's/^# \{0,1\}//' | sed '$d'; }

# ---- 인자 --------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -a) [[ $# -ge 2 ]] || die "-a 뒤에 IP 가 필요합니다."
            IFS=',' read -r -a _ips <<< "$2"; ADMIN_IPS+=("${_ips[@]}"); shift 2 ;;
        -d) [[ $# -ge 2 ]] || die "-d 뒤에 폴더가 필요합니다."; CONF_DIR="${2%/}"; shift 2 ;;
        -c) [[ $# -ge 2 ]] || die "-c 뒤에 컨테이너 이름이 필요합니다."; CONTAINER="$2"; shift 2 ;;
        -n) [[ $# -ge 2 ]] || die "-n 뒤에 네트워크 이름이 필요합니다."; NETWORK="$2"; shift 2 ;;
        -s) [[ $# -ge 2 ]] || die "-s 뒤에 값이 필요합니다."; SECRET_ARG="$2"; shift 2 ;;
        --denied) ONLY_DENIED=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "알 수 없는 옵션: $1  (도움말: -h)" ;;
    esac
done

# ---- 관리자 IP 검증 (파일을 건드리기 전에 모두 검사) -----------------------------------
NORM_IPS=()
valid_ip() {   # 통과하면 정규화한 값을 NORM 에 담는다
    local v="$1" ip="${1%%/*}" pfx=""
    [[ "$v" == */* ]] && pfx="${v#*/}"
    [[ "$ip" =~ ^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$ ]] || return 1
    local a="${BASH_REMATCH[1]}" b="${BASH_REMATCH[2]}" c="${BASH_REMATCH[3]}" d="${BASH_REMATCH[4]}"
    if (( 10#$a > 255 || 10#$b > 255 || 10#$c > 255 || 10#$d > 255 )); then return 1; fi
    NORM="$((10#$a)).$((10#$b)).$((10#$c)).$((10#$d))"
    if [[ -n "$pfx" ]]; then
        [[ "$pfx" =~ ^[0-9]{1,2}$ ]] || return 1
        if (( 10#$pfx > 32 )); then return 1; fi
        if (( 10#$pfx < 16 )); then
            die "관리자 IP 대역이 너무 넓습니다: $v  (/16 이상으로 좁게 지정하세요)"
        fi
        NORM="$NORM/$((10#$pfx))"
    fi
    return 0
}
for raw in ${ADMIN_IPS[@]+"${ADMIN_IPS[@]}"}; do
    raw="${raw//[[:space:]]/}"
    [[ -n "$raw" ]] || continue
    NORM=""
    valid_ip "$raw" || die "IP 형식이 올바르지 않습니다: '$raw'  (예: 192.168.12.77 또는 192.168.12.0/24)"
    NORM_IPS+=("$NORM")
done

if [[ -n "$SECRET_ARG" ]]; then
    [[ "$SECRET_ARG" =~ ^[A-Za-z0-9._-]{16,}$ ]] || die "-s 값은 영문·숫자·._- 로 16자 이상이어야 합니다."
fi

# ---- 사전 점검 -----------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || die "docker 명령을 찾을 수 없습니다."
running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)"
[[ "$running" == "true" ]] || die "'$CONTAINER' 컨테이너가 실행 중이 아닙니다. (이름이 다르면 -c 로 지정)"

show_denied() {
    # 관리자 화면 주소(/rag/admin…)와 관리자 API 경로(app.conf 의 관리자 전용 규칙·업로드)
    local re='"[A-Z]+ /(rag/admin|(rag/)?llama/(finetune|admin|dev|debug|milvus|db|docs|openapi\.json|generate|test|files/purge|upload))(/[^" ]*)?" 403'
    local out
    out="$(docker logs --tail 5000 "$CONTAINER" 2>&1 | grep -E "$re" | awk '{print $1}' | sort | uniq -c | sort -rn | head -10 || true)"
    if [[ -z "$out" ]]; then
        say "  (최근 로그에 관리자 경로 403 이 없습니다. 관리 화면을 한 번 열어 본 뒤 다시 실행하세요.)"
        return 0
    fi
    say "  건수  접속 IP"
    say "$out" | sed 's/^/  /'
    if say "$out" | awk '{print $2}' | grep -Eq '^(172\.(1[6-9]|2[0-9]|3[01])\.|127\.)'; then
        warn "172.x / 127.x 는 Docker 내부 주소입니다. Apache 가 X-Forwarded-For 로 실제 IP 를 전달하지 않는 것일 수 있습니다."
        warn "이 주소를 관리자 IP 로 등록하면 모든 사용자가 관리자로 통과하니 등록하지 마세요. Apache 설정부터 확인하세요."
    fi
    say "  본인 IP 라면:  sudo ./apply_gateway.sh -a <IP>"
}

if (( ONLY_DENIED )); then
    say "관리자 경로(관리·파인튜닝 등)에서 403 을 받은 접속 IP — 최근 로그 기준"
    show_denied
    exit 0
fi

[[ -d "$CONF_DIR" ]] || die "conf.d 폴더가 없습니다: $CONF_DIR  (-d 로 지정)"
for f in "${FILES[@]}"; do
    [[ -f "$SRC_DIR/$f" ]] || die "$SRC_DIR/$f 가 없습니다. 다섯 파일(app.conf, real_ip.inc, admin_ips.inc, admin_gate.map, apply_gateway.sh)을 같은 폴더에 두세요."
done
[[ -w "$CONF_DIR" ]] || die "$CONF_DIR 에 쓸 수 없습니다. sudo 로 실행하세요."

# ---- real_ip.inc 점검 (파일을 바꾸기 전에) ------------------------------------------------
# X-Forwarded-For 를 믿는 출발지가 잘못되면 그 출발지에서 관리자 IP 를 위조할 수 있다(app.conf 머리말 "Apache 위치별 설정").
# 서버에 real_ip.inc 가 있으면 그 파일(보존됨)을, 없으면 이번에 복사할 파일을 본다.
realip_file="$CONF_DIR/real_ip.inc"
[[ -e "$realip_file" ]] || realip_file="$SRC_DIR/real_ip.inc"
other_confs=()
for f in "$CONF_DIR"/*.conf; do
    [[ -e "$f" && "$(basename "$f")" != "app.conf" ]] && other_confs+=("$f")
done
trusted="$(awk '$1=="set_real_ip_from" {v=$2; sub(/;.*/, "", v); print v}' \
            "$SRC_DIR/app.conf" "$realip_file" ${other_confs[@]+"${other_confs[@]}"} | sort -u)"

# 1) 대역 금지: IP 하나(/32, /128)만 허용
wide="$(printf '%s\n' "$trusted" | grep '/' | grep -vE '/(32|128)$' || true)"
[[ -z "$wide" ]] || die "set_real_ip_from 에 IP 대역이 있습니다: $(echo $wide)  ($realip_file 등)
       이 대역의 컨테이너·PC 가 X-Forwarded-For 를 위조하면 관리자 IP 로 판정됩니다. Apache 가 접속하는 IP 하나씩 적으세요."

# 2) ragnet 게이트웨이 IP 신뢰는 gateway 포트가 127.0.0.1 에만 열려 있을 때만
gw_ip="$(docker network inspect "$NETWORK" --format '{{range .IPAM.Config}}{{.Gateway}} {{end}}' 2>/dev/null | awk '{print $1}' || true)"
bindings="$(docker port "$CONTAINER" 80/tcp 2>/dev/null || true)"     # 예: 0.0.0.0:90 / [::]:90 / 127.0.0.1:90
lan_bound="$(printf '%s\n' "$bindings" | grep -v '^$' | grep -v '^127\.0\.0\.1:' || true)"
if [[ -z "$gw_ip" ]]; then
    warn "'$NETWORK' 네트워크를 찾지 못해 real_ip.inc 점검 일부를 건너뜁니다. (네트워크 이름이 다르면 -n 으로 지정)"
elif printf '%s\n' "$trusted" | grep -qxE "${gw_ip//./\.}(/32)?"; then
    [[ -z "$lan_bound" ]] || die "real_ip.inc 가 $NETWORK 게이트웨이 IP($gw_ip)를 믿는데 gateway 포트가 내부망에도 열려 있습니다: $(echo $lan_bound)
       이 상태에서는 다른 컨테이너 등이 이 IP 로 들어와 X-Forwarded-For 를 위조할 수 있습니다.
       스택의 gateway 포트를 \"127.0.0.1:90:80\" 으로 바꿔 재배포한 뒤 다시 실행하세요."
elif [[ -n "$bindings" && -z "$lan_bound" ]]; then
    warn "gateway 포트가 127.0.0.1 에만 열려 있는데 real_ip.inc 에 $NETWORK 게이트웨이 IP($gw_ip)가 없습니다."
    warn "같은 서버의 Apache 를 거친 요청이 모두 $gw_ip 하나로 보여 속도 제한을 함께 쓰고 관리자 판정이 되지 않습니다."
fi

# 3) 관리자 IP 로 등록하면 안 되는 주소
for ip in ${NORM_IPS[@]+"${NORM_IPS[@]}"}; do
    host="${ip%/*}"
    if [[ "$host" =~ ^(127\.|172\.(1[6-9]|2[0-9]|3[01])\.) ]] || printf '%s\n' "$trusted" | grep -qxF "$host"; then
        die "관리자 IP 로 등록할 수 없는 주소입니다: $ip  (Docker 내부 주소이거나 real_ip.inc 의 프록시 주소)"
    fi
done

# ---- 백업 ------------------------------------------------------------------------------
BACKUP_DIR="${CONF_DIR}.bak.$(date +%Y%m%d-%H%M%S)"   # 마운트 폴더 밖(옆)에 둔다 — conf.d 안의 *.conf 는 모두 읽히기 때문
cp -a "$CONF_DIR" "$BACKUP_DIR"
say "백업: $BACKUP_DIR"

NEW_FILES=()
for f in "${FILES[@]}"; do
    [[ -e "$CONF_DIR/$f" ]] || NEW_FILES+=("$f")
done

rollback() {
    warn "원래 설정으로 되돌립니다."
    local f
    for f in "${FILES[@]}"; do
        if [[ -e "$BACKUP_DIR/$f" ]]; then
            cp -a "$BACKUP_DIR/$f" "$CONF_DIR/$f"
        else
            rm -f "$CONF_DIR/$f"
        fi
    done
    if docker exec "$CONTAINER" nginx -t >/dev/null 2>&1 && docker exec "$CONTAINER" nginx -s reload >/dev/null 2>&1; then
        warn "원복 완료(백업: $BACKUP_DIR)."
    else
        warn "원복 후 nginx 검사/reload 가 실패했습니다. 직접 확인하세요: docker exec $CONTAINER nginx -t  (백업: $BACKUP_DIR)"
    fi
}
fail() { warn "$*"; rollback; exit 1; }

# ---- 파일 설치 --------------------------------------------------------------------------
install -m 644 "$SRC_DIR/app.conf" "$CONF_DIR/app.conf.new"
mv -f "$CONF_DIR/app.conf.new" "$CONF_DIR/app.conf"

[[ -e "$CONF_DIR/real_ip.inc" ]] || install -m 644 "$SRC_DIR/real_ip.inc" "$CONF_DIR/real_ip.inc"
[[ -e "$CONF_DIR/admin_ips.inc" ]] || install -m 644 "$SRC_DIR/admin_ips.inc" "$CONF_DIR/admin_ips.inc"
[[ -e "$CONF_DIR/admin_gate.map" ]] || install -m 600 "$SRC_DIR/admin_gate.map" "$CONF_DIR/admin_gate.map"

# 관리자 IP 추가 (이미 있으면 건너뜀)
ips_file="$CONF_DIR/admin_ips.inc"
for ip in ${NORM_IPS[@]+"${NORM_IPS[@]}"}; do
    if awk -v ip="$ip" '$1==ip && $2=="1;" {f=1} END{exit !f}' "$ips_file"; then
        say "관리자 IP 이미 있음: $ip"
    else
        if [[ -s "$ips_file" && -n "$(tail -c1 "$ips_file")" ]]; then echo >> "$ips_file"; fi
        printf '%-18s 1;   # 추가 %s\n' "$ip" "$(date +%F)" >> "$ips_file"
        say "관리자 IP 추가: $ip"
    fi
done

# 관리자 경로 인증값
map_file="$CONF_DIR/admin_gate.map"
current_secret="$(sed -nE 's/^[[:space:]]*1[[:space:]]+"([^"]+)";.*/\1/p' "$map_file" | head -1)"
secret="$current_secret"
if [[ -n "$SECRET_ARG" ]]; then
    secret="$SECRET_ARG"
elif [[ -z "$secret" ]]; then
    if command -v openssl >/dev/null 2>&1; then
        secret="$(openssl rand -hex 32)"
    else
        secret="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    fi
fi
if [[ "$secret" != "$current_secret" ]]; then
    ( umask 077
      { grep '^#' "$SRC_DIR/admin_gate.map" | grep -v '^# 1 "' || true; printf '1 "%s";\n' "$secret"; } > "$map_file.new"
    )
    mv -f "$map_file.new" "$map_file"
    [[ -n "$current_secret" ]] && say "관리자 경로 인증값을 바꿨습니다(스택 ADMIN_GATE_SECRET 도 같은 값으로 바꿔야 합니다)." \
                              || say "관리자 경로 인증값을 새로 만들었습니다."
else
    say "관리자 경로 인증값: 기존 값 유지"
fi
chmod 600 "$map_file"

# ---- 검사 → reload → 확인 ---------------------------------------------------------------
say "nginx 설정 검사..."
if ! test_out="$(docker exec "$CONTAINER" nginx -t 2>&1)"; then
    say "$test_out" >&2
    fail "nginx -t 실패 — 적용하지 않았습니다."
fi
say "$test_out" | sed 's/^/  /'

docker exec "$CONTAINER" nginx -s reload || fail "nginx reload 실패."
sleep 2

hz="$(docker exec "$CONTAINER" wget -q -O- http://127.0.0.1/healthz 2>/dev/null || true)"
[[ "$hz" == "ok" ]] || fail "reload 후 /healthz 응답이 없습니다."
if docker logs --since 10s "$CONTAINER" 2>&1 | grep -Eq '\[(emerg|alert|crit)\]'; then
    docker logs --since 10s "$CONTAINER" 2>&1 | grep -E '\[(emerg|alert|crit)\]' >&2 || true
    fail "reload 직후 gateway 로그에 오류가 있습니다."
fi
say "reload 완료, /healthz 정상"

echo
say "================ 적용 결과 ================"
say "app.conf        : $CONF_DIR/app.conf"
say "X-Forwarded-For 를 믿는 출발지(real_ip.inc 등): $(echo $trusted)"
say "관리자 IP        :"
active_ips="$(awk '$1 !~ /^#/ && $2=="1;" {print "  " $1}' "$ips_file")"
if [[ -n "$active_ips" ]]; then
    say "$active_ips"
else
    warn "등록된 관리자 IP 가 없어 모든 PC 에서 관리·파인튜닝 화면이 403 입니다.  →  sudo ./apply_gateway.sh -a <내 PC IP>"
fi
say "관리자 경로 인증값(ADMIN_GATE_SECRET): $secret"
say "  ↑ Portainer 스택의 환경변수 ADMIN_GATE_SECRET 에 같은 값을 넣어야 합니다(새 이미지 적용 단계에서 사용)."
say "    다시 보려면:  sudo cat $map_file"
say "백업            : $BACKUP_DIR"
say "되돌리기        : sudo cp -a $BACKUP_DIR/. $CONF_DIR/ && docker exec $CONTAINER nginx -s reload"
echo
say "내 IP 를 모르겠다면: 관리 화면(모델 관리 등)을 한 번 열어 403 을 받은 뒤  sudo ./apply_gateway.sh --denied"
