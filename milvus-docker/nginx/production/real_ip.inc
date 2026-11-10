# Gateway 가 X-Forwarded-For 를 믿는 출발지 (app.conf ★1)
# 서버마다 값이 달라 app.conf 와 분리했다. apply_gateway.sh 는 이 파일이 서버에 없을 때만 복사하고,
# 있으면 건드리지 않는다(admin_ips.inc 와 같다). 서버 conf.d 의 이 파일을 고친 뒤:  sudo ./apply_gateway.sh
#
# 적는 것
#   - Apache(도메인)가 gateway 에 접속하는 IP
#   - Apache 앞에 X-Forwarded-For 를 붙이는 WAF·L7 장비가 더 있으면 그 IP
#     (공유기·NAT 처럼 X-Forwarded-For 를 붙이지 않는 장비는 넣지 않는다 — 넣으면 그 뒤의 PC 가 위조할 수 있다)
# 지킬 것
#   - IP 하나씩 적는다. 대역(172.16.0.0/12 등)을 적으면 그 대역의 컨테이너·PC 가 X-Forwarded-For 를 위조해
#     관리자 IP 로 판정받을 수 있다. 대역이 있으면 apply_gateway.sh 가 적용을 멈춘다.
#   - 여기 적은 IP 를 admin_ips.inc(관리자 IP)에 넣지 않는다.
#
# Apache 위치에 따른 값 (app.conf 머리말 "Apache 위치별 설정")
#   Apache 가 다른 서버 : 그 서버 IP
#   Apache 가 같은 서버 : ragnet 게이트웨이 IP(172.x.0.1) — gateway 포트를 127.0.0.1:90:80 으로 묶은 뒤에만

set_real_ip_from 192.168.12.55;     # Apache(도메인) 서버 — 수행기관 시범 환경
