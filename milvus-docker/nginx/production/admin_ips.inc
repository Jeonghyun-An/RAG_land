# 관리자(연구 담당) PC 의 IP 또는 대역 — 이 파일만 고친 뒤
#   docker exec gateway nginx -t && docker exec gateway nginx -s reload
# 형식: "IP  1;" (한 줄에 하나). 주석뿐이면 관리·파인튜닝 화면은 모두 403 이다.
#
# 192.168.12.100   1;     # 예: 연구 담당자 PC
# 192.168.12.0/24  1;     # 예: 대역 전체
