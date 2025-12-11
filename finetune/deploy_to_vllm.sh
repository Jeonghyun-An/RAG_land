#!/bin/bash
"""
vLLM 서버에 파인튜닝 모델 배포 스크립트
finetune/deploy_to_vllm.sh

사용법:
    bash finetune/deploy_to_vllm.sh [OPTIONS]
    
옵션:
    --use-lora          LoRA 어댑터 방식으로 배포 (기본값)
    --use-merged        병합된 모델로 배포
    --port 28080        vLLM 서버 포트 (기본: 28080)
    
예시:
    # LoRA 어댑터 방식 (메모리 효율적)
    bash finetune/deploy_to_vllm.sh --use-lora
    
    # 병합 모델 방식
    bash finetune/deploy_to_vllm.sh --use-merged
"""

set -e  # 에러 발생 시 중단

# ==================== 기본 설정 ====================
DEPLOYMENT_MODE="lora"  # lora or merged
VLLM_PORT=28080
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
LORA_PATH="/workspace/output/qwen2.5-7b-nuclear-lora"
MERGED_PATH="/workspace/output/qwen2.5-7b-nuclear-merged"

# ==================== 인자 파싱 ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-lora)
            DEPLOYMENT_MODE="lora"
            shift
            ;;
        --use-merged)
            DEPLOYMENT_MODE="merged"
            shift
            ;;
        --port)
            VLLM_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "vLLM 서버 배포"
echo "========================================"
echo "배포 모드: $DEPLOYMENT_MODE"
echo "포트: $VLLM_PORT"
echo "========================================"

# ==================== LoRA 방식 배포 ====================
if [ "$DEPLOYMENT_MODE" == "lora" ]; then
    echo ""
    echo "📦 LoRA 어댑터 방식 배포"
    echo ""
    
    # LoRA 어댑터 존재 확인
    if [ ! -f "$LORA_PATH/adapter_config.json" ]; then
        echo " LoRA 어댑터를 찾을 수 없습니다: $LORA_PATH"
        echo "   먼저 train_qlora.py를 실행하세요!"
        exit 1
    fi
    
    echo " LoRA 어댑터 확인됨: $LORA_PATH"
    
    # docker-compose 파일 생성
    cat > docker-compose.vllm-finetuned.yml << EOF
version: "3.9"

services:
  vllm-finetuned:
    image: vllm/vllm-openai:latest
    container_name: nuclear-vllm-finetuned
    
    volumes:
      - finetune-output:/workspace/lora_adapters:ro
      - /data/models/.hf-cache:/root/.cache/huggingface:rw
    
    environment:
      CUDA_VISIBLE_DEVICES: "0"
      NVIDIA_VISIBLE_DEVICES: "all"
      NVIDIA_DRIVER_CAPABILITIES: "compute,utility"
      HF_TOKEN: "\${HUGGINGFACE_TOKEN}"
    
    command:
      - --model
      - "$BASE_MODEL"
      - --enable-lora
      - --lora-modules
      - nuclear-lora=/workspace/lora_adapters/qwen2.5-7b-nuclear-lora
      - --max-lora-rank
      - "64"
      - --port
      - "8000"
      - --host
      - "0.0.0.0"
      - --dtype
      - bfloat16
      - --max-model-len
      - "4096"
      - --gpu-memory-utilization
      - "0.9"
      - --trust-remote-code
    
    ports:
      - "$VLLM_PORT:8000"
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: ["gpu"]
    
    networks:
      - ragnet
    
    restart: unless-stopped

volumes:
  finetune-output:
    external: true
    name: nuclear-finetune-output

networks:
  ragnet:
    external: true
    name: ragnet
EOF

    echo ""
    echo " docker-compose.vllm-finetuned.yml 생성됨"
    echo ""
    echo " vLLM 서버 시작 중..."
    
    docker-compose -f docker-compose.vllm-finetuned.yml up -d
    
    echo ""
    echo "⏳ 서버 초기화 대기 중 (60초)..."
    sleep 60
    
    # 헬스 체크
    echo ""
    echo "🔍 서버 상태 확인..."
    
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo " vLLM 서버 정상 작동 중"
        echo ""
        echo " API 엔드포인트:"
        echo "   http://localhost:$VLLM_PORT/v1/completions"
        echo "   http://localhost:$VLLM_PORT/v1/chat/completions"
        echo ""
        echo " 테스트 명령어:"
        echo "   curl http://localhost:$VLLM_PORT/v1/models"
    else
        echo "⚠️  서버가 아직 준비되지 않았습니다"
        echo "   docker logs nuclear-vllm-finetuned 로 상태 확인"
    fi

# ==================== 병합 모델 방식 배포 ====================
elif [ "$DEPLOYMENT_MODE" == "merged" ]; then
    echo ""
    echo "📦 병합 모델 방식 배포"
    echo ""
    
    # 병합 모델 존재 확인
    if [ ! -f "$MERGED_PATH/config.json" ]; then
        echo " 병합 모델을 찾을 수 없습니다: $MERGED_PATH"
        echo "   먼저 merge_model.py를 실행하세요!"
        exit 1
    fi
    
    echo " 병합 모델 확인됨: $MERGED_PATH"
    
    # docker-compose 파일 생성
    cat > docker-compose.vllm-finetuned.yml << EOF
version: "3.9"

services:
  vllm-finetuned:
    image: vllm/vllm-openai:latest
    container_name: nuclear-vllm-finetuned
    
    volumes:
      - finetune-output:/workspace/models:ro
      - /data/models/.hf-cache:/root/.cache/huggingface:rw
    
    environment:
      CUDA_VISIBLE_DEVICES: "0"
      NVIDIA_VISIBLE_DEVICES: "all"
      NVIDIA_DRIVER_CAPABILITIES: "compute,utility"
      HF_TOKEN: "\${HUGGINGFACE_TOKEN}"
    
    command:
      - --model
      - "/workspace/models/qwen2.5-7b-nuclear-merged"
      - --port
      - "8000"
      - --host
      - "0.0.0.0"
      - --dtype
      - bfloat16
      - --max-model-len
      - "4096"
      - --gpu-memory-utilization
      - "0.9"
      - --trust-remote-code
    
    ports:
      - "$VLLM_PORT:8000"
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: ["gpu"]
    
    networks:
      - ragnet
    
    restart: unless-stopped

volumes:
  finetune-output:
    external: true
    name: nuclear-finetune-output

networks:
  ragnet:
    external: true
    name: ragnet
EOF

    echo ""
    echo " docker-compose.vllm-finetuned.yml 생성됨"
    echo ""
    echo " vLLM 서버 시작 중..."
    
    docker-compose -f docker-compose.vllm-finetuned.yml up -d
    
    echo ""
    echo " 서버 초기화 대기 중 (60초)..."
    sleep 60
    
    # 헬스 체크
    echo ""
    echo " 서버 상태 확인..."
    
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo " vLLM 서버 정상 작동 중"
        echo ""
        echo " API 엔드포인트:"
        echo "   http://localhost:$VLLM_PORT/v1/completions"
        echo "   http://localhost:$VLLM_PORT/v1/chat/completions"
        echo ""
        echo " 테스트 명령어:"
        echo "   curl http://localhost:$VLLM_PORT/v1/models"
    else
        echo "  서버가 아직 준비되지 않았습니다"
        echo "   docker logs nuclear-vllm-finetuned 로 상태 확인"
    fi
fi

echo ""
echo "========================================"
echo " 배포 완료!"
echo "========================================"
echo ""
echo "다음 단계:"
echo "1. API 테스트:"
echo "   python finetune/test_api.py --url http://localhost:$VLLM_PORT"
echo ""
echo "2. 성능 비교:"
echo "   python finetune/compare_models.py \\"
echo "     --base-url http://localhost:18080 \\"
echo "     --finetuned-url http://localhost:$VLLM_PORT"
echo ""
echo "3. 운영 환경 업데이트:"
echo "   - nuclearchat의 VLLM_BASE_URL을 http://nuclear-vllm-finetuned:8000으로 변경"
echo "   - docker-compose restart llama"
echo ""
echo "========================================"