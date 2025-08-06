# app/config.py
from dotenv import load_dotenv
import os

# .env 파일 불러오기
load_dotenv()

# 환경변수 읽기
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
