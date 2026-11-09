import os
import sys

# 레포 루트를 import 경로에 추가 (app.* 모듈을 테스트에서 바로 import)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
