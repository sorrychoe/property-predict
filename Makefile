PYTHON ?= python3
IN ?= input.csv
OUT ?= output.csv

.PHONY: help install collect diagnose evaluate predict test clean

help:  ## 명령 목록
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

install:  ## 의존성 설치
	$(PYTHON) -m pip install -r requirements.txt

collect:  ## 실거래 수집·정규화(RTMS_API_KEY, data/cache 재사용)
	$(PYTHON) collect.py

diagnose:  ## 회귀 진단·후보 변수 비교 -> reports/model_diagnosis.md
	$(PYTHON) diagnose.py

evaluate:  ## 홀드아웃 평가·보정 -> data/calibration.json, reports/eval_*
	$(PYTHON) evaluate.py --api-sample 30

predict:  ## 시세 예측: make predict IN=input.csv OUT=output.csv
	$(PYTHON) predict.py --input $(IN) --output $(OUT)

test:  ## 외부 API 없는 테스트
	$(PYTHON) test_predict.py

clean:  ## 파이썬 캐시·임시 파일 삭제(수집 데이터 data/cache는 유지)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . \( -name '*.pyc' -o -name '*.tmp' \) -type f -delete
