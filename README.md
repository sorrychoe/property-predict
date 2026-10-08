# property-predict

서울 강서구 화곡동·강남구·관악구의 다세대주택 매매 시세를 추정하는 명령행 프로그램입니다.
지번·층·호를 넣으면 추정 시세, 가격 범위, 신뢰도, 산출 근거를 CSV로 돌려줍니다.

## 설치

Python 3.10 이상.

```bash
pip install -r requirements.txt    # 또는 make install
cp .env.example .env               # 외부 조회를 쓰려면 키 입력(아래 참고)
```

### API 키(선택)

키가 없어도 저장소에 포함된 수집 자료로 동작합니다. 키가 있으면 실행 중에 건축물대장(면적·준공 연도)과 카카오 지도(건물·지하철역 위치)를 조회합니다.

1. [공공데이터포털](https://www.data.go.kr)에서 아래 두 API를 각각 활용신청합니다(개발계정 일 10,000건).
   - 국토교통부_연립다세대 매매 실거래가 자료 (15126467) — 데이터 수집용
   - 국토교통부_건축HUB_건축물대장정보 서비스 (15134735) — 예측 중 조회용
2. 마이페이지의 일반 인증키(Decoding 또는 Encoding)를 `.env`의 `RTMS_API_KEY`, `BLD_API_KEY`에 넣습니다.
   인증키가 하나라면 두 변수에 같은 값을 넣습니다.
3. [Kakao Developers](https://developers.kakao.com) 앱에서 Web 플랫폼 도메인을 등록하고 카카오맵(로컬) 사용 설정을 켠 뒤,
   **REST API 키**를 `KAKAO_API_KEY`에 넣습니다(JavaScript 키는 동작하지 않음).

- 셸 환경변수가 `.env`보다 우선합니다. `.env`는 `.gitignore`에 포함되어 있으며 키는 로그·출력에 남지 않습니다.
- 카카오 응답은 저장이 금지되어 실행할 때마다 위치를 다시 조회합니다(1회 약 7,600건 호출). 앱의 일일 호출 한도를 확인하세요.

## 실행

```bash
python predict.py --input input.csv --output output.csv
```

입력 예시(UTF-8, BOM 허용):

```csv
id,sigungu,dong,jibun,floor,ho,area_m2
1,서울특별시 관악구,신림동,1465-1,4,402,44.46
2,관악구,봉천동,1529-30,B1,,
```

| 입력 열 | 설명 |
| --- | --- |
| `sigungu` | `서울특별시 관악구`, `서울 관악구`, `관악구` 모두 허용. 강서구는 화곡동만 |
| `jibun` | `123-45`, `123`, `산12-3` |
| `floor` | 정수. 지하는 `-1` 또는 `B1`(0층은 오류) |
| `ho`, `area_m2` | 비워도 됨. 면적 단위 ㎡ |

출력 열 `id,price_est,price_low,price_high,confidence,basis,status`

| 출력 열 | 설명 |
| --- | --- |
| `price_est`, `price_low`, `price_high` | 추정 시세와 범위(원, 만원 단위 반올림) |
| `confidence` | 0~1. 높을수록 믿을 만함. 거래 자료가 없는 법정동은 미검증으로 0 |
| `basis` | 계산 근거 한 줄(사용한 사례 수, 보정, 빠진 정보를 채운 출처) |
| `status` | `ok` 또는 `fail`. 실패 시 `basis`에 사유, 가격·신뢰도는 빈 값 |

- 입력 행 순서와 `id` 원문을 그대로 유지합니다. 한 행의 오류나 조회 실패는 다른 행에 영향을 주지 않습니다.
- 입력 파일을 읽을 수 없거나 필수 헤더가 없으면 종료 코드 2로 끝나며 기존 출력 파일은 바뀌지 않습니다.
- 로그는 표준 오류로 출력됩니다. 외부 조회는 카카오 5분, 전체 20분까지만 기다리므로 20건은 30분 안에 끝납니다.

## 기타 명령

```bash
make help        # 전체 명령 목록
make test        # 테스트(외부 API 없음)
make collect     # 실거래 다시 수집(RTMS_API_KEY 필요) → data/trades.csv, data/collect_log.json
make evaluate    # 자체 검증·보정 → data/calibration.json, reports/eval_*(로컬)
make diagnose    # 시세 요인 진단 → reports/model_diagnosis.md(로컬)
make clean       # __pycache__·*.pyc·*.tmp 삭제(수집 데이터는 유지)
```

저장소에 정제된 거래 자료와 보정값이 들어 있어 키 없이 바로 `predict`를 실행할 수 있습니다.
`evaluate`는 `data/calibration.json`을 다시 만듭니다. 저장된 보정값은 카카오 위치 정보를 쓴 결과이므로, 같은 값을 재현하려면 `KAKAO_API_KEY`가 필요합니다.
`data/cache/`가 없는 상태에서 `collect`를 실행하면 실거래를 새로 받으므로 그 사이 신고된 거래가 추가되어 결과가 조금 달라질 수 있습니다.

## 폴더 구조

```text
Makefile        실행·정리 명령
predict.py      예측 CLI
model.py        정규화, 빠진 정보 보완, 시세 추정
api.py          공공데이터포털·카카오 호출
collect.py      실거래 수집·정리
evaluate.py     자체 검증·보정
diagnose.py     시세 요인 진단
test_predict.py 테스트
data/           법정동 코드, 수집 거래, 수집 기록, 보정값
reports/        예시 3건·20건 실행 입출력
```

## 사용 데이터

| 데이터 | 출처 | 이용조건 |
| --- | --- | --- |
| 연립다세대 매매 실거래(2021-10 ~ 2026-10-05 계약, 20,920건) | 국토교통부, 공공데이터포털 15126467 | 이용허락범위 제한 없음 |
| 건축물대장(표제부, 전유공용면적) | 국토교통부 건축HUB, 공공데이터포털 15134735 | 이용허락범위 제한 없음 |
| 법정동 코드 | 국토교통부_전국 법정동, 공공데이터포털 15063424(2026-07-29 파일) | 이용허락범위 제한 없음 |
| 건물·지하철역 위치 | 카카오 로컬 API | 응답 저장 금지 → 실행 중 조회만, 저장소에 포함하지 않음 |

## License

Copyright 2026 Jaeseong Choe

Licensed under the [Apache License, Version 2.0](LICENSE). 데이터 파일은 각 제공처의 이용조건을 따릅니다.
