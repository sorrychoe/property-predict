"""다세대주택 매매 시세 예측 CLI.

python predict.py --input input.csv --output output.csv
"""
import argparse
import csv
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import api
from model import BASE_DATE, load_model, validate

IN_COLS = ['id', 'sigungu', 'dong', 'jibun', 'floor', 'ho', 'area_m2']
OUT_COLS = ['id', 'price_est', 'price_low', 'price_high', 'confidence', 'basis', 'status']
log = logging.getLogger('predict')


def read_input(path):
    """입력 CSV(UTF-8, BOM 허용)를 문자열 dict 목록으로 읽는다. 필수 헤더가 없으면 ValueError."""
    with open(path, encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        missing = [c for c in IN_COLS if c not in (r.fieldnames or [])]
        if missing:
            raise ValueError(f'필수 헤더 누락: {", ".join(missing)}')
        return list(r)


def predict_rows(rows, model, lookup=api):
    """행마다 독립 처리. 한 행의 실패가 다른 행에 영향을 주지 않는다."""
    out = []
    for row in rows:
        rid = row.get('id') or ''
        try:
            if not rid.strip():
                raise ValueError('id 없음')
            p, err = validate(row, model.codes)
            if err:
                raise ValueError(err)
            e = model.estimate(p, lookup)
            out.append({'id': rid, **{k: e[k] for k in OUT_COLS[1:6]}, 'status': 'ok'})
            out[-1]['basis'] = ' '.join(out[-1]['basis'].split())  # 근거는 한 줄(외부 오류 메시지의 줄바꿈 제거)
        except Exception as ex:  # 행 단위 실패 사유를 남기고 계속
            log.warning('id=%s 실패: %s', rid, ex)
            out.append({'id': rid, 'price_est': '', 'price_low': '', 'price_high': '', 'confidence': '',
                        'basis': ' '.join(f'실패: {ex}'.split()), 'status': 'fail'})
    return out


def write_output(path, rows):
    """임시 파일에 쓴 뒤 교체하여 실패 시 기존 출력 파일을 보존한다."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent if str(path.parent) else '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, OUT_COLS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def main(argv=None):
    """CLI 진입점. 입력 오류는 종료 코드 2, 정상 처리 시 0(개별 행 실패는 출력 CSV에 기록)."""
    ap = argparse.ArgumentParser(description='다세대주택 매매 시세 예측')
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stderr)
    t0 = time.monotonic()
    api.DEADLINE = t0 + 20 * 60
    api.KAKAO_DEADLINE = t0 + 5 * 60
    api._fails.clear()
    try:
        rows = read_input(a.input)
        model = load_model(api=api)
    except (OSError, ValueError, UnicodeDecodeError) as e:
        log.error('%s', e)
        return 2
    if 'edges' not in model.calib:
        log.warning('data/calibration.json 없음: 범위·신뢰도가 보정되지 않은 기본값으로 계산됨(evaluate.py 실행 필요)')
    if not os.environ.get('KAKAO_API_KEY'):
        log.warning('KAKAO_API_KEY 미설정: 좌표 미사용(인근 사례는 본번 근접, 지하철역 거리 제외). 보정값은 좌표 사용 기준')
    if not os.environ.get('BLD_API_KEY'):
        log.warning('BLD_API_KEY 미설정: 캐시에 없는 건축물대장 조회는 실패 처리 후 대체 추정')
    log.info('기준일 %s, 학습 거래 %d건, 입력 %d건', BASE_DATE, model.n_fit, len(rows))
    out = predict_rows(rows, model)
    write_output(a.output, out)
    ok = sum(r['status'] == 'ok' for r in out)
    log.info('완료: 성공 %d / 실패 %d, %.1f초', ok, len(out) - ok, time.monotonic() - t0)
    return 0


if __name__ == '__main__':
    sys.exit(main())
