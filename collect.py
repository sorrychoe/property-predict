"""연립다세대 매매 실거래 수집·정규화 -> data/trades.csv, data/collect_log.json.

python collect.py            # 기준일 이전 5년치(대상 3개 시군구) 수집
원본 응답은 data/cache/에 저장되며 재실행 시 재사용한다(최신 자료로 갱신하려면 캐시 삭제).
좌표(카카오 로컬)는 저장이 금지되어 수집하지 않고 predict·evaluate·diagnose 실행 중 실시간으로 조회한다.
"""
import argparse
import json
import logging
import math
import os
import sys
from datetime import date, datetime

import pandas as pd

import api
from model import BASE_DATE, DATA, DONG_ONLY, LAWD, YEARS_FIT, jibun_key, parse_jibun

log = logging.getLogger('collect')
COLS = ['sigungu', 'dong', 'jibun', 'name', 'house_type', 'area_m2', 'land_m2', 'floor', 'build_year',
        'deal_date', 'price', 'dealing', 'rgst_date']


def months(start, end):
    """start가 속한 달부터 end가 속한 달까지 'YYYYMM' 문자열을 차례로 생성한다."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield f'{y}{m:02d}'
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def norm(it, sigungu):
    """API item -> 정규화 행. 해석 불가 시 (None, 사유)."""
    jb = parse_jibun(it.get('jibun') or '')
    if not jb:
        return None, '지번 없음/형식 오류'
    if (it.get('cdealType') or '').strip() or (it.get('cdealDay') or '').strip():
        return None, '해제 거래'
    try:
        price = int(it['dealAmount'].replace(',', '').strip()) * 10000  # 원본 단위 만원
        d = date(int(it['dealYear']), int(it['dealMonth']), int(it['dealDay']))
        area = float(it['excluUseAr'])
        floor = int(it['floor'])
    except (KeyError, ValueError):
        return None, '필수값(금액·일자·면적·층) 해석 불가'
    by = (it.get('buildYear') or '').strip()
    return {
        'sigungu': sigungu, 'dong': it.get('umdNm', '').strip(), 'jibun': jibun_key(*jb),
        'name': (it.get('mhouseNm') or '').strip(), 'house_type': (it.get('houseType') or '').strip(),
        'area_m2': area, 'land_m2': (it.get('landAr') or '').strip(), 'floor': floor,
        'build_year': int(by) if by.isdigit() and 1900 < int(by) <= d.year + 1 else '',
        'deal_date': d.isoformat(), 'price': price, 'dealing': (it.get('dealingGbn') or '').strip(),
        'rgst_date': (it.get('rgstDate') or '').strip(),
    }, None


def clean(rows, excluded):
    """기간·권역·중복·이상치 처리. 제외 사유별 건수를 excluded에 누적."""
    df = pd.DataFrame(rows, columns=COLS)
    def drop(mask, why):
        """mask에 해당하는 행을 제외하고 제외 건수를 사유별로 누적한다."""
        excluded[why] = excluded.get(why, 0) + int(mask.sum())
        return df[~mask]
    df = drop(pd.to_datetime(df.deal_date).dt.date >= BASE_DATE, f'기준일({BASE_DATE}) 이후 계약')
    df = drop(df.apply(lambda r: r.dong not in DONG_ONLY.get(r.sigungu, {r.dong}), axis=1), '대상 법정동 외(강서구 화곡동 외)')
    df = drop(df.duplicated(['sigungu', 'dong', 'jibun', 'floor', 'area_m2', 'deal_date', 'price']), '중복 거래')
    df = drop((df.area_m2 < 10) | (df.area_m2 > 300), '전용면적 10㎡ 미만 또는 300㎡ 초과')
    df = drop(df.build_year == '', '건축년도 없음')
    # 이상치: 시군구별 log(㎡당 가격)이 중앙값에서 robust 표준편차 5배 이상 벗어난 거래(입력 오류·특수 거래 추정)
    lp = (df.price / df.area_m2).map(math.log)
    med = lp.groupby(df.sigungu).transform('median')
    mad = (lp - med).abs().groupby(df.sigungu).transform('median') * 1.4826
    df = drop((lp - med).abs() > 5 * mad, '㎡당 가격 극단치(시군구 중앙값±5×robust SD 밖)')
    return df.sort_values(['sigungu', 'dong', 'deal_date', 'jibun', 'floor']).reset_index(drop=True)


def main():
    """대상 3개 시군구의 월별 실거래를 수집·정규화해 trades.csv와 수집 기록을 저장한다."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--years', type=int, default=YEARS_FIT)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stderr)
    start = date(BASE_DATE.year - a.years, BASE_DATE.month, 1)
    rows, excluded, raw = [], {}, {}
    for sg, lawd in LAWD.items():
        n = 0
        for ym in months(start, BASE_DATE):
            items = api.rh_trades(lawd, ym)
            n += len(items)
            for it in items:
                r, why = norm(it, sg)
                if r:
                    rows.append(r)
                else:
                    excluded[why] = excluded.get(why, 0) + 1
            log.info('%s %s: %d건', sg, ym, len(items))
        raw[sg] = n
    df = clean(rows, excluded)
    tmp = DATA / 'trades.csv.tmp'
    df.to_csv(tmp, index=False, encoding='utf-8')
    os.replace(tmp, DATA / 'trades.csv')
    meta = {
        'source': '국토교통부_연립다세대 매매 실거래가 자료(data.go.kr 15126467, RTMSDataSvcRHTrade)',
        'collected_at': datetime.now().isoformat(timespec='seconds'),
        'period': f'{start} ~ {BASE_DATE} 미만(계약일 기준)', 'base_date': str(BASE_DATE),
        'note': '기준일 이전 계약이라도 수집 시점까지 신고·공개된 건만 포함(신고기한 30일)',
        'raw_items_by_sigungu': raw, 'excluded': excluded, 'kept': len(df),
        'kept_by_sigungu': df.sigungu.value_counts().to_dict(),
        'kept_by_house_type': df.house_type.replace('', '(필드 없음)').value_counts().to_dict(),
    }
    (DATA / 'collect_log.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    log.info('저장 %d건, 제외 %s', len(df), excluded)


if __name__ == '__main__':
    try:
        main()
    except api.ApiError as e:
        sys.exit(f'수집 실패: {e}')
