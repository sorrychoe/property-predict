"""홀드아웃 평가와 범위·신뢰도 보정. predict.py와 같은 Model.estimate를 사용한다.

python evaluate.py [--api-sample N]

분리 방식
- 평가 대상: 기준일 직전 12개월 계약 거래. 건물(시군구·법정동·지번) 해시로 10%는 보정(calib), 10%는 최종 평가(test).
- 홀드아웃 거래는 회귀 적합과 비교사례에서 모두 제외한다. 같은 건물의 과거 거래는 실제 운영과 같이 사용한다
  (변형 평가 unseen_building은 test 건물의 모든 거래를 제외).
- 각 거래는 계약월 1일을 기준일로 그 이전 자료만으로 다시 적합한 모델로 예측한다(미래 정보 차단).
- calib에서 파라미터(k_b, m, k_n, near)·범위 배수·신뢰도 표를 정하고 test는 결과 보고에만 사용한다.
"""
import argparse
import hashlib
import itertools
import json
import logging
import math
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import api
from model import BASE_DATE, DATA, HIT, Model, fetch_locations, load_trades

REPORTS = DATA.parent / 'reports'
GRID = {'k_b': [0.5, 1, 2, 4], 'm': [5, 10, 20], 'k_n': [1, 3, 10], 'near': ['bun', 'dist']}  # dist는 KAKAO_API_KEY 필요
COVER = 0.8  # 가격 범위 목표 포함률(보정 표본 기준)
log = logging.getLogger('evaluate')


def split(trades):
    """기준일 직전 12개월 거래에 건물 해시로 역할(calib 10%, test 10%, 나머지 pool)을 붙인다."""
    t = trades.copy()
    t['bkey'] = t.sigungu + ' ' + t.dong + ' ' + t.jibun
    h = t.bkey.map(lambda k: int(hashlib.sha1(k.encode()).hexdigest(), 16) % 10)
    recent = t.deal_date >= BASE_DATE - timedelta(days=365)
    if (t.house_type != '').any():
        recent &= t.house_type == '다세대'
    t['role'] = np.where(recent & (h == 0), 'calib', np.where(recent & (h == 1), 'test', 'pool'))
    return t


class TradeRegister:
    """건축년도를 조회할 수 있다고 가정한 모델 평가용 대역(가격은 제공하지 않음).
    실제 API 성공·물건 매칭까지 검증하는 평가가 아니다. 세대 면적은 제공하지 않는다."""
    def __init__(self, build_year):
        """build_year: 해당 거래의 건축년도."""
        self.by = build_year

    def building_title(self, code, san, bun, ji):
        """표제부 대역: 사용승인일만 채운다."""
        return [{'useAprDay': f'{self.by}0101'}]

    def building_units(self, code, san, bun, ji):
        """전유공용면적 대역: 빈 목록."""
        return []


def run(models, rows, params, drop_area=False, lookup='stub'):
    """거래 행들을 입력 물건으로 바꿔 계약월 모델로 추정하고 실제가·추정가·APE를 행별로 반환한다(실패 행 포함)."""
    out = []
    for r in rows.itertuples():
        m = models[(r.deal_date.year, r.deal_date.month)]
        m.params = params
        p = {'sigungu': r.sigungu, 'dong': r.dong, 'san': r.san, 'bun': r.bun,
             'ji': int(r.jibun.split('-')[1]) if '-' in r.jibun else 0, 'jibun': r.jibun,
             'floor': int(r.floor), 'ho': '', 'area': None if drop_area else float(r.area_m2)}
        rec = {'sigungu': r.sigungu, 'dong': r.dong, 'jibun': r.jibun, 'floor': r.floor, 'area_m2': r.area_m2,
               'build_year': r.build_year, 'deal_date': r.deal_date, 'dealing': r.dealing, 'actual': r.price}
        try:
            e = m.estimate(p, TradeRegister(r.build_year) if lookup == 'stub' else lookup)
            rec.update({k: e[k] for k in ('price_est', 'price_low', 'price_high', 'confidence', 'sigma',
                                         'area_used', 'area_src', 'by_imputed', 'n_bldg', 'n_near', 'basis')}, status='ok')
        except Exception as ex:
            rec.update(status='fail', basis=f'실패: {ex}')
        out.append(rec)
    df = pd.DataFrame(out)
    if 'price_est' not in df:
        df['price_est'] = np.nan
    df['ape'] = (df.price_est - df.actual).abs() / df.actual * 100
    return df


def models_for(pool, months, calib=None, loc=({}, None)):
    """각 (연, 월)에 대해 그 달 1일을 기준일로 pool만 사용해 적합한 Model을 만든다. loc: fetch_locations() 결과."""
    return {(y, mo): Model(pool, date(y, mo, 1), calib=calib, geo=loc[0], stations=loc[1]) for y, mo in months}


def summarize(df):
    """건수, 실패 수·실패율, MdAPE, MAPE, 10%·20% 이내 비율, 가격 범위 포함률을 계산한다."""
    ok = df[df.status == 'ok']
    return {'n': len(df), 'fail': int((df.status == 'fail').sum()),
            'fail_rate_%': round(100 * (df.status == 'fail').mean(), 1),
            'MdAPE_%': round(ok.ape.median(), 2) if len(ok) else None,
            'MAPE_%': round(ok.ape.mean(), 2) if len(ok) else None,
            'within10_%': round(100 * (ok.ape <= 10).mean(), 1) if len(ok) else None,
            'within20_%': round(100 * (ok.ape <= 20).mean(), 1) if len(ok) else None,
            'range_cover_%': round(100 * ((ok.actual >= ok.price_low) & (ok.actual <= ok.price_high)).mean(), 1)
            if len(ok) else None}


def md_table(rows):
    """dict 목록을 마크다운 표 문자열로 만든다(첫 행의 키가 열 이름)."""
    rows = list(rows)
    head = list(rows[0].keys())
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
                     + ['| ' + ' | '.join(str(r[h]) for h in head) + ' |' for r in rows])


def main():
    """파라미터 선택 → 면적 보완 배수 → 범위·신뢰도 보정(calib) → test 평가·보고서 저장."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--api-sample', type=int, default=0, help='test 중 N건을 면적 없이 건축물대장 실조회로 평가')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stderr)
    t = split(load_trades())
    pool = t[t.role == 'pool']
    cal, test = t[t.role == 'calib'], t[t.role == 'test']
    log.info('pool %d, calib %d건(%d개 건물), test %d건(%d개 건물)', len(pool), len(cal), cal.bkey.nunique(),
             len(test), test.bkey.nunique())
    months = sorted({(d.year, d.month) for d in pd.concat([cal, test]).deal_date})
    loc = fetch_locations(t, api)  # 건물·역 좌표(가격 정보 아님), 카카오 정책상 저장하지 않고 메모리에서만 사용
    models = models_for(pool, months, loc=loc)

    # 1) 보정 표본으로 파라미터 선택(중앙값 절대오차율 최소)
    best = None
    grid = {**GRID, 'near': GRID['near'] if loc[0] else ['bun']}
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid, combo))
        mdape = run(models, cal, params).ape.median()
        if best is None or mdape < best[0]:
            best = (mdape, params)
    params = best[1]
    # 2) 면적 보완 오차 배수: 같은 calib 거래를 면적 없이 예측했을 때 |log 오차| 중앙값 / 면적 입력 시
    c_main = run(models, cal, params)
    c_na = run(models, cal, params, drop_area=True)
    base = np.log(c_main.actual / c_main.price_est).abs()
    na_err = np.log(c_na.actual / c_na.price_est).abs()
    area_factor = {src: round(max(1.0, float(na_err[c_na.area_src == src].median() / base[c_na.area_src == src].median())), 3)
                   for src in c_na.area_src.dropna().unique() if (c_na.area_src == src).sum() >= 10}
    # 연식 보완 오차 배수: 건축물대장 조회 실패(연식을 법정동 중앙값으로 대체)한 행의 오차 비율
    c_nr = run(models, cal, params, lookup=None)
    imp = c_nr.by_imputed.fillna(False).astype(bool)
    by_factor = round(max(1.0, float(np.log(c_nr.actual / c_nr.price_est).abs()[imp].median()
                                     / base[imp].median())), 3) if imp.sum() >= 10 else 1.3
    for mo in models.values():
        mo.calib = {'area_factor': area_factor, 'by_factor': by_factor}
    # 3) 면적 입력·미입력 calib을 합쳐 범위 배수 z와 sigma 구간별 신뢰도 산정
    c = pd.concat([c_main, run(models, cal, params, drop_area=True), run(models, cal, params, lookup=None)])
    c_ok = c[c.status == 'ok']
    ratio = (np.log(c_ok.actual / c_ok.price_est).abs() / c_ok.sigma)
    z = float(np.quantile(ratio, COVER))
    edges = [float(x) for x in np.quantile(c_ok.sigma, [0.2, 0.4, 0.6, 0.8])]
    bins = np.searchsorted(edges, c_ok.sigma, side='right')
    rates = [float((c_ok.ape[bins == i] <= HIT * 100).mean()) for i in range(5)]
    rates = [round(float(x), 3) for x in np.minimum.accumulate(rates)]  # sigma가 클수록 신뢰도 비증가
    calib = {'params': params, 'area_factor': area_factor, 'by_factor': by_factor, 'z': round(z, 3), 'edges': edges, 'rates': rates,
             'definition': f'신뢰도 = 보정 표본 중 같은 sigma 구간 물건의 절대오차율 {int(HIT*100)}% 이내 비율',
             'cover_target': COVER, 'n_calib': len(c_main), 'calib_MdAPE_%': round(float(best[0]), 2),
             'created_at': datetime.now().isoformat(timespec='seconds'), 'base_date': str(BASE_DATE)}
    (DATA / 'calibration.json').write_text(json.dumps(calib, ensure_ascii=False, indent=1), encoding='utf-8')
    log.info('보정: %s', calib)

    # 4) 최종 평가(test)
    for mo in models.values():
        mo.calib = calib
    res = {'main': run(models, test, params)}
    res['no_area'] = run(models, test, params, drop_area=True)
    res['no_register'] = run(models, test, params, lookup=None)
    unseen_pool = pool[~pool.bkey.isin(set(test.bkey))]
    res['unseen_building'] = run(models_for(unseen_pool, months, calib, loc), test, params)
    res['no_kakao'] = run(models_for(pool, months, calib), test, params)  # 카카오 키 없이 실행하는 경우
    if a.api_sample:
        sample = test.sample(min(a.api_sample, len(test)), random_state=0)
        res['no_area_api'] = run(models, sample, params, drop_area=True, lookup=api)
    REPORTS.mkdir(exist_ok=True)
    allres = pd.concat([d.assign(variant=k) for k, d in res.items()])
    allres.to_csv(REPORTS / 'eval_test.csv', index=False, encoding='utf-8')
    write_report(t, calib, res)


def write_report(t, calib, res):
    """test 결과를 변형·권역·거래유형·신뢰도 구간·오차 상위 사례별로 reports/eval_summary.md에 쓴다."""
    main = res['main']
    ok = main[main.status == 'ok'].copy()
    lines = [f'# 자체 검증 결과 (기준일 {BASE_DATE}, 생성 {datetime.now():%Y-%m-%d %H:%M})', '',
             '## 분리', '', md_table([{'역할': r, '거래 수': int((t.role == r).sum()),
                                      '건물 수': t[t.role == r].bkey.nunique()} for r in ('pool', 'calib', 'test')]),
             '', f'보정 결과: 파라미터 {calib["params"]}, 범위 배수 z={calib["z"]}(목표 포함률 {COVER:.0%}), '
             f'sigma 구간 경계 {[round(e, 3) for e in calib["edges"]]}, 구간별 신뢰도 {calib["rates"]}', '',
             '## 변형별 결과(test)', '',
             md_table([{'변형': k, **summarize(d)} for k, d in res.items()]), '',
             '- 기본 평가는 건축물대장 대신 평가 거래의 건축년도를 제공하는 대역을 사용한다. 실제 조회 성공을 가정한 모델 평가이며 CLI 전체 경로의 실조회 성능이 아니다.',
             f'- main {len(main)}건 중 {int(main.basis.str.contains("건축물대장 사용승인", na=False).sum())}건이 대역의 건축년도를 사용했다. 가격 정답은 제공하지 않는다.',
             '- main: 면적 입력, 호 미입력 / no_area: 면적·호 미입력, 면적은 거래 자료로 대체 추정',
             '- no_register: 건축물대장 조회 실패 상황(연식도 동일 지번 거래 없으면 법정동 중앙값)',
             '- unseen_building: test 건물의 과거 거래까지 모두 제외(처음 보는 건물)',
             '- no_kakao: 카카오 좌표 없이 실행(인근 사례는 본번 근접, 지하철역 거리 미사용)',
             '- no_area_api: 면적 미입력, 실제 조회 함수 사용(공공데이터 캐시가 있으면 재사용). 기본 평가와 별도 조건·표본이며 표본 수를 함께 보고한다.', '',
             '## 권역별(main)', '',
             md_table([{'권역': f'{sg} {"화곡동" if "강서" in sg else ""}'.strip(), **summarize(g)}
                       for sg, g in main.groupby('sigungu')]), '',
             '## 거래 유형별(main)', '',
             md_table([{'거래유형': k or '(미기재)', **summarize(g)} for k, g in main.groupby('dealing')]), '']
    if len(ok):
        ok['cbin'] = pd.cut(ok.confidence, [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0])
        lines += ['## 신뢰도 구간별 실제 오차(main)', '',
                  md_table([{'신뢰도': str(k), 'n': len(g), '평균 신뢰도': round(g.confidence.mean(), 2),
                             '실제 10% 이내 비율': round((g.ape <= 10).mean(), 2),
                             'MdAPE_%': round(g.ape.median(), 1),
                             '범위 포함률': round(((g.actual >= g.price_low) & (g.actual <= g.price_high)).mean(), 2)}
                            for k, g in ok.groupby('cbin', observed=True)]), '',
                  '## 동일 지번 비교사례 유무(main)', '',
                  md_table([{'동일 지번 거래': '있음' if k else '없음', **summarize(g)}
                            for k, g in main.assign(has=main.n_bldg.fillna(0) > 0).groupby('has')]), '',
                  '## 오차 상위 10건(main)', '',
                  md_table([{'법정동': r.dong, '지번': r.jibun, '층': r.floor, '면적': r.area_m2,
                             '건축년도': r.build_year, '계약일': r.deal_date, '거래유형': r.dealing,
                             '실거래가': f'{r.actual:,}', '추정': f'{int(r.price_est):,}', 'APE_%': round(r.ape, 1),
                             '동일지번': int(r.n_bldg)} for r in ok.nlargest(10, 'ape').itertuples()]), '']
    (REPORTS / 'eval_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    log.info('reports/eval_summary.md, reports/eval_test.csv 저장')


if __name__ == '__main__':
    main()
