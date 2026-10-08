"""헤도닉 회귀 적합성 진단과 후보 변수 비교 -> reports/model_diagnosis.md.

python diagnose.py
test 표본은 사용하지 않는다. pool로 적합하고 잔차 구조를 보며, 후보 변수 채택 여부는 calib 표본 외 오차로 판단한다.
"""
import logging
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from evaluate import REPORTS, md_table, split
import api
from model import BASE_DATE, LAWD, NB, Model, fetch_locations, load_trades

SPLIT = BASE_DATE - timedelta(days=365)  # 후보 변수 비교 시 pool 적합 기준일(calib 기간 시작)


def residual_tables(m):
    """pool 적합 잔차를 연식·면적·층·유형·대지권 비율 구간별로 요약한다(평균이 0에서 벗어나면 함수형 누락)."""
    r = m.fit_rows.copy()
    r['lr'] = pd.to_numeric(r.land_m2, errors='coerce') / r.area_m2
    y = np.log(r.price / r.area_m2)
    out = [f'- 적합 거래 {len(r):,}건, R² {1 - r.r.var() / y.var():.3f}, 잔차 SD {r.r.std():.3f}, robust SD {m.s_model:.3f}']
    b = r.groupby(['sigungu', 'dong', 'jibun']).r
    multi = b.transform('count') >= 2
    out.append(f'- 잔차 분산 중 건물(지번) 평균이 차지하는 비중(거래 2건 이상 건물): '
               f'{b.transform("mean")[multi].var() / r.r[multi].var():.2f}')
    specs = [('연식(년)', r.age, [-1, 5, 10, 20, 30, 35, 40, 70]),
             ('전용면적(㎡)', r.area_m2, [0, 20, 30, 40, 50, 60, 85, 300]),
             ('층', r.floor, [-5, -1, 1, 2, 3, 4, 5, 30]),
             ('주택유형', r.house_type, None),
             ('대지권/전용 비율', r.lr, [0, .4, .5, .6, .8, 1, 1.5, 10])]
    for name, col, bins in specs:
        g = pd.cut(col, bins) if bins else col
        rows = [{name: str(k), 'n': len(v), '잔차 평균': round(v.mean(), 3), '잔차 SD': round(v.std(), 3)}
                for k, v in r.r.groupby(g, observed=True)]
        out += ['', f'### {name}별 잔차', '', md_table(rows)]
    return out


def compare_variables(t, geo, st):
    """기본 변수 집합에 후보 변수를 더해 pool(SPLIT 이전)로 적합하고 calib 표본 외 오차를 비교한다."""
    t = t.copy()
    t['blr'] = (pd.to_numeric(t.land_m2, errors='coerce') / t.area_m2).groupby(
        [t.sigungu, t.dong, t.jibun]).transform('median')  # 건물 단위 대지권 비율(건물 속성)
    if geo and st is not None:  # 가장 가까운 지하철역 거리(m). 좌표 없는 건물은 법정동 중앙값
        xy = np.array([geo.get(k, (np.nan, np.nan)) for k in zip(t.sigungu, t.dong, t.jibun)])
        d = np.sqrt(((xy[:, None, :] - st[None, :, :]) ** 2).sum(-1)).min(1)
        t['sta'] = pd.Series(d, index=t.index).fillna(pd.Series(d, index=t.index).groupby([t.sigungu, t.dong]).transform('median'))
    pool = t[(t.role == 'pool') & (t.deal_date < SPLIT) & (t.deal_date >= SPLIT - timedelta(days=5 * 365))
             & (t.dealing != '직거래')].copy()
    cal = t[t.role == 'calib'].copy()
    dongs = sorted(set(zip(pool.sigungu, pool.dong)))
    di, sgs = {k: i for i, k in enumerate(dongs)}, sorted(LAWD)

    def X(d, v, bucket0=False):
        """후보 변수 v를 포함한 설계행렬."""
        age = (d.deal_date.map(lambda x: x.year) - d.build_year).clip(0, 70).to_numpy() / 10
        la = np.log(d.area_m2.to_numpy()) - np.log(50)
        f = d.floor.astype(int).to_numpy()
        cols = [la, la ** 2, age, age ** 2, f < 0, f == 1, f >= 4]
        if 'shape' in v:
            cols += [f >= 5, d.area_m2.to_numpy() < 20, (d.house_type == '연립').to_numpy()]
        if 'station' in v:
            ls = np.log(d.sta.clip(100, 3000).to_numpy() / 500)
            cols += [ls, ls ** 2]
        if 'land' in v:
            lr = np.log(d.blr.clip(0.1, 3).to_numpy())
            cols += [lr, lr ** 2, lr * age]
        D, T = np.zeros((len(d), len(dongs))), np.zeros((len(d), len(sgs) * NB))
        b = np.zeros(len(d), int) if bucket0 else \
            ((SPLIT - d.deal_date).map(lambda x: x.days) // 91).clip(upper=NB - 1).to_numpy()
        for i, (s, dn) in enumerate(zip(d.sigungu, d.dong)):
            if (s, dn) in di:
                D[i, di[(s, dn)]] = 1
            T[i, sgs.index(s) * NB + b[i]] = 1
        return np.column_stack([np.asarray(c, float) for c in cols] + [D, T])

    rows = []
    for v, label in [((), '기본'), (('shape',), '+형태(5층+, 20㎡ 미만, 연립)'), (('land',), '+대지권 비율'),
                     (('shape', 'land'), '+형태+대지권 비율')] + \
            ([(('shape', 'station'), '+형태+지하철역 거리')] if 'sta' in t else []):
        Xp, y = X(pool, v), np.log(pool.price / pool.area_m2).to_numpy()
        beta = np.linalg.solve(Xp.T @ Xp + 1e-3 * np.eye(Xp.shape[1]), Xp.T @ (y - y.mean()))
        res = y - y.mean() - Xp @ beta
        g = pd.Series(res, index=pool.index)[pool.deal_date >= SPLIT - timedelta(days=3 * 365)].groupby(
            [pool.sigungu, pool.dong, pool.jibun]).agg(['sum', 'count'])
        keys = list(zip(cal.sigungu, cal.dong, cal.jibun))
        s = np.array([g['sum'].get(k, 0.0) for k in keys])
        n = np.array([g['count'].get(k, 0) for k in keys])
        yh = y.mean() + X(cal, v, True) @ beta
        ape_h = np.abs(np.exp(yh) * cal.area_m2 / cal.price - 1) * 100
        ape_c = np.abs(np.exp(yh + s / (n + 1)) * cal.area_m2 / cal.price - 1) * 100
        rows.append({'변수 집합': label, 'pool R²': round(1 - res.var() / y.var(), 3),
                     'calib MdAPE 회귀만': round(float(np.median(ape_h)), 2),
                     'calib MdAPE 동일지번 보정': round(float(np.median(ape_c)), 2),
                     '동일지번 없음': round(float(np.median(ape_h[n == 0])), 2),
                     '동일지번 있음': round(float(np.median(ape_c[n > 0])), 2)})
    return rows, len(cal)


def main():
    """진단 표와 후보 변수 비교 결과를 reports/model_diagnosis.md로 저장한다."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stderr)
    t = split(load_trades())
    geo, sta = fetch_locations(t, api)  # 카카오 좌표는 저장하지 않고 이번 실행 메모리에서만 사용
    m = Model(t[t.role == 'pool'], BASE_DATE, geo=geo, stations=sta)
    rows, ncal = compare_variables(t, geo, sta[0] if sta else None)
    lines = [f'# 헤도닉 회귀 진단 (기준일 {BASE_DATE}, test 표본 미사용)', '', '## 적합도와 잔차 구조(pool)', '',
             *residual_tables(m), '', f'## 후보 변수 비교(calib {ncal}건, pool은 {SPLIT} 이전으로 적합)', '',
             '시점 효과는 calib 거래 모두 최근 구간으로 두었으므로 절대 수준보다 변수 집합 간 차이를 본다.', '',
             md_table(rows), '']
    (REPORTS / 'model_diagnosis.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
