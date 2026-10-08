"""입력 정규화와 시세 추정. predict.py(실행)와 evaluate.py(검증)가 같은 Model.estimate를 사용한다.

추정 방식
1. 헤도닉 회귀: log(㎡당 가격) ~ 법정동 + 전용면적 + 연식 + 층 구분 + 연립 + 지하철역 거리 + 시군구별 분기(91일 단위).
   계수는 기준일 이전 실거래로 적합한다(릿지, 소량 정규화).
2. 비교사례 잔차 보정: 동일 건물(같은 지번) 과거 거래와 인근 거래(좌표 거리 또는 같은 법정동 본번 차이 기준)의
   회귀 잔차를 축소 평균하여 더한다. 건물 고유의 가격 수준과 국지적 입지를 반영한다.
3. 오차 척도 sigma: 사례 수·사례 분산·면적 보완 여부로 계산하고, 보정 표본에서 sigma 구간별
   실제 오차로 범위 배수(z)와 신뢰도(오차 10% 이내 비율)를 정한다(data/calibration.json).
"""
import csv
import json
import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get('PROPERTY_DATA_DIR', Path(__file__).parent / 'data'))
log = logging.getLogger('model')
BASE_DATE = date(2026, 10, 6)  # 산출 기준일 = 과제 안내일. 이 날짜 이전 계약만 사용
LAWD = {'서울특별시 강서구': '11500', '서울특별시 관악구': '11620', '서울특별시 강남구': '11680'}
DONG_ONLY = {'서울특별시 강서구': {'화곡동'}}  # 강서구는 화곡동만 대상
YEARS_FIT = 5        # 회귀 적합 기간(년)
YEARS_COMP = 3       # 비교사례 기간(년)
NB = YEARS_FIT * 4   # 분기 구간 수
HIT = 0.10           # 신뢰도 정의: 절대오차율 10% 이내
DEFAULT_PARAMS = {'k_b': 1.0, 'm': 10, 'k_n': 3.0, 'near': 'bun'}  # evaluate.py가 보정 표본으로 선택(near: bun/dist)
# 면적·연식 보완 시 오차 배수 초기값. evaluate.py가 보정 표본으로 다시 추정한 값(calibration.json)을 우선 사용
AREA_FACTOR = {'given': 1.0, 'register': 1.0, 'register_median': 1.2, 'building_trades': 1.2, 'dong_median': 1.6}


def load_dongs():
    """data/bjdong.csv를 읽어 {(시군구, 법정동): 법정동코드 10자리}로 반환한다."""
    with open(DATA / 'bjdong.csv', encoding='utf-8') as f:
        return {(r['sigungu'], r['dong']): r['bjdong_code'] for r in csv.DictReader(f)}


def to_m(lon, lat):
    """경도·위도를 서울 기준(127.0E, 37.5N) 평면 좌표(m)로 근사 변환한다. 수 km 범위 거리 계산용."""
    return (np.asarray(lon, float) - 127.0) * 88_320, (np.asarray(lat, float) - 37.5) * 110_950


def fetch_locations(trades, api, workers=8):
    """카카오 로컬 API를 실시간 호출해 거래 건물 좌표와 지하철역 좌표를 메모리로만 얻는다(카카오는 결과 저장 금지).
    반환 (geo, stations): geo={(시군구, 법정동, 지번): (x_m, y_m)}, stations=(역 좌표 n×2, 역명 목록) 또는 None.
    키가 없거나 조회가 모두 실패하면 ({}, None)이며, 모델은 본번 근접 인근 사례·역 거리 미사용으로 동작한다."""
    if not os.environ.get('KAKAO_API_KEY'):
        return {}, None
    codes = load_dongs()
    keys = sorted(set(zip(trades.sigungu, trades.dong, trades.jibun)))

    def one(k):
        """건물 1개 (키, (경도, 위도) 또는 None)."""
        san, bun, ji = parse_jibun(k[2])
        try:
            return k, api.geocode(codes[(k[0], k[1])], san, bun, ji, ' '.join(k))
        except Exception:
            return k, None

    with ThreadPoolExecutor(workers) as ex:
        res = [(k, ll) for k, ll in ex.map(one, keys) if ll]
    if time.monotonic() >= getattr(api, 'KAKAO_DEADLINE', math.inf):
        log.warning('좌표 조회 시간 소진: 부분 좌표를 버리고 좌표 없는 모델로 전환')
        return {}, None
    log.info('좌표 실시간 조회: 건물 %d곳 중 %d곳 확인', len(keys), len(res))
    if not res:
        return {}, None
    lon, lat = np.array([ll[0] for _, ll in res]), np.array([ll[1] for _, ll in res])
    x, y = to_m(lon, lat)
    geo = {k: (float(a), float(b)) for (k, _), a, b in zip(res, x, y)}
    st = set()
    sgs = np.array([k[0] for k, _ in res])
    for sg in LAWD:  # 시군구별 건물 범위 + 약 2km 여유 안의 지하철역
        m = sgs == sg
        if m.any():
            try:
                st |= set(api.subway_stations(f'{lon[m].min() - .02},{lat[m].min() - .02},'
                                              f'{lon[m].max() + .02},{lat[m].max() + .02}'))
            except Exception as e:
                log.warning('%s 지하철역 조회 실패: %s', sg, e)
    log.info('지하철역 %d개', len(st))
    if time.monotonic() >= getattr(api, 'KAKAO_DEADLINE', math.inf):
        return {}, None
    if not st:
        return geo, None
    st = sorted(st)
    return geo, (np.column_stack(to_m([v[1] for v in st], [v[2] for v in st])), [v[0] for v in st])


def station_term(dist):
    """지하철역 거리(m)를 회귀 항으로: log(거리/500), 100~3000m로 제한. diagnose.py 비교로 채택."""
    return np.log(np.clip(np.asarray(dist, float), 100, 3000) / 500)


# ---------- 입력 정규화 ----------

def norm_sigungu(s):
    """'서울특별시 관악구', '서울 관악구', '관악구' 등을 정식 명칭으로 바꾼다. 대상 외는 None."""
    s = re.sub(r'\s+', ' ', s.strip())
    gu = s.split(' ')[-1] if s else ''
    for full in LAWD:
        if full.endswith(' ' + gu) and (s == gu or s.startswith(('서울특별시 ', '서울시 ', '서울 '))):
            return full
    return None


def parse_jibun(s):
    """'123-45', '123', '산12-3', '123번지' -> (산여부, 본번, 부번). 형식 오류는 None."""
    s = s.strip().replace(' ', '').removesuffix('번지')
    san = s.startswith('산')
    m = re.fullmatch(r'(\d{1,4})(?:-(\d{1,4}))?', s.removeprefix('산'))
    if not m or int(m[1]) == 0:
        return None
    return san, int(m[1]), int(m[2] or 0)


def jibun_key(san, bun, ji):
    """(산여부, 본번, 부번)을 비교용 문자열로 만든다. 예: '123-45', '123', '산12-3'."""
    return ('산' if san else '') + (f'{bun}-{ji}' if ji else str(bun))


def parse_floor(s):
    """정수 층. 지하는 음수('B1', '지하1', '-1'). 0층·형식 오류는 None."""
    s = s.strip().upper().removesuffix('층')
    m = re.fullmatch(r'(?:B|지하|지)(\d{1,2})', s)
    if m:
        return -int(m[1]) or None
    m = re.fullmatch(r'-?\d{1,3}', s)
    return int(s) if m and int(s) != 0 and abs(int(s)) <= 80 else None


def parse_area(s):
    """전용면적 문자열을 float으로 바꾼다. 숫자가 아니거나 0 이하·1000㎡ 초과·무한대면 None."""
    try:
        a = float(s.strip().removesuffix('㎡'))
    except ValueError:
        return None
    return a if math.isfinite(a) and 0 < a <= 1000 else None


def norm_ho(s):
    """'301호', 'B01', '비01' 등에서 비교용 키 추출. 숫자만 남긴 값과 원문 둘 다 비교."""
    s = s.strip().removesuffix('호').replace(' ', '')
    return s, re.sub(r'\D', '', s).lstrip('0')


def validate(row, dongs):
    """입력 1행 -> (정규화 dict, None) 또는 (None, 실패 사유)."""
    errs = []
    sg = norm_sigungu(row.get('sigungu') or '')
    dong = (row.get('dong') or '').strip()
    if not sg:
        errs.append(f'대상 권역 밖 시군구({row.get("sigungu")!r})')
    elif dong not in DONG_ONLY.get(sg, {dong}) or (sg, dong) not in dongs:
        errs.append(f'대상 권역 밖 법정동({sg} {dong!r})')
    jb = parse_jibun(row.get('jibun') or '')
    if not jb:
        errs.append(f'지번 형식 오류({row.get("jibun")!r})')
    fl = parse_floor(row.get('floor') or '')
    if fl is None:
        errs.append(f'층 형식 오류({row.get("floor")!r})')
    area_raw = (row.get('area_m2') or '').strip()
    area = parse_area(area_raw) if area_raw else None
    if area_raw and area is None:
        errs.append(f'전용면적 형식 오류({area_raw!r})')
    if errs:
        return None, '; '.join(errs)
    return {'sigungu': sg, 'dong': dong, 'san': jb[0], 'bun': jb[1], 'ji': jb[2],
            'jibun': jibun_key(*jb), 'floor': fl, 'ho': (row.get('ho') or '').strip(), 'area': area}, None


# ---------- 건축물대장 조회(실행 중 보완) ----------

def register_info(api, code, p):
    """건축물대장에서 사용승인연도와 대상 세대 전용면적을 찾는다. 확정 못 하면 근거와 함께 중앙값."""
    info = {}
    titles = api.building_title(code, p['san'], p['bun'], p['ji'])
    years = sorted({t['useAprDay'][:4] for t in titles if (t.get('useAprDay') or '')[:4].isdigit()})
    if years:
        info['build_year'] = int(years[-1]) if len(years) == 1 else int(np.median([int(y) for y in years]))
        info['build_year_note'] = '건축물대장 사용승인' + ('' if len(years) == 1 else f'(동별 {years[0]}~{years[-1]} 중앙값)')
    purposes = {t.get('mainPurpsCdNm') for t in titles if t.get('mainPurpsCdNm')}
    if purposes and not any('다세대' in x or '연립' in x or '공동주택' in x for x in purposes):
        info['purpose_note'] = '대장 주용도 ' + '/'.join(sorted(purposes))
    units = [u for u in api.building_units(code, p['san'], p['bun'], p['ji'])
             if '전유' in (u.get('exposPubuseGbCdNm') or '') and (u.get('area') or '').replace('.', '', 1).isdigit()]
    kinds = {u.get('mainPurpsCdNm') or '' for u in units} - {''}
    if kinds:
        info['rowhouse'] = all('연립' in k for k in kinds)
    f = p['floor']
    on_floor = [u for u in units if (u.get('flrNo') or '').lstrip('-').isdigit()
                and int(u['flrNo']) == abs(f) and ('지하' in (u.get('flrGbCdNm') or '')) == (f < 0)]
    if p['ho']:
        raw, digits = norm_ho(p['ho'])
        hit = [u for u in on_floor if norm_ho(u.get('hoNm') or '')[0] == raw]
        if not hit and digits:
            hit = [u for u in on_floor if norm_ho(u.get('hoNm') or '')[1] == digits]
        areas = _unit_areas(hit)
        if len(set(areas)) == 1:
            info.update(area=areas[0], area_src='register', area_note=f'건축물대장 {p["ho"]}호 전용면적')
            return info
        if areas:
            info.update(area=float(np.median(areas)), area_src='register_median',
                        area_note=f'건축물대장 {f}층 {p["ho"]}호 후보 {len(areas)}세대 면적 {min(areas)}~{max(areas)}㎡ 중앙값(세대 미확정)')
            return info
    areas = _unit_areas(on_floor)
    if len(set(areas)) == 1:
        info.update(area=areas[0], area_src='register', area_note=f'건축물대장 {f}층 세대 전용면적(단일)')
    elif areas:
        info.update(area=float(np.median(areas)), area_src='register_median',
                    area_note=f'건축물대장 {f}층 {len(areas)}세대 {min(areas)}~{max(areas)}㎡ 중앙값(호 미확정)')
    return info


def _unit_areas(units):
    """세대(동+호)별 전유면적 합계."""
    by = {}
    for u in units:
        k = (u.get('dongNm') or '', u.get('hoNm') or '')
        by[k] = by.get(k, 0.0) + float(u['area'])
    return [round(v, 2) for v in by.values()]


# ---------- 모델 ----------

def floor_dummies(f):
    """층 배열을 [지하, 1층, 4층 이상, 5층 이상] 더미로 바꾼다. 2~3층이 기준(5층 이상은 4층 대비 추가 효과)."""
    f = np.asarray(f)
    return np.stack([f < 0, f == 1, f >= 4, f >= 5], axis=1).astype(float)


def sigma_group(age, floor):
    """잔차 분산 구간: 연식(30년 미만/30~34/35+) × 지하 여부. 노후·지하일수록 잔차 분산이 큼(진단 결과)."""
    return (0 if age < 30 else 1 if age < 35 else 2, floor < 0)


class Model:
    """헤도닉 회귀 + 비교사례 잔차 보정 시세 모델. as_of(기준일) 이전 거래만으로 만든다."""
    def __init__(self, trades, as_of=BASE_DATE, params=None, calib=None, geo=None, stations=None):
        """trades: data/trades.csv 형식 DataFrame. as_of 이전 계약만 사용한다.
        geo, stations: fetch_locations() 결과(실행 중 메모리에만 보관). 없으면 좌표 관련 기능 미사용."""
        self.as_of = as_of
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.calib = calib or {}
        self.codes = load_dongs()
        t = trades[(trades.deal_date < as_of) & (trades.deal_date >= as_of - timedelta(days=365 * YEARS_FIT))]
        t = t[t.dealing != '직거래'].copy()  # 직거래는 특수관계 거래 가능성이 높아 비교사례에서 제외
        t['bucket'] = ((as_of - t.deal_date).map(lambda d: d.days) // 91).clip(upper=NB - 1)
        self.dongs = sorted({(s, d) for s, d in zip(t.sigungu, t.dong)})
        self.di = {k: i for i, k in enumerate(self.dongs)}
        counts = t.groupby(['sigungu', 'dong']).size()
        self.dong_weights = (counts / counts.groupby(level=0).transform('sum')).to_dict()
        self.sgs = sorted(LAWD)
        t['age'] = t.deal_date.map(lambda d: d.year) - t.build_year
        self.geo = geo or {}
        self.st_xy, self.st_names = stations if stations else (None, [])
        xy = np.array([self.geo.get(k, (np.nan, np.nan)) for k in zip(t.sigungu, t.dong, t.jibun)]).reshape(-1, 2)
        t['mx'], t['my'] = xy[:, 0], xy[:, 1]
        t['sta'] = self._station(xy)[0] if self.st_xy is not None else np.nan
        # 좌표 없는 건물의 역 거리는 법정동 중앙값으로 대체(전체 1% 미만)
        self.sta_dong = t.groupby(['sigungu', 'dong']).sta.median().to_dict()
        t['sta'] = t.sta.fillna(t.groupby(['sigungu', 'dong']).sta.transform('median')).fillna(500)
        X = self._X(t.sigungu, t.dong, t.age, t.floor, t.area_m2, t.bucket, t.house_type == '연립', t.sta)
        y = np.log(t.price.to_numpy() / t.area_m2.to_numpy())
        self.y0 = float(y.mean())  # 중심화: 릿지(공선성 해소용 소량)가 가격 수준을 줄이지 않게
        self.beta = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ (y - self.y0))
        t['r'] = y - self.y0 - X @ self.beta
        robust = lambda r: float(1.4826 * np.median(np.abs(r - r.median())))
        self.s_model = robust(t.r)
        grp = [sigma_group(a, f) for a, f in zip(t.age, t.floor)]
        self.s_group = {k: robust(g.r) for k, g in t.groupby(pd.Series(grp, index=t.index)) if len(g) >= 50}
        comp = t[t.deal_date >= as_of - timedelta(days=365 * YEARS_COMP)]
        self.by_bldg = {k: g for k, g in comp.groupby(['sigungu', 'dong', 'jibun'])}
        self.by_dong = {k: g for k, g in comp.groupby(['sigungu', 'dong'])}
        self.by_sg = {k: g.dropna(subset=['mx']) for k, g in comp.groupby('sigungu')}
        self.fit_rows = t  # 연식·면적 보완용(동일 건물, 법정동 중앙값)
        self.n_fit = len(t)

    def _station(self, xy):
        """평면 좌표 배열(n×2) -> (가장 가까운 역까지 거리 m, 역 번호). 좌표가 NaN이면 거리 NaN."""
        d = np.sqrt(((np.asarray(xy, float)[:, None, :] - self.st_xy[None, :, :]) ** 2).sum(-1))
        return d.min(1), d.argmin(1)

    def _X(self, sg, dong, age, floor, area, bucket, rowhouse, sta):
        """회귀 설계행렬: log면적(2차), 20㎡ 미만, 연식(2차), 층 더미, 연립 여부, 지하철역 거리(2차),
        법정동 더미, 시군구×91일 구간 더미."""
        sg, dong = list(sg), list(dong)
        n, nd, k = len(sg), len(self.dongs), 12
        age = np.clip(np.asarray(age, float), 0, 70) / 10
        area = np.asarray(area, float)
        la = np.log(area) - np.log(50)
        X = np.zeros((n, k + nd + len(self.sgs) * NB))
        X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4] = la, la ** 2, area < 20, age, age ** 2
        X[:, 5:9] = floor_dummies(floor)
        X[:, 9] = np.asarray(rowhouse, float)
        ls = station_term(sta)
        X[:, 10], X[:, 11] = ls, ls ** 2
        for i, (s, d, b) in enumerate(zip(sg, dong, np.asarray(bucket))):
            if (s, d) in self.di:
                X[i, k + self.di[(s, d)]] = 1
            else:
                for (gs, gd), weight in self.dong_weights.items():
                    if gs == s:
                        X[i, k + self.di[(gs, gd)]] = weight
            X[i, k + nd + self.sgs.index(s) * NB + int(b)] = 1
        return X

    # 결측 보완 -------------------------------------------------
    def _fill(self, p, api):
        """면적은 대장 우선, 연식은 동일 지번 거래 우선으로 보완한다. 동 자료가 없으면 시군구 중앙값."""
        notes, area, src, by, by_note = [], p['area'], 'given' if p['area'] else None, None, None
        rowhouse = None
        key = (p['sigungu'], p['dong'], p['jibun'])
        bt = self.fit_rows[(self.fit_rows.sigungu == key[0]) & (self.fit_rows.dong == key[1])
                           & (self.fit_rows.jibun == key[2])]
        if len(bt):
            by, by_note = int(bt.build_year.mode().iloc[0]), '실거래 건축년도'
            rowhouse = bool((bt.house_type == '연립').mean() > 0.5)
        if api is not None and (area is None or by is None):
            try:
                reg = register_info(api, self.codes[(p['sigungu'], p['dong'])], p)
                if area is None and 'area' in reg:
                    area, src = reg['area'], reg['area_src']
                    notes.append(reg['area_note'])
                if by is None and 'build_year' in reg:
                    by, by_note = reg['build_year'], reg['build_year_note']
                if rowhouse is None and 'rowhouse' in reg:
                    rowhouse = reg['rowhouse']
                if 'purpose_note' in reg:
                    notes.append(reg['purpose_note'])
            except Exception as e:  # 조회 실패는 대체 추정으로 진행
                notes.append(f'건축물대장 조회 실패({str(e)[:60]})')
        if area is None and len(bt):
            same = bt[bt.floor == p['floor']]
            ref = same if len(same) else bt
            area, src = float(ref.area_m2.median()), 'building_trades'
            notes.append(f'면적 미확인→동일 지번 {"같은 층 " if len(same) else ""}거래 {len(ref)}건 면적 중앙값 {area:.1f}㎡')
        dt = self.fit_rows[(self.fit_rows.sigungu == key[0]) & (self.fit_rows.dong == key[1])]
        scope = '법정동'
        if dt.empty:
            dt = self.fit_rows[self.fit_rows.sigungu == key[0]]
            scope = '시군구'
        if dt.empty:
            raise ValueError('외부 조회 후에도 시군구 비교 자료 없음; ' + '; '.join(notes))
        if by is None:
            by, by_note = int(dt.build_year.median()), f'연식 미확인→{scope} 거래 중앙값'
            notes.append(f'{by_note} {by}년')
        if area is None:
            near = dt[(dt.build_year - by).abs() <= 5]
            ref = near if len(near) >= 10 else dt
            area, src = float(ref.area_m2.median()), 'dong_median'
            notes.append(f'면적 미확인→{scope} 유사연식 거래 {len(ref)}건 면적 중앙값 {area:.1f}㎡로 대체 추정')
        xy = self.geo.get(key)
        if xy is None and api is not None and hasattr(api, 'geocode'):
            try:
                ll = api.geocode(self.codes[(p['sigungu'], p['dong'])], p['san'], p['bun'], p['ji'],
                                 f'{p["sigungu"]} {p["dong"]} {p["jibun"]}')
                xy = tuple(float(v) for v in to_m(*ll)) if ll else None
            except Exception as e:
                notes.append(f'좌표 조회 실패({str(e)[:40]})')
        return area, src, by, by_note, bool(rowhouse), xy, notes

    # 추정 -----------------------------------------------------
    def estimate(self, p, api=None):
        """p: validate() 결과. 반환 dict(price_est, price_low, price_high, confidence, basis) 또는 ValueError."""
        unseen_dong = (p['sigungu'], p['dong']) not in self.di
        area, area_src, by, by_note, rowhouse, xy, notes = self._fill(p, api)
        by_imputed = by_note.startswith('연식 미확인')
        age = self.as_of.year - by
        sta, sta_note = self.sta_dong.get((p['sigungu'], p['dong']), 500), '역 거리 법정동 중앙값'
        if self.st_xy is None:
            sta_note = '역 거리 미사용'
        elif unseen_dong:
            sta_note = '역 거리 미확인→500m 대체'
        if xy is not None and self.st_xy is not None:
            dist, idx = self._station([xy])
            sta, sta_note = float(dist[0]), f'{self.st_names[idx[0]]} {dist[0]:,.0f}m'
        if not np.isfinite(sta):
            sta = 500
        x = self._X([p['sigungu']], [p['dong']], [age], [p['floor']], [area], [0], [rowhouse], [sta])[0]
        yhat = self.y0 + float(x @ self.beta)
        kb, m, kn = self.params['k_b'], int(self.params['m']), self.params['k_n']
        b = self.by_bldg.get((p['sigungu'], p['dong'], p['jibun']))
        nb = 0 if b is None else len(b)
        d = self.by_dong[(p['sigungu'], p['dong'])] if (p['sigungu'], p['dong']) in self.by_dong else None
        rn, nn, near_note = 0.0, 0, ''
        g = self.by_sg.get(p['sigungu'])
        if self.params.get('near') == 'dist' and xy is not None and g is not None and len(g):
            g = g[((g.dong != p['dong']) | (g.jibun != p['jibun'])) & ((g.build_year - by).abs() <= 10)]
            dist = np.hypot(g.mx.to_numpy() - xy[0], g.my.to_numpy() - xy[1])
            idx = np.argsort(dist, kind='stable')[:m]
            nn = len(idx)
            if nn:
                rn = float(g.r.iloc[idx].sum() / (nn + kn))
                near_note = f'반경 {dist[idx].max():,.0f}m 내 '
        elif d is not None:
            d = d[(d.jibun != p['jibun']) & ((d.build_year - by).abs() <= 10)]
            if len(d):
                near = d.iloc[np.argsort((d.bun - p['bun']).abs().to_numpy() + (d.san != p['san']) * 10000,
                                         kind='stable')[:m]]
                nn = len(near)
                rn = float(near.r.sum() / (nn + kn))
        rb = float((b.r.sum() + kb * rn) / (nb + kb)) if nb else rn
        ppm = math.exp(yhat + rb)
        est = ppm * area
        sd_b = float(b.r.std()) if nb >= 2 else 0.0
        s0 = self.s_group.get(sigma_group(age, p['floor']), self.s_model)
        sigma = math.sqrt(s0 ** 2 * kb / (nb + kb) + (sd_b ** 2 / nb if nb >= 2 else 0)) \
            * {**AREA_FACTOR, **self.calib.get('area_factor', {})}[area_src] \
            * (self.calib.get('by_factor', 1.3) if by_imputed else 1.0)
        z, conf = self._calibrated(sigma)
        if unseen_dong:
            # ponytail: 미관측 동은 시군구 위치효과로 대체; 별도 홀드아웃 보정 전 신뢰도는 0.
            sigma, conf = max(sigma, self.s_model), 0.0
            notes.append('법정동 학습 자료 없음→시군구 거래 비중으로 위치효과 대체; 가격 범위·신뢰도 미검증(신뢰도 0)')
        low, high = est * math.exp(-z * sigma), est * math.exp(z * sigma)
        r = lambda v: int(round(v / 10000)) * 10000  # 만원 단위
        price_est, price_low, price_high = r(est), r(low), r(high)
        price_low, price_high = min(price_low, price_est), max(price_high, price_est)
        if price_low <= 0:
            raise ValueError('추정 가격이 양수가 아님')
        parts = [f'헤도닉 {p["dong"]}·{by}년식({by_note})·{p["floor"]}층·{area:.1f}㎡{"·연립" if rowhouse else ""}·{sta_note} 기준 ㎡당 {math.exp(yhat)/1e4:,.0f}만원']
        if nb:
            last = b.deal_date.max()
            parts.append(f'동일 지번 {p["jibun"]} 최근{YEARS_COMP}년 거래 {nb}건(최근 {last:%Y-%m}) 잔차보정 {math.exp(rb)-1:+.1%}')
        else:
            parts.append(f'동일 지번 거래 없음, {near_note}인근 {nn}건 잔차보정 {math.exp(rb)-1:+.1%}')
        parts.append(f'추정 ㎡당 {ppm/1e4:,.0f}만원')
        if area_src == 'given':
            parts.append('면적 입력값')
        parts += notes
        if not p['ho']:
            parts.append('호 미입력(층 기준 추정)')
        return {'price_est': price_est, 'price_low': price_low, 'price_high': price_high,
                'confidence': round(conf, 2), 'basis': '; '.join(parts), 'sigma': sigma,
                'area_used': area, 'area_src': area_src, 'by_imputed': by_imputed, 'n_bldg': nb, 'n_near': nn}

    def _calibrated(self, sigma):
        """보정 표본에서 정한 범위 배수 z와 sigma 구간별 오차 10% 이내 비율."""
        if 'edges' not in self.calib:
            return 1.28, max(0.0, min(1.0, math.erf(math.log(1 + HIT) / (sigma * math.sqrt(2)))))
        i = int(np.searchsorted(self.calib['edges'], sigma, side='right'))
        return self.calib['z'], self.calib['rates'][i]


def load_trades(path=None):
    """trades.csv를 읽어 날짜·연식 형식을 맞추고 산 여부와 본번 열을 추가한다. 면적·연식 결측 행은 제외."""
    df = pd.read_csv(path or DATA / 'trades.csv', dtype={'jibun': str, 'dealing': str, 'house_type': str},
                     keep_default_na=False, na_values={'area_m2': [''], 'build_year': ['']})
    df['deal_date'] = pd.to_datetime(df.deal_date).dt.date
    df = df.dropna(subset=['area_m2', 'build_year'])
    df['build_year'] = df.build_year.astype(int)
    df['san'] = df.jibun.str.startswith('산')
    df['bun'] = df.jibun.str.removeprefix('산').str.split('-').str[0].astype(int)
    return df


def load_model(as_of=BASE_DATE, api=None):
    """수집 거래와 data/calibration.json(있으면)으로 기준일 모델을 만든다. api가 있으면 좌표를 실시간 조회한다."""
    calib_path = DATA / 'calibration.json'
    calib = json.loads(calib_path.read_text(encoding='utf-8')) if calib_path.exists() else None
    trades = load_trades()
    geo, stations = fetch_locations(trades, api) if api is not None else ({}, None)
    return Model(trades, as_of, calib['params'] if calib else None, calib, geo, stations)
