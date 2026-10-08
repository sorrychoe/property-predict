"""입출력 계약과 결측·실패 처리 테스트. 외부 API 없이 합성 거래 데이터로 실행한다.

python test_predict.py
합성 데이터 결과는 실제 가격 성능 근거가 아니다(실제 성능은 evaluate.py).
"""
import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from io import BytesIO
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import predict
from collect import COLS
from model import BASE_DATE, Model, load_dongs, load_trades, parse_floor, parse_jibun, validate, register_info, fetch_locations

HERE = Path(__file__).parent
DONGS = {('서울특별시 관악구', '봉천동'): 900, ('서울특별시 관악구', '신림동'): 800,
         ('서울특별시 강서구', '화곡동'): 700, ('서울특별시 강남구', '역삼동'): 1500}


def synth_trades(path, seed=0):
    """합성 거래: 법정동 ㎡단가 × 연식·층·건물 효과 × 시간 추세 × 잡음."""
    rng = np.random.default_rng(seed)
    rows, start = [], BASE_DATE - timedelta(days=5 * 365)
    for (sg, dong), base in DONGS.items():
        for b in range(40):
            bun, ji = int(rng.integers(1, 1500)), int(rng.integers(0, 30))
            by, eff = int(rng.integers(1990, 2024)), rng.normal(0, 0.08)
            areas = rng.choice([30.5, 42.1, 55.3, 68.9], 2)
            for _ in range(6):
                d = start + timedelta(days=int(rng.integers(0, (BASE_DATE - start).days)))
                fl, area = int(rng.choice([-1, 1, 2, 3, 4, 5])), float(rng.choice(areas))
                age = d.year - by
                lp = math.log(base * 1e4) - 0.01 * age + 0.03 * (fl >= 2) - 0.15 * (fl < 0) + eff \
                    + 0.0015 * (d - start).days / 30 + rng.normal(0, 0.05)
                rows.append({'sigungu': sg, 'dong': dong, 'jibun': f'{bun}-{ji}' if ji else str(bun), 'name': '',
                             'house_type': '', 'area_m2': area, 'land_m2': '', 'floor': fl, 'build_year': by,
                             'deal_date': d.isoformat(), 'price': int(round(math.exp(lp) * area / 1e4)) * 10000,
                             'dealing': '직거래' if rng.random() < 0.05 else '중개거래', 'rgst_date': ''})
    pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)


class FakeApi:
    """건축물대장 응답 형식(swagger 필드명)을 흉내 낸 고정 응답."""
    def __init__(self, fail=False):
        """fail=True면 조회 시 예외를 던져 API 오류 상황을 흉내 낸다."""
        self.fail = fail

    def building_title(self, code, san, bun, ji):
        """표제부 고정 응답(사용승인 2015년, 다세대)."""
        if self.fail:
            raise RuntimeError('timeout')
        return [{'useAprDay': '20150310', 'mainPurpsCdNm': '공동주택(다세대주택)'}]

    def building_units(self, code, san, bun, ji):
        """전유공용면적 고정 응답: 3층 301호 41.2㎡, 302호 55.0㎡(공용 행 포함)."""
        return [{'exposPubuseGbCdNm': '전유', 'flrGbCdNm': '지상', 'flrNo': '3', 'hoNm': '301호', 'dongNm': '', 'area': '41.2'},
                {'exposPubuseGbCdNm': '공용', 'flrGbCdNm': '지상', 'flrNo': '3', 'hoNm': '301호', 'dongNm': '', 'area': '9.0'},
                {'exposPubuseGbCdNm': '전유', 'flrGbCdNm': '지상', 'flrNo': '3', 'hoNm': '302호', 'dongNm': '', 'area': '55.0'}]


class Base(unittest.TestCase):
    """합성 거래 데이터를 임시 폴더에 만들고 모델을 한 번 적합해 공유한다."""
    @classmethod
    def setUpClass(cls):
        """임시 데이터 폴더에 법정동 코드와 합성 거래를 쓰고 기준일 모델을 만든다."""
        cls.tmp = Path(tempfile.mkdtemp())
        shutil.copy(HERE / 'data' / 'bjdong.csv', cls.tmp / 'bjdong.csv')
        synth_trades(cls.tmp / 'trades.csv')
        cls.trades = load_trades(cls.tmp / 'trades.csv')
        cls.model = Model(cls.trades)
        cls.known = cls.trades.iloc[0]  # 거래가 있는 건물

    @classmethod
    def tearDownClass(cls):
        """임시 데이터 폴더를 삭제한다."""
        shutil.rmtree(cls.tmp)

    def row(self, **kw):
        """거래가 있는 건물을 가리키는 기본 입력 행에 kw로 일부 값을 바꿔 반환한다."""
        k = self.known
        r = {'id': '1', 'sigungu': k.sigungu, 'dong': k.dong, 'jibun': k.jibun, 'floor': '3', 'ho': '301', 'area_m2': '42.1'}
        return {**r, **kw}


class TestParse(Base):
    """지번·층·시군구·권역 검증 테스트."""
    def test_parsers(self):
        """지번·층 문자열 해석과 형식 오류 처리."""
        self.assertEqual(parse_jibun('123-045'), (False, 123, 45))
        self.assertEqual(parse_jibun('산12'), (True, 12, 0))
        self.assertIsNone(parse_jibun('12-가'))
        self.assertEqual(parse_floor('B1'), -1)
        self.assertEqual(parse_floor('-2'), -2)
        self.assertIsNone(parse_floor('0'))
        self.assertIsNone(parse_floor('3.5'))

    def test_region(self):
        """시군구 표기 변형 허용, 강서구 화곡동 외·대상 외 시군구·잘못된 면적 거부."""
        codes = load_dongs()
        ok = lambda **kw: validate(self.row(**kw), codes)[1]
        self.assertIsNone(ok(sigungu='서울 관악구', dong='봉천동'))
        self.assertIsNone(ok(sigungu='관악구', dong='신림동'))
        self.assertIn('권역 밖', ok(sigungu='서울특별시 강서구', dong='등촌동'))
        self.assertIn('권역 밖', ok(sigungu='서울특별시 마포구', dong='합정동'))
        self.assertIn('면적', ok(area_m2='abc'))
        self.assertIn('면적', ok(area_m2='-5'))


class TestEstimate(Base):
    """추정 결과의 가격·신뢰도 조건과 결측 보완 경로 테스트."""
    def est(self, api=None, **kw):
        """기본 입력 행을 kw로 바꿔 검증 후 추정한다."""
        p, err = validate(self.row(**kw), load_dongs())
        self.assertIsNone(err)
        return self.model.estimate(p, api)

    def check(self, e):
        """0 < 하한 <= 추정 <= 상한, 신뢰도 0~1 유한값인지 확인한다."""
        self.assertTrue(0 < e['price_low'] <= e['price_est'] <= e['price_high'])
        self.assertTrue(math.isfinite(e['confidence']) and 0 <= e['confidence'] <= 1)

    def test_known_building(self):
        """동일 지번 거래가 있는 물건은 근거에 동일 지번 보정과 면적 입력값이 표시된다."""
        e = self.est()
        self.check(e)
        self.assertIn('동일 지번', e['basis'])
        self.assertIn('면적 입력값', e['basis'])

    def test_area_from_register(self):
        """호 일치 시 건축물대장 면적을 쓰고, 호가 없으면 층 중앙값과 호 미확정을 표시한다."""
        e = self.est(api=FakeApi(), area_m2='')
        self.assertEqual(e['area_used'], 41.2)
        self.assertIn('301호', e['basis'])
        e = self.est(api=FakeApi(), area_m2='', ho='')  # 같은 층 면적이 둘 -> 중앙값, 호 미확정 표시
        self.assertEqual(e['area_src'], 'register_median')
        self.assertIn('호 미입력', e['basis'])

    def test_register_fail_fallbacks(self):
        """조회 실패 시 동일 지번 거래 면적, 미수집 건물이면 법정동 중앙값으로 대체하고 신뢰도가 낮아진다."""
        e = self.est(api=FakeApi(fail=True), area_m2='')
        self.check(e)
        self.assertIn('조회 실패', e['basis'])
        self.assertEqual(e['area_src'], 'building_trades')
        e2 = self.est(api=FakeApi(fail=True), area_m2='', jibun='1499-99')  # 미수집 건물 + 조회 실패
        self.check(e2)
        self.assertEqual(e2['area_src'], 'dong_median')
        self.assertLess(e2['confidence'], self.est()['confidence'] + 1e-9)

    def test_unknown_building_lookup(self):
        """사전 자료에 없는 건물도 건축물대장 조회와 인근 지번 사례로 추정한다."""
        e = self.est(api=FakeApi(), jibun='1499-99', area_m2='')
        self.check(e)
        self.assertIn('건축물대장 사용승인', e['basis'])
        self.assertIn('동일 지번 거래 없음', e['basis'])

    def test_no_dong_data_queries_then_uses_sigungu(self):
        """미수집 동도 먼저 조회하고, 조회 실패 시 시군구 자료로 보완하되 미검증임을 알린다."""
        for fail in (False, True):
            lookup = FakeApi(fail=fail)
            with patch.object(lookup, 'building_title', wraps=lookup.building_title) as title:
                e = self.est(api=lookup, sigungu='서울특별시 강남구', dong='율현동', area_m2='')
            title.assert_called_once()
            self.check(e)
            self.assertEqual(e['confidence'], 0)
            self.assertIn('시군구', e['basis'])
            self.assertIn('미검증', e['basis'])
            self.assertEqual(e['area_src'], 'dong_median' if fail else 'register')

    def test_register_separates_floors_and_prefers_exact_ho(self):
        """화곡동 342-23의 지상101/지층101 충돌과 같은 층의 숫자 유사 호를 구분한다."""
        lookup = FakeApi()
        units = [dict(exposPubuseGbCdNm='전유', flrGbCdNm=kind, flrNo='1', hoNm=ho,
                      dongNm='', area=area)
                 for kind, ho, area in [('지상', '101호', '46.71'), ('지하', '지층101호', '56.34'),
                                        ('지상', '가101호', '70')]]
        with patch.object(lookup, 'building_units', return_value=units):
            for floor, ho, area in [('1', '101', 46.71), ('-1', '지층101', 56.34), ('-1', '101', 56.34)]:
                p, _ = validate(self.row(floor=floor, ho=ho, area_m2=''), load_dongs())
                e = register_info(lookup, '1150010300', p)
                self.assertEqual(e['area'], area)
                self.assertEqual(e['area_src'], 'register')

    def test_synthetic_sanity(self):
        """합성 데이터에서 계산이 크게 어긋나지 않는지만 확인(성능 평가 아님)."""
        recent = self.trades[(self.trades.deal_date >= BASE_DATE - timedelta(days=120)) & (self.trades.dealing != '직거래')]
        apes = []
        for r in recent.itertuples():
            e = self.est(jibun=r.jibun, sigungu=r.sigungu, dong=r.dong, floor=str(r.floor), area_m2=str(r.area_m2))
            apes.append(abs(e['price_est'] - r.price) / r.price)
        self.assertLess(np.median(apes), 0.15)


class TestGeocode(unittest.TestCase):
    """카카오 주소 검색 결과에서 법정동코드·산·본번·부번이 모두 일치하는 단일 좌표만 인정하는지."""
    def test_match_rules(self):
        """일치 1건은 좌표, 부번 불일치·서로 다른 좌표 2건은 None."""
        import api
        doc = lambda x, ji: {'x': x, 'y': '37.5', 'address': {'b_code': '1162010200', 'mountain_yn': 'N',
                                                              'main_address_no': '1465', 'sub_address_no': ji}}
        cases = [([doc('126.9', '1')], (126.9, 37.5)), ([doc('126.9', '2')], None),
                 ([doc('126.9', '1'), doc('126.8', '1')], None)]
        orig = api.kakao
        try:
            for docs, want in cases:
                api.kakao = lambda op, params, docs=docs: {'documents': docs}
                self.assertEqual(api.geocode('1162010200', False, 1465, 1, 'q'), want)
        finally:
            api.kakao = orig


class TestBreaker(unittest.TestCase):
    """외부 서비스 장애 시 연속 실패 후 호출을 끊어 20건 30분 제한을 지키는지."""
    def test_stops_after_consecutive_failures(self):
        """BREAK_AFTER번 실패 뒤에는 요청 없이 바로 실패하고, 다른 서비스에는 영향이 없다."""
        import api
        calls = []

        def boom(path, *a):
            calls.append(path)
            raise api.ApiError('timed out')
        orig, api._get_once = api._get_once, boom
        api._fails.clear()
        try:
            for _ in range(api.BREAK_AFTER + 3):
                with self.assertRaises(api.ApiError):
                    api._get('kakao/search/address', 'u', 1, 0)
            self.assertEqual(len(calls), api.BREAK_AFTER)
            with self.assertRaises(api.ApiError):
                api._get('BldRgstHubService/getBrTitleInfo', 'u', 1, 0)
            self.assertEqual(len(calls), api.BREAK_AFTER + 1)
        finally:
            api._get_once = orig
            api._fails.clear()

    def test_successful_but_slow_requests_use_total_budget(self):
        """성공이 반복되어도 예산 소진 시 중단하고, 대장 조회에는 별도 남은 예산을 쓴다."""
        import api
        now, calls = [0.0], []
        def response(req, timeout):
            calls.append(timeout)
            now[0] += 3
            return BytesIO(b'{}')
        with patch.object(api, 'DEADLINE', 20), patch.object(api, 'KAKAO_DEADLINE', 5), \
             patch.object(api.time, 'monotonic', side_effect=lambda: now[0]), \
             patch.object(api.urllib.request, 'urlopen', side_effect=response):
            self.assertEqual(api._get('kakao/search/address', 'https://example.invalid', 10, 3), b'{}')
            with self.assertRaisesRegex(api.ApiError, '시간 소진'):
                api._get('kakao/search/address', 'https://example.invalid', 10, 3)
            with self.assertRaisesRegex(api.ApiError, '시간 소진'):
                api._get('kakao/search/address', 'https://example.invalid', 10, 3)
            self.assertEqual(calls, [5, 2])
            self.assertEqual(api._get('BldRgstHubService/test', 'https://example.invalid', 10, 3), b'{}')
        api._fails.clear()

    def test_retry_wait_and_partial_coordinates_respect_deadline(self):
        """재시도 대기도 남은 예산 이하이며, 기한을 넘긴 부분 좌표는 사용하지 않는다."""
        import api
        now = [0.0]
        with patch.object(api, 'DEADLINE', 1), patch.object(api.time, 'monotonic', side_effect=lambda: now[0]), \
             patch.object(api.time, 'sleep', side_effect=lambda n: now.__setitem__(0, now[0] + n)) as sleep, \
             patch.object(api.urllib.request, 'urlopen', side_effect=api.socket.timeout('timeout')):
            with self.assertRaisesRegex(api.ApiError, '시간 소진'):
                api._get('BldRgstHubService/test', 'https://example.invalid', 10, 3)
            sleep.assert_called_once_with(1)
        api._fails.clear()
        t = pd.DataFrame([dict(sigungu='서울특별시 관악구', dong='봉천동', jibun='1')])
        with patch.dict(os.environ, KAKAO_API_KEY='test'), patch.object(api, 'KAKAO_DEADLINE', 0), \
             patch.object(api, 'geocode', return_value=(127.0, 37.5)), patch.object(api, 'subway_stations') as stations:
            self.assertEqual(fetch_locations(t, api), ({}, None))
            stations.assert_not_called()


class TestCli(Base):
    """predict.py를 하위 프로세스로 실행해 CSV 계약을 확인한다."""
    def run_cli(self, rows, header=predict.IN_COLS, bom=True):
        """입력 CSV를 쓰고 API 키 없이 predict.py를 실행해 (프로세스 결과, 출력 경로)를 반환한다."""
        inp, out = self.tmp / 'in.csv', self.tmp / 'out.csv'
        with open(inp, 'w', encoding='utf-8-sig' if bom else 'utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        env = {**os.environ, 'RTMS_API_KEY': '', 'BLD_API_KEY': '', 'KAKAO_API_KEY': ''}  # 빈 값이 .env보다 우선 -> 외부 호출 없음
        env['PROPERTY_DATA_DIR'] = str(self.tmp)
        p = subprocess.run([sys.executable, str(HERE / 'predict.py'), '--input', str(inp), '--output', str(out)],
                           env=env, capture_output=True, text=True, timeout=300)
        return p, out

    def test_contract(self):
        """헤더·행 순서·id 원문 보존, 성공 행 가격·신뢰도 조건, 실패 행 빈 값·사유, 행별 근거 차이."""
        k = self.known
        rows = [['007', k.sigungu, k.dong, k.jibun, '3', '301', '42.1'],
                ['A-2', k.sigungu, k.dong, k.jibun, '2', '', ''],          # 면적·호 누락, 키 없음 -> 대체 추정
                ['3', '서울특별시 마포구', '합정동', '1', '1', '', ''],     # 권역 밖
                ['4', k.sigungu, k.dong, 'x-1', '0', '', ''],               # 잘못된 값
                ['5', k.sigungu, k.dong, '1499-99', 'B1', 'B01', '']]       # 미수집 건물, 지하, 면적 누락
        p, out = self.run_cli(rows)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(out, encoding='utf-8', newline='') as f:
            res = list(csv.DictReader(f))
        self.assertEqual(list(res[0].keys()), predict.OUT_COLS)
        self.assertEqual([r['id'] for r in res], ['007', 'A-2', '3', '4', '5'])
        self.assertEqual([r['status'] for r in res], ['ok', 'ok', 'fail', 'fail', 'ok'])
        for r in res:
            if r['status'] == 'ok':
                lo, est, hi = int(r['price_low']), int(r['price_est']), int(r['price_high'])
                self.assertTrue(0 < lo <= est <= hi)
                self.assertTrue(0 <= float(r['confidence']) <= 1)
            else:
                self.assertEqual([r[c] for c in ('price_est', 'price_low', 'price_high', 'confidence')], [''] * 4)
                self.assertTrue(r['basis'].startswith('실패'))
        self.assertIn('조회 실패', res[1]['basis'])
        self.assertNotIn(res[0]['basis'], [res[1]['basis'], res[4]['basis']])

    def test_missing_header_keeps_output(self):
        """필수 헤더가 없으면 비정상 종료하고 기존 출력 파일을 건드리지 않는다."""
        out = self.tmp / 'out.csv'
        out.write_text('previous', encoding='utf-8')
        p, _ = self.run_cli([['1', '관악구', '봉천동', '1', '1', '']], header=predict.IN_COLS[:-1])
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('area_m2', p.stderr)
        self.assertEqual(out.read_text(encoding='utf-8'), 'previous')


if __name__ == '__main__':
    unittest.main(verbosity=2)
