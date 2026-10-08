"""외부 API 호출: 공공데이터포털(연립다세대 매매 실거래, 건축물대장), 카카오 로컬(주소→좌표, 지하철역).

공공데이터포털 엔드포인트와 응답 필드는 data.go.kr 데이터셋 15126467, 15134735 페이지의 swagger 명세,
카카오는 developers.kakao.com 로컬 API 개발 가이드 기준. 공공데이터포털 성공 응답은 data/cache/에 저장해
재사용한다. 카카오 응답은 운영정책상 저장이 금지되어 캐시하지 않고 실행 중 메모리에서만 쓴다. 키는 저장하지 않는다.
"""
import hashlib
import json
import logging
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = 'https://apis.data.go.kr/1613000'
CACHE = Path(os.environ.get('PROPERTY_DATA_DIR', Path(__file__).parent / 'data')) / 'cache'
log = logging.getLogger(__name__)


class ApiError(Exception):
    """API 키 미설정, 네트워크 실패, 오류 응답(resultCode 비정상) 등 외부 조회 실패."""
    pass


def load_env(path=Path(__file__).parent / '.env'):
    """프로젝트 .env의 KEY=VALUE를 환경변수로 읽는다. 이미 설정된 환경변수가 우선한다."""
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            k, sep, v = line.partition('=')
            if sep and not k.strip().startswith('#'):
                os.environ.setdefault(k.strip(), v.strip().strip('\'"'))


load_env()
KEY_VARS = {'RTMS': 'RTMS_API_KEY', 'BldRgst': 'BLD_API_KEY', 'kakao': 'KAKAO_API_KEY'}  # API 접두어 -> 키 환경변수
KAKAO = 'https://dapi.kakao.com/v2/local'
BREAK_AFTER = 5  # 같은 서비스가 연속 이만큼 최종 실패하면 이번 실행 동안 호출을 중단(장애 시 30분 제한 보호)
_fails = {}  # 서비스(경로 첫 단어) -> 연속 실패 수
DEADLINE = float('inf')  # CLI가 설정하는 전체 외부 조회 종료 시각(monotonic)
KAKAO_DEADLINE = float('inf')


def remaining(path):
    """성공 응답·페이지·재시도에도 적용하는 누적 조회 시간 예산."""
    end = min(DEADLINE, KAKAO_DEADLINE) if path.startswith('kakao/') else DEADLINE
    left = end - time.monotonic()
    if left <= 0:
        raise ApiError(f'{path.split("/")[0]} 조회 시간 소진, 수집 자료로 대체 추정')
    return left


def key_var(path):
    """API 경로에 해당하는 키 환경변수 이름."""
    return next(v for p, v in KEY_VARS.items() if path.startswith(p))


def _key(path):
    """API별 환경변수에서 서비스키를 읽는다. 인코딩된 키도 디코딩해 이중 인코딩을 막는다."""
    k = os.environ.get(key_var(path), '').strip()
    if not k:
        raise ApiError(f'환경변수 {key_var(path)} 미설정')
    return urllib.parse.unquote(k)  # 인코딩/디코딩 키 모두 허용


def _get(path, url, timeout, retries, headers=None):
    """GET 본문을 바이트로 반환한다. 네트워크 오류·5xx·429·빈 본문(건축물대장에서 간헐 발생)은
    재시도하고 4xx는 즉시 실패. 같은 서비스가 연속 BREAK_AFTER번 실패하면 이후 호출은 바로 실패시킨다.
    path는 오류 메시지·서비스 구분용."""
    svc = path.split('/')[0]
    remaining(path)
    if _fails.get(svc, 0) >= BREAK_AFTER:
        raise ApiError(f'{svc} 연속 {BREAK_AFTER}회 실패로 이번 실행에서 조회 중단')
    try:
        body = _get_once(path, url, timeout, retries, headers)
    except ApiError:
        _fails[svc] = _fails.get(svc, 0) + 1
        raise
    _fails[svc] = 0
    return body


def _get_once(path, url, timeout, retries, headers):
    """_get의 실제 요청·재시도 부분."""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}),
                                        timeout=min(timeout, remaining(path))) as r:
                chunks = []
                while True:
                    remaining(path)
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                body = b''.join(chunks)
            remaining(path)
            if body.strip():
                return body
            err = '빈 응답'
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            code = getattr(e, 'code', None)
            if code and code < 500 and code != 429:
                raise ApiError(f'{path} 호출 실패: HTTP {code} {getattr(e, "reason", "")}') from None
            err = getattr(e, 'reason', e)
        if attempt == retries:
            raise ApiError(f'{path} 호출 실패: {err}')
        time.sleep(min(2 * (attempt + 1), remaining(path)))


def _parse(body, path):
    """XML 응답을 (item dict 목록, totalCount)로 변환한다. 오류 메시지나 비정상 resultCode는 ApiError."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        raise ApiError(f'{path} 응답 해석 불가: {body[:120]!r}') from None
    err = root.findtext('.//errMsg') or root.findtext('.//returnAuthMsg')
    code = root.findtext('.//resultCode')
    if err or (code is not None and code.strip() not in ('00', '000')):
        raise ApiError(f'{path} 오류: {code} {err or root.findtext(".//resultMsg")}')
    items = [{c.tag: (c.text or '').strip() for c in it} for it in root.iter('item')]
    return items, int(root.findtext('.//totalCount') or 0)


def call(path, params, rows=100, timeout=10, retries=3):
    """모든 페이지를 모아 item dict 목록으로 반환. 성공 결과만 캐시."""
    f = CACHE / (hashlib.sha1(json.dumps([path, params], sort_keys=True).encode()).hexdigest() + '.json')
    if f.exists():
        return json.loads(f.read_text(encoding='utf-8'))
    out, page = [], 1
    while True:
        q = {'serviceKey': _key(path), **params, 'numOfRows': rows, 'pageNo': page}
        items, total = _parse(_get(path, f'{BASE}/{path}?' + urllib.parse.urlencode(q), timeout, retries), path)
        out += items
        if not items or len(out) >= total:
            break
        page += 1
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    return out


def rh_trades(lawd_cd, ym):
    """연립다세대 매매 실거래. lawd_cd: 시군구 5자리, ym: 'YYYYMM'. 금액 단위는 만원."""
    return call('RTMSDataSvcRHTrade/getRTMSDataSvcRHTrade', {'LAWD_CD': lawd_cd, 'DEAL_YMD': ym}, rows=1000)


def _bld(op, bjdong_code, san, bun, ji):
    """건축물대장 오퍼레이션 공통 호출. 법정동코드 10자리를 시군구·법정동으로 나누고 본번·부번은 4자리로 채운다."""
    return call(f'BldRgstHubService/{op}', {
        'sigunguCd': bjdong_code[:5], 'bjdongCd': bjdong_code[5:], 'platGbCd': '1' if san else '0',
        'bun': f'{bun:04d}', 'ji': f'{ji:04d}'})


def building_title(bjdong_code, san, bun, ji):
    """건축물대장 표제부(getBrTitleInfo). 사용승인일(useAprDay), 주용도 등을 얻는다."""
    return _bld('getBrTitleInfo', bjdong_code, san, bun, ji)


def building_units(bjdong_code, san, bun, ji):
    """건축물대장 전유공용면적(getBrExposPubuseAreaInfo). 세대별 층·호·전유면적을 얻는다."""
    return _bld('getBrExposPubuseAreaInfo', bjdong_code, san, bun, ji)


def kakao(op, params, timeout=10, retries=3):
    """카카오 로컬 API(op: 'search/address', 'search/category') JSON 응답."""
    url = f'{KAKAO}/{op}.json?' + urllib.parse.urlencode(params)
    body = _get(f'kakao/{op}', url, timeout, retries, {'Authorization': 'KakaoAK ' + _key('kakao')})
    return json.loads(body)


def geocode(bjdong_code, san, bun, ji, address):
    """지번 주소 -> (경도, 위도). 법정동코드·산 여부·본번·부번이 모두 일치하는 결과만 인정하고 없으면 None."""
    docs = kakao('search/address', {'query': address, 'analyze_type': 'exact', 'size': 30})['documents']
    hits = {(d['x'], d['y']) for d in docs if d.get('address')
            and d['address'].get('b_code') == bjdong_code
            and (d['address'].get('mountain_yn') == 'Y') == san
            and d['address'].get('main_address_no') == str(bun)
            and (d['address'].get('sub_address_no') or '0') == str(ji)}
    if len(hits) != 1:  # 일치 없음 또는 서로 다른 좌표 여러 개 -> 임의 확정하지 않음
        return None
    x, y = hits.pop()
    return float(x), float(y)


def subway_stations(rect):
    """rect('경도1,위도1,경도2,위도2') 안의 지하철역 목록 [(역명, 경도, 위도)]. 최대 45쪽×15건."""
    out = []
    for page in range(1, 46):
        r = kakao('search/category', {'category_group_code': 'SW8', 'rect': rect, 'page': page, 'size': 15})
        out += [(d['place_name'], float(d['x']), float(d['y'])) for d in r['documents']]
        if r['meta']['is_end']:
            break
    return out
