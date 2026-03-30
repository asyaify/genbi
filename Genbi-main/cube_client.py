import logging
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import re
import warnings

import os
from dotenv import load_dotenv

load_dotenv(interpolate=False)
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

CUBE_URL = os.getenv('CUBE_URL')
USERNAME = os.getenv('CUBE_USERNAME') or os.getenv('USERNAME')
PASSWORD = os.getenv('CUBE_PASSWORD') or os.getenv('PASSWORD')
CATALOG = os.getenv('CATALOG')
AUTH_METHOD = os.getenv('AUTH_METHOD', 'auto').lower()  # 'basic', 'ntlm', 'auto'

# Попытка импорта NTLM (опционально)
try:
    from requests_ntlm import HttpNtlmAuth
    _NTLM_AVAILABLE = True
except ImportError:
    _NTLM_AVAILABLE = False


def _build_auth():
    """Выбирает метод аутентификации: NTLM, Basic или auto (NTLM → Basic fallback)."""
    if AUTH_METHOD == 'ntlm':
        if not _NTLM_AVAILABLE:
            logger.warning("AUTH_METHOD=ntlm, но requests-ntlm не установлен. pip install requests-ntlm")
            return HTTPBasicAuth(USERNAME, PASSWORD)
        return HttpNtlmAuth(USERNAME, PASSWORD)
    elif AUTH_METHOD == 'basic':
        return HTTPBasicAuth(USERNAME, PASSWORD)
    else:
        # auto: пробуем NTLM если доступен, иначе Basic
        if _NTLM_AVAILABLE:
            return HttpNtlmAuth(USERNAME, PASSWORD)
        return HTTPBasicAuth(USERNAME, PASSWORD)

def decode_ssas_name(name: str) -> str:
    """Декодирует HEX-коды и очищает имена колонок"""
    def replace_hex(match):
        return chr(int(match.group(1), 16))
    
    decoded = re.sub(r'_x([0-9a-fA-F]{4})_', replace_hex, name)
    if '[' in decoded and ']' in decoded:
        match = re.search(r'\[([^\]]+)\]$', decoded)
        if match: 
            return match.group(1)
    return decoded

def execute_dax(dax_query: str) -> pd.DataFrame:
    xmla_payload = f"""<Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/">
      <Body>
        <Execute xmlns="urn:schemas-microsoft-com:xml-analysis">
          <Command><Statement><![CDATA[{dax_query}]]></Statement></Command>
          <Properties><PropertyList><Catalog>{CATALOG}</Catalog><Format>Tabular</Format></PropertyList></Properties>
        </Execute>
      </Body>
    </Envelope>"""

    headers = {'Content-Type': 'text/xml; charset=utf-8'}
    
    try:
        auth = _build_auth()
        response = requests.post(CUBE_URL, data=xmla_payload.encode('utf-8'), 
                                 headers=headers, auth=auth, timeout=30)
        if "windows-1251" in (response.headers.get("Content-Type", "").lower()):
            response.encoding = 'windows-1251'
        else:
            response.encoding = 'utf-8'
        raw_xml = response.text

        if response.status_code == 401:
            auth_headers = response.headers.get("WWW-Authenticate", "")
            hint = ""
            if "NTLM" in auth_headers and not _NTLM_AVAILABLE:
                hint = " Сервер требует NTLM. Установите: pip install requests-ntlm"
            elif "NTLM" in auth_headers and AUTH_METHOD == 'basic':
                hint = " Сервер требует NTLM. Задайте AUTH_METHOD=ntlm в .env"
            else:
                hint = " Проверьте CUBE_USERNAME и CUBE_PASSWORD в .env"
            return f"ОШИБКА КУБА: HTTP 401 — Ошибка аутентификации.{hint}"

        if response.status_code >= 400:
            snippet = re.sub(r'\s+', ' ', raw_xml)[:240]
            return f"ОШИБКА КУБА: HTTP {response.status_code}. {snippet}"

        # Некоторые IIS/прокси ошибки возвращаются в виде HTML при статусе 200.
        if raw_xml.lstrip().lower().startswith("<!doctype html") or "<html" in raw_xml[:300].lower():
            snippet = re.sub(r'\s+', ' ', raw_xml)[:240]
            return f"ОШИБКА КУБА: получен HTML вместо XMLA. {snippet}"

        # ПРОВЕРКА НА ОШИБКУ КУБА
        if "<Description>" in raw_xml:
            error_msg = re.search(r'<Description>(.*?)</Description>', raw_xml)
            if error_msg:
                return f"ОШИБКА КУБА: {error_msg.group(1)}"

        # ПАРСИНГ СТРОК
        rows = re.findall(r'<[a-z0-9]*:?row>(.*?)</[a-z0-9]*:?row>', raw_xml, flags=re.DOTALL)
        if not rows:
            return pd.DataFrame()

        all_data = []
        for row_content in rows:
            cols = re.findall(r'<([^>]+)>(.*?)</\1>', row_content, flags=re.DOTALL)
            row_dict = {decode_ssas_name(tag): val for tag, val in cols}
            all_data.append(row_dict)

        df = pd.DataFrame(all_data)
        
        # Безопасное приведение к числам (без FutureWarning)
        for col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce')
            if not series.isna().all():
                df[col] = series
            
        return df
    except Exception as e:
        return f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}"