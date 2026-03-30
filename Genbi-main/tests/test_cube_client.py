"""
Тесты для cube_client.py — execute_dax и decode_ssas_name.
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cube_client import execute_dax, decode_ssas_name


# ────────────────────────────────────────
# decode_ssas_name
# ────────────────────────────────────────

class TestDecodeSsasName:
    def test_plain_name(self):
        assert decode_ssas_name("Amount") == "Amount"

    def test_hex_code(self):
        # _x0020_ → пробел
        assert decode_ssas_name("Sales_x0020_Amount") == "Sales Amount"

    def test_bracket_extraction(self):
        assert decode_ssas_name("Table[Column]") == "Column"

    def test_nested_bracket(self):
        assert decode_ssas_name("[Продажи ед.]") == "Продажи ед."


# ────────────────────────────────────────
# execute_dax — HTTP ошибки
# ────────────────────────────────────────

XMLA_SUCCESS_BODY = """<?xml version="1.0"?>
<Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/">
  <Body>
    <ExecuteResponse xmlns="urn:schemas-microsoft-com:xml-analysis">
      <return>
        <root>
          <row><Col1>Москва</Col1><Col2>1000</Col2></row>
          <row><Col1>Питер</Col1><Col2>500</Col2></row>
        </root>
      </return>
    </ExecuteResponse>
  </Body>
</Envelope>"""

XMLA_ERROR_BODY = """<?xml version="1.0"?>
<Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/">
  <Body>
    <Fault><faultstring>err</faultstring>
    <detail><Error><Description>Column 'X' not found</Description></Error></detail>
    </Fault>
  </Body>
</Envelope>"""


class TestExecuteDax:
    """Тесты execute_dax с мокированием requests.post."""

    @patch('cube_client.requests.post')
    def test_http_401_returns_error_string(self, mock_post):
        resp = MagicMock()
        resp.status_code = 401
        resp.headers = {"Content-Type": "text/html; charset=windows-1251"}
        resp.text = "<!DOCTYPE html><html><body>Unauthorized</body></html>"
        resp.encoding = 'windows-1251'
        mock_post.return_value = resp

        result = execute_dax("EVALUATE {1}")
        assert isinstance(result, str)
        assert "ОШИБКА КУБА" in result
        assert "401" in result

    @patch('cube_client.requests.post')
    def test_http_500_returns_error_string(self, mock_post):
        resp = MagicMock()
        resp.status_code = 500
        resp.headers = {"Content-Type": "text/xml"}
        resp.text = "<error>internal</error>"
        resp.encoding = 'utf-8'
        mock_post.return_value = resp

        result = execute_dax("EVALUATE {1}")
        assert isinstance(result, str)
        assert "ОШИБКА КУБА" in result

    @patch('cube_client.requests.post')
    def test_html_at_200_returns_error(self, mock_post):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/html"}
        resp.text = "<!DOCTYPE html><html><body>Login page</body></html>"
        resp.encoding = 'utf-8'
        mock_post.return_value = resp

        result = execute_dax("EVALUATE {1}")
        assert isinstance(result, str)
        assert "ОШИБКА КУБА" in result

    @patch('cube_client.requests.post')
    def test_cube_error_description(self, mock_post):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/xml"}
        resp.text = XMLA_ERROR_BODY
        resp.encoding = 'utf-8'
        mock_post.return_value = resp

        result = execute_dax("EVALUATE bad query")
        assert isinstance(result, str)
        assert "Column 'X' not found" in result

    @patch('cube_client.requests.post')
    def test_successful_query_returns_dataframe(self, mock_post):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/xml"}
        resp.text = XMLA_SUCCESS_BODY
        resp.encoding = 'utf-8'
        mock_post.return_value = resp

        result = execute_dax("EVALUATE SUMMARIZECOLUMNS(...)")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "Col1" in result.columns

    @patch('cube_client.requests.post')
    def test_empty_result_returns_empty_df(self, mock_post):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/xml"}
        resp.text = """<?xml version="1.0"?><Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/"><Body><ExecuteResponse xmlns="urn:schemas-microsoft-com:xml-analysis"><return><root></root></return></ExecuteResponse></Body></Envelope>"""
        resp.encoding = 'utf-8'
        mock_post.return_value = resp

        result = execute_dax("EVALUATE FILTER(...)")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch('cube_client.requests.post')
    def test_connection_error(self, mock_post):
        mock_post.side_effect = Exception("Connection refused")

        result = execute_dax("EVALUATE {1}")
        assert isinstance(result, str)
        assert "КРИТИЧЕСКАЯ ОШИБКА" in result

    @patch('cube_client.requests.post')
    def test_numeric_coercion(self, mock_post):
        body = """<?xml version="1.0"?>
        <Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/">
          <Body><ExecuteResponse xmlns="urn:schemas-microsoft-com:xml-analysis">
            <return><root>
              <row><Name>A</Name><Value>123.45</Value></row>
              <row><Name>B</Name><Value>678.90</Value></row>
            </root></return>
          </ExecuteResponse></Body>
        </Envelope>"""
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/xml"}
        resp.text = body
        resp.encoding = 'utf-8'
        mock_post.return_value = resp

        result = execute_dax("EVALUATE ...")
        assert isinstance(result, pd.DataFrame)
        assert result["Value"].dtype in ('float64', 'int64')
