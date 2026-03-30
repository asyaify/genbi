"""
Тесты для app.py — визуализация, PDF, агрегация, KPI.
"""
import pytest
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import (
    _clean_df_for_display,
    _aggregate_top_n,
    _format_value_axis,
    build_insights,
    build_executive_summary,
    build_chart,
    build_query_graph_dot,
    build_kpi_cards,
    dataframe_to_pdf_bytes,
    dataframe_to_excel_bytes,
    dataframe_to_word_bytes,
)


# ────────────────────────────────────────
# _clean_df_for_display
# ────────────────────────────────────────

class TestCleanDf:
    def test_nan_replaced(self):
        df = pd.DataFrame({"Регион": ["Москва", None, "Питер"], "Val": [1, 2, 3]})
        cleaned = _clean_df_for_display(df)
        assert "(не указано)" in cleaned["Регион"].values

    def test_empty_string_replaced(self):
        df = pd.DataFrame({"Регион": ["Москва", "", "Питер"], "Val": [1, 2, 3]})
        cleaned = _clean_df_for_display(df)
        assert "(не указано)" in cleaned["Регион"].values

    def test_numeric_not_touched(self):
        df = pd.DataFrame({"Val": [1.0, float("nan"), 3.0]})
        cleaned = _clean_df_for_display(df)
        assert pd.isna(cleaned["Val"].iloc[1])


# ────────────────────────────────────────
# _aggregate_top_n
# ────────────────────────────────────────

class TestAggregateTopN:
    def test_small_df_unchanged(self):
        df = pd.DataFrame({"Cat": ["A", "B", "C"], "Val": [10, 20, 30]})
        result = _aggregate_top_n(df, "Cat", "Val", n=5)
        assert len(result) == 3

    def test_large_df_aggregated(self):
        df = pd.DataFrame({
            "Cat": [f"Item_{i}" for i in range(50)],
            "Val": list(range(50, 0, -1)),
        })
        result = _aggregate_top_n(df, "Cat", "Val", n=10)
        assert len(result) == 11  # 10 top + 1 «Остальные»
        last_row = result.iloc[-1]
        assert "Остальные" in str(last_row["Cat"])

    def test_others_sum_correct(self):
        df = pd.DataFrame({
            "Cat": ["A", "B", "C", "D", "E"],
            "Val": [100, 80, 60, 40, 20],
        })
        result = _aggregate_top_n(df, "Cat", "Val", n=3)
        others_val = result[result["Cat"].str.contains("Остальные")]["Val"].iloc[0]
        assert others_val == 60  # 40 + 20


# ────────────────────────────────────────
# _format_value_axis
# ────────────────────────────────────────

class TestFormatValueAxis:
    def test_billions(self):
        assert "млрд" in _format_value_axis(2_671_840_942.75)

    def test_millions(self):
        assert "млн" in _format_value_axis(5_375_937.51)

    def test_thousands(self):
        assert "тыс" in _format_value_axis(1500.0)

    def test_small(self):
        result = _format_value_axis(42.0)
        assert "42" in result


# ────────────────────────────────────────
# build_chart
# ────────────────────────────────────────

class TestBuildChart:
    def test_bar_chart_with_many_categories(self):
        df = pd.DataFrame({
            "Region": [f"Region_{i}" for i in range(100)],
            "Revenue": list(range(100, 0, -1)),
        })
        fig = build_chart(df, "bar")
        assert fig is not None

    def test_line_chart(self):
        df = pd.DataFrame({
            "Month": ["Янв", "Фев", "Мар"],
            "Revenue": [100, 200, 150],
        })
        fig = build_chart(df, "line")
        assert fig is not None

    def test_table_returns_none(self):
        df = pd.DataFrame({"A": [1], "B": [2]})
        assert build_chart(df, "table") is None

    def test_empty_df_returns_none(self):
        assert build_chart(pd.DataFrame(), "bar") is None

    def test_bar_horizontal(self):
        df = pd.DataFrame({"Cat": ["A", "B"], "Val": [10, 20]})
        fig = build_chart(df, "bar")
        # Проверяем что бары горизонтальные
        assert fig.data[0].orientation == "h"


# ────────────────────────────────────────
# build_kpi_cards
# ────────────────────────────────────────

class TestBuildKpiCards:
    def test_basic(self):
        df = pd.DataFrame({"Region": ["A", "B"], "Revenue": [1000, 2000]})
        cards = build_kpi_cards(df)
        assert len(cards) >= 1
        assert any("Строк" in c["label"] for c in cards)

    def test_empty_df(self):
        cards = build_kpi_cards(pd.DataFrame())
        assert cards == []

    def test_numeric_formatting(self):
        df = pd.DataFrame({"X": ["A"], "Revenue": [5_000_000]})
        cards = build_kpi_cards(df)
        # Должен быть отформатирован в млн
        revenue_cards = [c for c in cards if "Revenue" in c["label"]]
        assert any("млн" in c["value"] for c in revenue_cards)


# ────────────────────────────────────────
# build_executive_summary
# ────────────────────────────────────────

class TestBuildExecutiveSummary:
    def test_no_nan_in_output(self):
        df = pd.DataFrame({"Region": ["Москва", None, "Питер"], "Val": [100, 50, 30]})
        summary = build_executive_summary(df)
        assert "nan" not in summary.lower()

    def test_formats_large_numbers(self):
        df = pd.DataFrame({"Region": ["A"], "Val": [2_000_000]})
        summary = build_executive_summary(df)
        assert "млн" in summary


# ────────────────────────────────────────
# PDF кириллица
# ────────────────────────────────────────

class TestPdfExport:
    def test_pdf_cyrillic_no_error(self):
        df = pd.DataFrame({"Регион": ["Москва", "Питер"], "Выручка": [1000, 2000]})
        pdf_bytes = dataframe_to_pdf_bytes(df, "Тестовый запрос кириллица", "Сводка по данным")
        assert len(pdf_bytes) > 100
        assert pdf_bytes[:5] == b"%PDF-"

    def test_pdf_contains_data(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        pdf_bytes = dataframe_to_pdf_bytes(df, "test", "summary")
        assert len(pdf_bytes) > 0


# ────────────────────────────────────────
# Graphviz
# ────────────────────────────────────────

class TestGraphviz:
    def test_dot_output(self):
        response = {
            "chart_type": "bar",
            "decomposition": {
                "metrics": ["Реализация руб."],
                "dimensions": ["Бизнес Регион"],
                "filters": [],
                "time_context": "2024 год",
            },
        }
        dot = build_query_graph_dot(response)
        assert "digraph" in dot
        assert "Реализация" in dot
        assert "Результат" in dot
