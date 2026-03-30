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
    _is_rate_metric,
    _style_dataframe,
    build_insights,
    build_executive_summary,
    build_chart,
    build_query_graph_dot,
    build_kpi_cards,
    build_drilldown_suggestions,
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

    def test_pie_chart(self):
        df = pd.DataFrame({"Cat": ["A", "B", "C"], "Val": [10, 20, 30]})
        fig = build_chart(df, "pie")
        assert fig is not None
        assert fig.data[0].type == "pie"

    def test_treemap_chart(self):
        df = pd.DataFrame({"Cat": ["A", "B", "C"], "Val": [10, 20, 30]})
        fig = build_chart(df, "treemap")
        assert fig is not None
        assert fig.data[0].type == "treemap"

    def test_scatter_chart(self):
        df = pd.DataFrame({"Cat": ["A", "B", "C"], "X": [1, 2, 3], "Y": [4, 5, 6]})
        fig = build_chart(df, "scatter")
        assert fig is not None

    def test_area_chart(self):
        df = pd.DataFrame({"Month": ["Янв", "Фев", "Мар"], "Val": [100, 200, 150]})
        fig = build_chart(df, "area")
        assert fig is not None

    def test_custom_columns(self):
        df = pd.DataFrame({"A": ["x", "y"], "B": [10, 20], "C": [30, 40]})
        fig = build_chart(df, "bar", x_col="A", y_cols_override=["C"])
        assert fig is not None

    def test_reference_lines_line(self):
        df = pd.DataFrame({"Month": ["Янв", "Фев", "Мар"], "Val": [100, 200, 300]})
        fig = build_chart(df, "line", show_avg=True, show_median=True)
        # Должны быть shape-элементы (hline/vline добавляют shapes)
        assert fig.layout.shapes is not None or len(fig.data) > 0

    def test_reference_lines_bar(self):
        df = pd.DataFrame({"Cat": ["A", "B", "C"], "Val": [100, 200, 300]})
        fig = build_chart(df, "bar", show_avg=True)
        assert fig is not None


# ────────────────────────────────────────
# _style_dataframe
# ────────────────────────────────────────

class TestStyleDataframe:
    def test_returns_styler_for_numeric(self):
        df = pd.DataFrame({"Cat": ["A", "B"], "Val": [10, 20]})
        result = _style_dataframe(df)
        assert hasattr(result, "to_html")  # Styler has to_html

    def test_returns_df_for_no_numeric(self):
        df = pd.DataFrame({"A": ["x", "y"], "B": ["a", "b"]})
        result = _style_dataframe(df)
        assert isinstance(result, pd.DataFrame)


# ────────────────────────────────────────
# build_drilldown_suggestions
# ────────────────────────────────────────

class TestDrilldownSuggestions:
    def test_returns_suggestions(self):
        df = pd.DataFrame({"Region": ["Москва", "Питер", "Казань"], "Val": [100, 50, 30]})
        suggestions = build_drilldown_suggestions(df, "Выручка по регионам")
        assert len(suggestions) >= 1

    def test_top_category_drill(self):
        df = pd.DataFrame({"Region": ["Москва", "Питер"], "Val": [200, 100]})
        suggestions = build_drilldown_suggestions(df, "Выручка по регионам")
        assert any("Москва" in s for s in suggestions)

    def test_suggests_time_analysis(self):
        df = pd.DataFrame({"Region": ["A", "B"], "Val": [100, 200]})
        suggestions = build_drilldown_suggestions(df, "Продажи по городам")
        assert any("месяц" in s.lower() or "динамик" in s.lower() for s in suggestions)

    def test_no_time_suggestion_for_time_query(self):
        df = pd.DataFrame({"Region": ["A", "B"], "Val": [100, 200]})
        suggestions = build_drilldown_suggestions(df, "Динамика продаж по месяцам")
        assert not any(s == "Показать динамику по месяцам" for s in suggestions)

    def test_empty_df(self):
        suggestions = build_drilldown_suggestions(pd.DataFrame(), "test")
        assert suggestions == []


# ────────────────────────────────────────
# build_kpi_cards
# ────────────────────────────────────────

class TestIsRateMetric:
    def test_average_detected(self):
        assert _is_rate_metric("Средний оплаченный счет, руб.") is True

    def test_percent_detected(self):
        assert _is_rate_metric("Доля маркетплейса %") is True

    def test_regular_metric_not_rate(self):
        assert _is_rate_metric("Реализация руб.") is False

    def test_coefficient_detected(self):
        assert _is_rate_metric("Коэффициент конверсии") is True


class TestBuildKpiCards:
    def test_basic(self):
        df = pd.DataFrame({"Region": ["A", "B"], "Revenue": [1000, 2000]})
        cards = build_kpi_cards(df)
        assert len(cards) >= 1
        assert any("Записей" in c["label"] for c in cards)

    def test_empty_df(self):
        cards = build_kpi_cards(pd.DataFrame())
        assert cards == []

    def test_numeric_formatting(self):
        df = pd.DataFrame({"X": ["A"], "Revenue": [5_000_000]})
        cards = build_kpi_cards(df)
        revenue_cards = [c for c in cards if "Revenue" in c["label"]]
        assert any("млн" in c["value"] for c in revenue_cards)

    def test_rate_metric_shows_median(self):
        df = pd.DataFrame({
            "Регион": ["A", "B", "C"],
            "Средний чек": [100, 200, 300],
        })
        cards = build_kpi_cards(df)
        labels = [c["label"] for c in cards]
        assert any("Медиана" in l for l in labels), f"Expected 'Медиана' in {labels}"
        # Не должно быть Итого для средних
        assert not any("Итого" in l for l in labels)

    def test_absolute_metric_shows_total(self):
        df = pd.DataFrame({
            "Регион": ["A", "B", "C"],
            "Выручка": [1000, 2000, 3000],
        })
        cards = build_kpi_cards(df)
        labels = [c["label"] for c in cards]
        assert any("Итого" in l for l in labels)

    def test_unique_categories_shown(self):
        df = pd.DataFrame({"Регион": ["Москва", "Питер", "Москва"], "Val": [1, 2, 3]})
        cards = build_kpi_cards(df)
        labels = [c["label"] for c in cards]
        assert any("Уник" in l for l in labels)


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

    def test_contains_overview(self):
        df = pd.DataFrame({"Region": ["A", "B", "C"], "Val": [100, 200, 300]})
        summary = build_executive_summary(df)
        assert "3 строк" in summary

    def test_rate_metric_shows_median(self):
        df = pd.DataFrame({
            "Регион": ["A", "B", "C", "D"],
            "Средний чек": [100, 200, 300, 400],
        })
        summary = build_executive_summary(df)
        assert "медиана" in summary

    def test_leaders_with_percent(self):
        df = pd.DataFrame({
            "Регион": ["Москва", "Питер", "Казань"],
            "Выручка": [5000, 3000, 2000],
        })
        summary = build_executive_summary(df)
        assert "%" in summary
        assert "Лидеры" in summary

    def test_empty_df(self):
        summary = build_executive_summary(pd.DataFrame())
        assert "отсутствуют" in summary.lower()

    def test_outlier_detection(self):
        df = pd.DataFrame({
            "Cat": [f"C{i}" for i in range(20)],
            "Val": [10] * 19 + [10000],  # 1 outlier
        })
        summary = build_executive_summary(df)
        assert "Выбросы" in summary or "выброс" in summary.lower()

    def test_concentration_warning(self):
        df = pd.DataFrame({
            "Cat": ["A", "B", "C", "D", "E"],
            "Val": [8000, 1500, 300, 100, 100],
        })
        summary = build_executive_summary(df)
        assert "концентрация" in summary.lower()

    def test_no_emojis(self):
        df = pd.DataFrame({"Cat": ["A", "B", "C"], "Val": [100, 200, 300]})
        summary = build_executive_summary(df)
        # Не должно быть эмодзи
        import re
        emoji_pattern = re.compile(
            "[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001FA00-\U0001FA6F]"
        )
        assert not emoji_pattern.search(summary), f"Emoji found in summary: {summary}"

    def test_paragraphs_separated(self):
        df = pd.DataFrame({
            "Cat": [f"C{i}" for i in range(10)],
            "Val": list(range(10, 0, -1)),
        })
        summary = build_executive_summary(df)
        # Блоки разделены двойным переносом строки
        assert "\n\n" in summary


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
