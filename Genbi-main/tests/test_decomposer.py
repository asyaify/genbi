"""
Тесты для decomposer_agent.py — декомпозиция пользовательских запросов.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from decomposer_agent import decompose_query, DECOMPOSER_PROMPT


# ────────────────────────────────────────
# Хелпер для мока LLM-ответа
# ────────────────────────────────────────

def _mock_llm_response(decomp_dict: dict):
    """Создаёт мок объекта OpenAI ChatCompletion."""
    msg = MagicMock()
    msg.content = json.dumps(decomp_dict, ensure_ascii=False)
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


EXPECTED_DECOMPOSITIONS = {
    "Продажи по видам номенклатуры": {
        "reflections": "Нужно показать продажи в разрезе видов номенклатуры",
        "metrics": ["Продажи (Реализация)"],
        "dimensions": ["Вид номенклатуры"],
        "filters": [],
        "time_context": "",
    },
    "Выручка в разрезе бизнес-регионов": {
        "reflections": "Выручка по бизнес-регионам",
        "metrics": ["Выручка (Реализация руб.)"],
        "dimensions": ["Бизнес Регион"],
        "filters": [],
        "time_context": "",
    },
    "Покажи остатки на складах в штуках": {
        "reflections": "Остатки в штуках по складам",
        "metrics": ["Остатки конечные ед"],
        "dimensions": ["Наименование склада"],
        "filters": [],
        "time_context": "",
    },
    "Топ-10 товаров по сумме реализации за 2022 год": {
        "reflections": "Топ-10 товаров по реализации за 2022",
        "metrics": ["Сумма реализации", "Топ-10"],
        "dimensions": ["Наименование номенклатуры"],
        "filters": [],
        "time_context": "2022 год",
    },
    "Выручка по месяцам только для контрагентов из Москвы": {
        "reflections": "Выручка помесячно, фильтр Москва",
        "metrics": ["Выручка (Реализация руб.)"],
        "dimensions": ["Месяц"],
        "filters": ["Бизнес Регион: Москва"],
        "time_context": "По месяцам",
    },
    'Остатки номенклатуры с видом "Винт" на текущий момент': {
        "reflections": "Остатки конечные для вида Винт",
        "metrics": ["Остатки Конечные ед"],
        "dimensions": ["Наименование номенклатуры"],
        "filters": ["Вид номенклатуры: Винт"],
        "time_context": "На текущий момент",
    },
    "Сравни продажи этого года с прошлым годом по месяцам": {
        "reflections": "Сравнение текущего и прошлого года помесячно",
        "metrics": ["Реализация руб.", "Реализация руб. (Прошлый год)"],
        "dimensions": ["Месяц"],
        "filters": [],
        "time_context": "Сравнение текущего и прошлого года по месяцам",
    },
}


class TestDecomposeQuery:
    """Проверяем, что decompose_query корректно вызывает LLM и парсит результат."""

    @pytest.mark.parametrize("query", list(EXPECTED_DECOMPOSITIONS.keys()))
    @patch('decomposer_agent.client')
    def test_decomposition_returns_valid_structure(self, mock_client, query):
        expected = EXPECTED_DECOMPOSITIONS[query]
        mock_client.chat.completions.create.return_value = _mock_llm_response(expected)

        result = decompose_query(query)

        assert isinstance(result, dict)
        assert "metrics" in result
        assert "dimensions" in result
        assert "filters" in result
        assert "time_context" in result
        assert len(result["metrics"]) > 0
        assert len(result["dimensions"]) > 0

    @pytest.mark.parametrize("query", list(EXPECTED_DECOMPOSITIONS.keys()))
    @patch('decomposer_agent.client')
    def test_decomposition_metrics_not_empty(self, mock_client, query):
        expected = EXPECTED_DECOMPOSITIONS[query]
        mock_client.chat.completions.create.return_value = _mock_llm_response(expected)

        result = decompose_query(query)
        assert len(result["metrics"]) >= 1, f"Для запроса '{query}' метрики пусты"

    @patch('decomposer_agent.client')
    def test_filter_query_has_filters(self, mock_client):
        query = 'Остатки номенклатуры с видом "Винт" на текущий момент'
        expected = EXPECTED_DECOMPOSITIONS[query]
        mock_client.chat.completions.create.return_value = _mock_llm_response(expected)

        result = decompose_query(query)
        assert len(result["filters"]) > 0, "Ожидался фильтр по виду номенклатуры"

    @patch('decomposer_agent.client')
    def test_time_query_has_time_context(self, mock_client):
        query = "Сравни продажи этого года с прошлым годом по месяцам"
        expected = EXPECTED_DECOMPOSITIONS[query]
        mock_client.chat.completions.create.return_value = _mock_llm_response(expected)

        result = decompose_query(query)
        assert result["time_context"], "Ожидался time_context для временного запроса"

    @patch('decomposer_agent.client')
    def test_llm_error_returns_empty_dict(self, mock_client):
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        result = decompose_query("Любой запрос")
        assert result == {}

    @patch('decomposer_agent.client')
    def test_invalid_json_returns_empty_dict(self, mock_client):
        msg = MagicMock()
        msg.content = "not valid json {"
        choice = MagicMock()
        choice.message = msg
        completion = MagicMock()
        completion.choices = [choice]
        mock_client.chat.completions.create.return_value = completion

        result = decompose_query("Сломанный ответ")
        assert result == {}


class TestDecomposerPrompt:
    """Проверяем, что системный промпт содержит критические секции."""

    def test_prompt_has_glossary(self):
        assert "ГЛОССАРИЙ" in DECOMPOSER_PROMPT

    def test_prompt_has_table_structure(self):
        assert "000 Календарь" in DECOMPOSER_PROMPT
        assert "005 Номенклатура" in DECOMPOSER_PROMPT

    def test_prompt_has_measures(self):
        assert "Реализация руб." in DECOMPOSER_PROMPT
        assert "Остатки" in DECOMPOSER_PROMPT

    def test_prompt_has_examples(self):
        assert "ПРИМЕРЫ" in DECOMPOSER_PROMPT
