"""
Тесты для genbi_agent.py — оркестратор GenBI (RAG + генерация DAX + исполнение).
Полный end-to-end pipeline с моками: decompose → retrieve → generate → execute.
"""
import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ────────────────────────────────────────
# Хелперы
# ────────────────────────────────────────

def _make_llm_response(data: dict):
    msg = MagicMock()
    msg.content = json.dumps(data, ensure_ascii=False)
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _make_decomp(metrics, dimensions, filters=None, time_context=""):
    return {
        "reflections": "test",
        "metrics": metrics,
        "dimensions": dimensions,
        "filters": filters or [],
        "time_context": time_context,
    }


def _make_dax_response(dax: str, chart_type: str = "bar"):
    return {
        "thought": "Рассуждение",
        "dax": dax,
        "chart_type": chart_type,
    }


# ────────────────────────────────────────
# Тестовые запросы (7 базовых)
# ────────────────────────────────────────

TEST_QUERIES = [
    {
        "query": "Продажи по видам номенклатуры",
        "decomp": _make_decomp(
            metrics=["Продажи ед."],
            dimensions=["Вид номенклатуры"],
        ),
        "dax": "EVALUATE TOPN(1000, SUMMARIZECOLUMNS('005 Номенклатура'[Вид номенклатуры], \"Продажи\", [Продажи ед.]))",
        "chart_type": "bar",
        "df_data": {"Вид номенклатуры": ["Болт", "Гайка", "Винт"], "Продажи": [100, 200, 150]},
    },
    {
        "query": "Выручка в разрезе бизнес-регионов",
        "decomp": _make_decomp(
            metrics=["Реализация руб."],
            dimensions=["Бизнес Регион"],
        ),
        "dax": "EVALUATE TOPN(500, SUMMARIZECOLUMNS('002 Контрагенты'[Бизнес Регион], \"Выручка\", [Реализация руб.]))",
        "chart_type": "bar",
        "df_data": {"Бизнес Регион": ["Москва", "Питер", "Урал"], "Выручка": [5000, 3000, 2000]},
    },
    {
        "query": "Покажи остатки на складах в штуках",
        "decomp": _make_decomp(
            metrics=["Остатки Конечные ед"],
            dimensions=["Наименование склада"],
        ),
        "dax": "EVALUATE TOPN(1000, SUMMARIZECOLUMNS('006 Склады'[Наименование склада], \"Остатки\", [Остатки Конечные ед]))",
        "chart_type": "bar",
        "df_data": {"Наименование склада": ["Склад 1", "Склад 2"], "Остатки": [500, 300]},
    },
    {
        "query": "Топ-10 товаров по сумме реализации за 2022 год",
        "decomp": _make_decomp(
            metrics=["Реализация руб."],
            dimensions=["Наименование номенклатуры"],
            time_context="2022 год",
        ),
        "dax": "EVALUATE TOPN(10, SUMMARIZECOLUMNS('005 Номенклатура'[Наименование], FILTER(VALUES('000 Календарь'[Год]), '000 Календарь'[Год] = 2022), \"Реализация\", [Реализация руб.]), [Реализация руб.], DESC)",
        "chart_type": "bar",
        "df_data": {"Наименование": [f"Товар {i}" for i in range(10)], "Реализация": list(range(1000, 0, -100))},
    },
    {
        "query": "Выручка по месяцам только для контрагентов из Москвы",
        "decomp": _make_decomp(
            metrics=["Реализация руб."],
            dimensions=["Месяц"],
            filters=["Бизнес Регион: Москва"],
            time_context="По месяцам",
        ),
        "dax": "EVALUATE SUMMARIZECOLUMNS('000 Календарь'[Месяц], FILTER(VALUES('002 Контрагенты'[Бизнес Регион]), '002 Контрагенты'[Бизнес Регион] = \"Москва\"), \"Выручка\", [Реализация руб.])",
        "chart_type": "line",
        "df_data": {"Месяц": ["Январь", "Февраль", "Март"], "Выручка": [1000, 1200, 1100]},
    },
    {
        "query": 'Остатки номенклатуры с видом "Винт" на текущий момент',
        "decomp": _make_decomp(
            metrics=["Остатки Конечные ед"],
            dimensions=["Наименование номенклатуры"],
            filters=["Вид номенклатуры: Винт"],
            time_context="На текущий момент",
        ),
        "dax": "EVALUATE SUMMARIZECOLUMNS('005 Номенклатура'[Наименование], FILTER(VALUES('005 Номенклатура'[Вид номенклатуры]), '005 Номенклатура'[Вид номенклатуры] = \"Винт\"), \"Остатки\", [Остатки Конечные ед])",
        "chart_type": "table",
        "df_data": {"Наименование": ["Винт М6", "Винт М8"], "Остатки": [50, 80]},
    },
    {
        "query": "Сравни продажи этого года с прошлым годом по месяцам",
        "decomp": _make_decomp(
            metrics=["Реализация руб.", "Реализация руб. (Прошлый год)"],
            dimensions=["Месяц"],
            time_context="Сравнение текущего и прошлого года по месяцам",
        ),
        "dax": "EVALUATE SUMMARIZECOLUMNS('000 Календарь'[Месяц Номер], '000 Календарь'[Месяц], FILTER('000 Календарь', '000 Календарь'[Год] = 2022), \"Текущий год\", [Реализация руб.], \"Прошлый год\", [Реализация руб. (Прошлый год)]) ORDER BY '000 Календарь'[Месяц Номер] ASC",
        "chart_type": "line",
        "df_data": {
            "Месяц Номер": [1, 2, 3],
            "Месяц": ["Январь", "Февраль", "Март"],
            "Текущий год": [1000, 1200, 1100],
            "Прошлый год": [900, 1000, 1050],
        },
    },
]


# ────────────────────────────────────────
# Тесты end-to-end pipeline
# ────────────────────────────────────────

class TestGenBIOrchestrator:
    """
    Полный workflow: decompose → qdrant retrieve → LLM generate → execute_dax.
    Все внешние зависимости мокируются.
    """

    @pytest.fixture(autouse=True)
    def _setup_orchestrator(self):
        """Импортируем оркестратор с мокированными зависимостями."""
        # Мокируем qdrant перед импортом genbi_agent
        with patch('genbi_agent.qdrant') as mock_qdrant, \
             patch('genbi_agent.client') as mock_llm:
            # Qdrant коллекции существуют
            mock_qdrant.get_collection.return_value = MagicMock()
            # Qdrant query_points возвращает результаты
            mock_point = MagicMock()
            mock_point.payload = {
                "name": "Реализация руб.",
                "table_name": "М001 Движения",
                "folder": "Продажи",
            }
            mock_result = MagicMock()
            mock_result.points = [mock_point]
            mock_qdrant.query_points.return_value = mock_result

            # Embedding
            emb_data = MagicMock()
            emb_data.embedding = [0.1] * 1536
            emb_resp = MagicMock()
            emb_resp.data = [emb_data]
            mock_llm.embeddings.create.return_value = emb_resp

            self.mock_qdrant = mock_qdrant
            self.mock_llm = mock_llm
            yield

    @pytest.mark.parametrize("test_case", TEST_QUERIES, ids=[tc["query"] for tc in TEST_QUERIES])
    @patch('genbi_agent.execute_dax')
    @patch('genbi_agent.decompose_query')
    @patch('genbi_agent.client')
    def test_full_pipeline_success(self, mock_llm, mock_decompose, mock_exec_dax, test_case):
        """Для каждого из 7 запросов: pipeline возвращает DataFrame без ошибок."""
        # 1. decompose
        mock_decompose.return_value = test_case["decomp"]

        # 2. LLM generate DAX
        dax_resp = _make_dax_response(test_case["dax"], test_case["chart_type"])
        mock_llm.chat.completions.create.return_value = _make_llm_response(dax_resp)

        # Embedding для RAG
        emb_data = MagicMock()
        emb_data.embedding = [0.1] * 1536
        emb_resp = MagicMock()
        emb_resp.data = [emb_data]
        mock_llm.embeddings.create.return_value = emb_resp

        # 3. execute_dax → DataFrame
        df = pd.DataFrame(test_case["df_data"])
        mock_exec_dax.return_value = df

        # Импортируем и вызываем
        from genbi_agent import GenBIOrchestrator
        orch = GenBIOrchestrator()
        result = orch.ask(test_case["query"])

        assert "error" not in result, f"Ошибка: {result.get('error')}"
        assert "df" in result
        assert isinstance(result["df"], pd.DataFrame)
        assert not result["df"].empty
        assert "dax" in result
        assert result["dax"]

    @pytest.mark.parametrize("test_case", TEST_QUERIES, ids=[tc["query"] for tc in TEST_QUERIES])
    @patch('genbi_agent.execute_dax')
    @patch('genbi_agent.decompose_query')
    @patch('genbi_agent.client')
    def test_dax_query_structure(self, mock_llm, mock_decompose, mock_exec_dax, test_case):
        """Проверяем, что сгенерированный DAX содержит EVALUATE."""
        mock_decompose.return_value = test_case["decomp"]

        dax_resp = _make_dax_response(test_case["dax"], test_case["chart_type"])
        mock_llm.chat.completions.create.return_value = _make_llm_response(dax_resp)

        emb_data = MagicMock()
        emb_data.embedding = [0.1] * 1536
        emb_resp = MagicMock()
        emb_resp.data = [emb_data]
        mock_llm.embeddings.create.return_value = emb_resp

        df = pd.DataFrame(test_case["df_data"])
        mock_exec_dax.return_value = df

        from genbi_agent import GenBIOrchestrator
        orch = GenBIOrchestrator()
        result = orch.ask(test_case["query"])

        dax = result.get("dax", "")
        assert "EVALUATE" in dax, f"DAX должен начинаться с EVALUATE: {dax}"

    @patch('genbi_agent.execute_dax')
    @patch('genbi_agent.decompose_query')
    @patch('genbi_agent.client')
    def test_retry_on_cube_error(self, mock_llm, mock_decompose, mock_exec_dax):
        """При ошибке куба оркестратор делает retry с самокоррекцией."""
        mock_decompose.return_value = _make_decomp(["Продажи ед."], ["Вид номенклатуры"])

        # Первый вызов LLM — с ошибкой DAX, второй — исправленный
        bad_dax = _make_dax_response("EVALUATE BAD_QUERY()")
        good_dax = _make_dax_response("EVALUATE TOPN(1000, SUMMARIZECOLUMNS('005 Номенклатура'[Вид номенклатуры], \"Продажи\", [Продажи ед.]))")
        mock_llm.chat.completions.create.side_effect = [
            _make_llm_response(bad_dax),
            _make_llm_response(good_dax),
        ]

        emb_data = MagicMock()
        emb_data.embedding = [0.1] * 1536
        emb_resp = MagicMock()
        emb_resp.data = [emb_data]
        mock_llm.embeddings.create.return_value = emb_resp

        # Сначала _detect_data_years (init), потом ошибка, потом успех
        df_ok = pd.DataFrame({"Вид": ["Болт"], "Продажи": [100]})
        df_years = pd.DataFrame({"Год": [2022], "val": [1000.0]})
        mock_exec_dax.side_effect = [
            df_years,  # _detect_data_years в __init__
            "ОШИБКА КУБА: Unknown function BAD_QUERY",
            df_ok,
        ]

        from genbi_agent import GenBIOrchestrator
        orch = GenBIOrchestrator()
        result = orch.ask("Продажи по видам номенклатуры")

        assert "error" not in result
        assert isinstance(result["df"], pd.DataFrame)
        assert mock_llm.chat.completions.create.call_count == 2

    @patch('genbi_agent.execute_dax')
    @patch('genbi_agent.decompose_query')
    @patch('genbi_agent.client')
    def test_max_retries_exhausted(self, mock_llm, mock_decompose, mock_exec_dax):
        """После max_retries попыток возвращает ошибку."""
        mock_decompose.return_value = _make_decomp(["X"], ["Y"])

        bad_dax = _make_dax_response("EVALUATE BAD()")
        mock_llm.chat.completions.create.return_value = _make_llm_response(bad_dax)

        emb_data = MagicMock()
        emb_data.embedding = [0.1] * 1536
        emb_resp = MagicMock()
        emb_resp.data = [emb_data]
        mock_llm.embeddings.create.return_value = emb_resp

        mock_exec_dax.return_value = "ОШИБКА КУБА: syntax error"

        from genbi_agent import GenBIOrchestrator
        orch = GenBIOrchestrator()
        result = orch.ask("test")

        assert "error" in result
        assert "Не удалось исправить ошибку" in result["error"]

    @patch('genbi_agent.decompose_query')
    def test_decomposition_failure(self, mock_decompose):
        """Если декомпозиция провалилась, возвращается ошибка."""
        mock_decompose.return_value = {}

        from genbi_agent import GenBIOrchestrator
        orch = GenBIOrchestrator()
        result = orch.ask("бессмысленный запрос")

        assert "error" in result

    @patch('genbi_agent.execute_dax')
    @patch('genbi_agent.decompose_query')
    @patch('genbi_agent.client')
    def test_empty_result_returns_warning(self, mock_llm, mock_decompose, mock_exec_dax):
        """Если куб вернул 0 строк, получаем warning."""
        mock_decompose.return_value = _make_decomp(["X"], ["Y"])

        dax_resp = _make_dax_response("EVALUATE SUMMARIZECOLUMNS(...)")
        mock_llm.chat.completions.create.return_value = _make_llm_response(dax_resp)

        emb_data = MagicMock()
        emb_data.embedding = [0.1] * 1536
        emb_resp = MagicMock()
        emb_resp.data = [emb_data]
        mock_llm.embeddings.create.return_value = emb_resp

        mock_exec_dax.return_value = pd.DataFrame()

        from genbi_agent import GenBIOrchestrator
        orch = GenBIOrchestrator()
        result = orch.ask("test")

        assert "warning" in result
        assert "0 строк" in result["warning"]


class TestValidateAndFixDax:
    """Тесты авто-корректора мер в DAX."""

    def test_typo_autocorrect(self):
        from genbi_agent import GenBIOrchestrator
        orch = GenBIOrchestrator()
        orch.all_measure_names = ["Реализация руб.", "Продажи ед."]

        dax = 'EVALUATE SUMMARIZECOLUMNS("X", [Реализация руб])'
        fixed = orch.validate_and_fix_dax(dax)
        assert "[Реализация руб.]" in fixed

    def test_no_change_for_correct_measure(self):
        from genbi_agent import GenBIOrchestrator
        orch = GenBIOrchestrator()
        orch.all_measure_names = ["Реализация руб."]

        dax = 'EVALUATE SUMMARIZECOLUMNS("X", [Реализация руб.])'
        fixed = orch.validate_and_fix_dax(dax)
        assert fixed == dax


class TestQdrantCollectionCheck:
    """Проверяем check_qdrant_collections."""

    @patch('genbi_agent.qdrant')
    def test_all_collections_present(self, mock_qdrant):
        mock_qdrant.get_collection.return_value = MagicMock()

        from genbi_agent import GenBIOrchestrator
        orch = GenBIOrchestrator()
        ready, missing = orch.check_qdrant_collections()
        assert ready is True
        assert missing == []

    @patch('genbi_agent.qdrant')
    def test_missing_collections(self, mock_qdrant):
        mock_qdrant.get_collection.side_effect = Exception("not found")

        from genbi_agent import GenBIOrchestrator
        orch = GenBIOrchestrator()
        ready, missing = orch.check_qdrant_collections()
        assert ready is False
        assert "genbi_measures" in missing
        assert "genbi_columns" in missing
