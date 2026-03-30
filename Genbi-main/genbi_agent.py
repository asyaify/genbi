import json
import re
import difflib
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime

from cube_client import execute_dax
from decomposer_agent import decompose_query

import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv(interpolate=False)


API_KEY = os.getenv('API_KEY') or os.getenv('SECRET_KEY')
BASE_URL = "https://api.aitunnel.ru/v1/"
MODEL_NAME = "gemini-3.1-flash-lite-preview"
EMBEDDING_MODEL = "text-embedding-3-small"
BIM_FILE_PATH = "OLAP_MK.bim"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
qdrant = QdrantClient(url="http://localhost:6333")

class OrcehestratorOutput(BaseModel):
    thought: str = Field(..., description="Размышления")
    dax: str = Field(..., description="Сгенерированный DAX запрос")
    chart_type: Literal["bar", "line", "pie", "treemap", "scatter", "area", "table"] = Field(..., description="Вид ответа")


class GenBIOrchestrator:
    def __init__(self):
        self.full_schema = {}  # table_name -> list of columns
        self.all_measure_names =[] # Для авто-корректора
        self.data_year_min = None
        self.data_year_max = None
        self.load_local_schema()
        self._detect_data_years()

    def load_local_schema(self):
        """Загружает полный .bim файл в память для сборки контекста (Schema Linking)"""
        try:
            with open(BIM_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for table in data.get('model', {}).get('tables',[]):
                if table.get('isHidden'): continue
                t_name = table.get('name')
                
                # Сохраняем столбцы
                cols =[c.get('name') for c in table.get('columns', []) if not c.get('isHidden')]
                self.full_schema[t_name] = cols
                
                # Сохраняем имена мер для валидатора
                for m in table.get('measures',[]):
                    if not m.get('isHidden'):
                        self.all_measure_names.append(m.get('name'))
                        
            logging.info(f"Локальная схема загружена: {len(self.full_schema)} таблиц.")
        except Exception as e:
            logging.error(f"Ошибка загрузки схемы: {e}")

    def _get_embedding(self, text: str) -> list:
        if not text: 
            return[]
        res = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return res.data[0].embedding

    def _detect_data_years(self):
        """Определяет актуальный диапазон лет данных в кубе."""
        try:
            df = execute_dax(
                "EVALUATE SUMMARIZECOLUMNS('000 Календарь'[Год], "
                '"val", [Реализация руб.])'
            )
            if hasattr(df, 'shape') and not df.empty:
                years = sorted(df['Год'].dropna().astype(int).tolist())
                if years:
                    self.data_year_min = years[0]
                    self.data_year_max = years[-1]
                    logging.info(f"Диапазон данных: {self.data_year_min}–{self.data_year_max}")
        except Exception as e:
            logging.warning(f"Не удалось определить годы: {e}")

    def check_qdrant_collections(self) -> tuple[bool, list[str]]:
        """Проверяет наличие обязательных коллекций для RAG-поиска."""
        required = ["genbi_measures", "genbi_columns"]
        missing = []
        for collection_name in required:
            try:
                qdrant.get_collection(collection_name)
            except Exception:
                missing.append(collection_name)
        return (len(missing) == 0, missing)

    def retrieve_context(self, decomp_json: dict) -> str:
        """RAG Engine: Ищет меры и таблицы в Qdrant на основе декомпозиции"""
        retrieved_measures = set()
        retrieved_tables = set()

        # 1. Поиск МЕР
        metrics = decomp_json.get("metrics",[])
        for metric in metrics:
            vector = self._get_embedding(metric)
            if not vector: 
                continue
            
            # Ищем топ-3 наиболее подходящие меры для каждой метрики
            try:
                hits = qdrant.query_points(
                    collection_name="genbi_measures",
                    query=vector,
                    limit=10
                )
            except UnexpectedResponse as e:
                raise RuntimeError(f"Qdrant не готов: {e}") from e
            for hit in hits.points:
                payload = hit.payload
                # Формируем строку для промпта
                retrieved_measures.add(f"[{payload['name']}] (Таблица: '{payload['table_name']}', Папка: {payload['folder']})")

        # 2. Поиск ИЗМЕРЕНИЙ И ФИЛЬТРОВ (Schema Linking)
        search_terms = decomp_json.get("dimensions", []) + decomp_json.get("filters",[])
        for term in search_terms:
            vector = self._get_embedding(term)
            if not vector: 
                continue
            
            # Ищем топ-2 столбца для каждого термина
            try:
                hits = qdrant.query_points(
                    collection_name="genbi_columns",
                    query=vector,
                    limit=5
                )
            except UnexpectedResponse as e:
                raise RuntimeError(f"Qdrant не готов: {e}") from e
            for hit in hits.points:
                retrieved_tables.add(hit.payload['table_name'])

        # 3. УМНЫЙ ХАК: Если есть контекст времени, принудительно добавляем Календарь
        time_ctx = decomp_json.get("time_context")
        if time_ctx and time_ctx.lower() != "null":
            retrieved_tables.add("000 Календарь")

        # 4. Сборка итогового контекста
        context_str = "ДОСТУПНЫЕ МЕРЫ (Measures):\n"
        context_str += "\n".join(list(retrieved_measures)) if retrieved_measures else "- Меры не найдены, используй базовые логические выводы.\n"
        
        context_str += "\n\nДОСТУПНЫЕ ТАБЛИЦЫ И ИХ СТОЛБЦЫ (Dimensions):\n"
        for t_name in retrieved_tables:
            if t_name in self.full_schema:
                # Берем все столбцы этой таблицы из локальной памяти
                cols = self.full_schema[t_name]
                context_str += f"- Таблица '{t_name}'\n  Столбцы: {', '.join(cols)}\n"

        return context_str

    def build_generator_prompt(self, context_str: str) -> str:

        now = datetime.now()
        current_date_context = now.strftime("%Y-%m-%d (день недели: %A, месяц: %B)")

        # Актуальный диапазон данных
        latest_year = self.data_year_max or 2022
        year_range_note = ""
        if self.data_year_min and self.data_year_max:
            year_range_note = (
                f"\nВАЖНО: Данные в кубе доступны за {self.data_year_min}–{self.data_year_max} годы. "
                f"Последний год с данными: {self.data_year_max}. "
                f'Если пользователь пишет "текущий год", "этот год", "за 2024" и т.п. — '
                f"используй последний доступный год ({self.data_year_max}).\n"
            )

        return f"""Ты — Senior BI-Аналитик (SSAS Tabular, DAX).
Твоя задача — сгенерировать валидный DAX запрос на основе контекста.

Сегодняшняя дата: {current_date_context}.
{year_range_note}

### КОНТЕКСТ ДАННЫХ (ТОЛЬКО ЭТИ ОБЪЕКТЫ СУЩЕСТВУЮТ):
{context_str}

### ЖЕСТКИЕ ПРАВИЛА DAX:
1. Используй ТОЛЬКО объекты, перечисленные в контексте выше. 
2. Синтаксис генерации таблицы: EVALUATE TOPN(1000, SUMMARIZECOLUMNS('Таблица'[Столбец], "Заголовок", [Мера]))
3. Все фильтры (WHERE) оборачивай в функцию FILTER() внутри SUMMARIZECOLUMNS.
4. ВНИМАНИЕ: Названия мер пиши в точности как в контексте! Если мера имеет точку (напр.[Продажи ед.]), пиши её с точкой.
5. НИКОГДА не используй агрегатные функции (SUM, MAX) для мер. Меры уже рассчитаны.
6. НИКОГДА не используй одинарные кавычки внутри квадратных скобок мер! 
   Правильно: [Название меры]
   ОШИБКА: ['Название меры']
7. При фильтрации внутри SUMMARIZECOLUMNS фильтруй ТОЛЬКО конкретный столбец через VALUES, а не всю таблицу! 
   ПРАВИЛЬНО: FILTER(VALUES('000 Календарь'[Год]), '000 Календарь'[Год] = {latest_year})
   ОШИБКА: FILTER('000 Календарь', '000 Календарь'[Год] = {latest_year})
8. ПРАВИЛО СРАВНЕНИЯ ПО ГОДАМ (Year-over-Year): Если пользователь просит сравнить текущий год с прошлым по месяцам:
   - Используй последний доступный год ({latest_year}) как "текущий".
   - ОБЯЗАТЕЛЬНО добавь фильтр: FILTER(VALUES('000 Календарь'[Год]), '000 Календарь'[Год] = {latest_year})
   - В качестве измерения (оси X) используй '000 Календарь'[Месяц] и '000 Календарь'[Месяц Номер].
   - В конце добавь сортировку: ORDER BY '000 Календарь'[Месяц Номер] ASC.

### ПРИМЕРЫ DAX:
EVALUATE TOPN(500, 
  SUMMARIZECOLUMNS(
    '002 Контрагенты'[Бизнес Регион],
    FILTER(VALUES('000 Календарь'[Год]), '000 Календарь'[Год] = {latest_year}),
    "Выручка", [Реализация руб.]
  )
)

Сравнение годов по месяцам:
EVALUATE 
  SUMMARIZECOLUMNS(
    '000 Календарь'[Месяц Номер], 
    '000 Календарь'[Месяц],
    FILTER(VALUES('000 Календарь'[Год]), '000 Календарь'[Год] = {latest_year}),
    "Текущий год", [Реализация руб.],
    "Прошлый год",[Реализация руб. (Прошлый год)]
  )
ORDER BY '000 Календарь'[Месяц Номер] ASC

ОТВЕТЬ СТРОГО В JSON ФОРМАТЕ:
{{{{
  "thought": "Твои рассуждения: какие таблицы, фильтры и меры ты выбрал",
  "dax": "Сгенерированный DAX запрос",
  "chart_type": "bar | line | pie | treemap | scatter | area | table"
}}}}
"""

    def validate_and_fix_dax(self, dax: str) -> str:
        """Синтаксический авто-корректор опечаток в мерах"""
        found_measures = set(re.findall(r'\[(.*?)\]', dax))
        
        for found in found_measures:
            # Игнорируем столбцы, т.к. перед ними обычно идет кавычка 'Таблица'[Столбец]
            if f"'{found}'" not in dax and f'"{found}"' not in dax:
                if found not in self.all_measure_names and self.all_measure_names:
                    # Fuzzy-поиск наиболее похожего имени
                    closest = difflib.get_close_matches(found, self.all_measure_names, n=1, cutoff=0.8)
                    if closest:
                        correct = closest[0]
                        logging.info(f"Авто-исправление опечатки: [{found}] -> [{correct}]")
                        dax = dax.replace(f"[{found}]", f"[{correct}]")
        return dax

    def ask(self, user_query: str) -> dict:
        """Main Agentic Workflow: Decompose -> Retrieve -> Generate -> Execute -> Reflect"""
        
        # 1. Декомпозиция
        logging.info(f'Запрос: {user_query}')
        decomp_json = decompose_query(user_query)
        if not decomp_json:
            return {"error": "Не удалось декомпозировать запрос."}

        ready, missing = self.check_qdrant_collections()
        if not ready:
            missing_list = ", ".join(missing)
            return {
                "error": (
                    f"В Qdrant отсутствуют коллекции: {missing_list}. "
                    "Выполните команду: python indexer_qdrant.py"
                ),
                "dax": "",
                "decomposition": decomp_json,
            }
            
        # 2. RAG Поиск
        logging.info("Поиск контекста в Qdrant...")
        try:
            context_str = self.retrieve_context(decomp_json)
        except RuntimeError as e:
            return {
                "error": (
                    "Не удалось выполнить поиск контекста в Qdrant. "
                    f"{e}. Проверьте Qdrant и выполните: python indexer_qdrant.py"
                ),
                "dax": "",
                "decomposition": decomp_json,
            }
        logging.info("Контекст собран.")
        logging.info(context_str)
        
        # 3. Настройка Генератора
        messages =[
            {"role": "system", "content": self.build_generator_prompt(context_str)},
            {"role": "user", "content": f"Напиши DAX запрос для: {user_query}"}
        ]
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            logging.info(f"Генерация DAX (Попытка {attempt + 1})...")
            
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    response_format = {  
                        "type": "json_schema",
                        "json_schema": {
                            "name": "decomposition", 
                            "strict": True,
                            "schema": OrcehestratorOutput.model_json_schema()
                        },
                    },
                    timeout=45.0
                )
                ai_data = json.loads(completion.choices[0].message.content)
                logging.info(ai_data)
            except Exception as e:
                return {"error": f"Ошибка LLM API: {e}", "dax": "", "decomposition": decomp_json}

            # 4. Валидация
            dax_query = ai_data.get("dax", "")
            dax_query = self.validate_and_fix_dax(dax_query)
            ai_data["dax"] = dax_query
            
            # 5. Выполнение запроса
            logging.info("Отправка запроса в SSAS...")
            logging.info(dax_query)
            df = execute_dax(dax_query)
            
            # 6. Обработка ошибок (Reflection)
            if isinstance(df, str) and "ОШИБКА" in df:
                logging.error(f"❌ Ошибка SSAS: {df}")
                if attempt < max_retries:
                    logging.error("Отправка ошибки ИИ для самокоррекции...")
                    # Добавляем неудачный ответ и ошибку в историю
                    messages.append({"role": "assistant", "content": json.dumps(ai_data, ensure_ascii=False)})
                    messages.append({
                        "role": "user", 
                        "content": f"Твой DAX вызвал ошибку сервера: {df}. Исправь синтаксис или имена объектов и верни исправленный JSON."
                    })
                    continue
                else:
                    ai_data["error"] = f"Не удалось исправить ошибку. Последний ответ сервера: {df}"
                    ai_data["decomposition"] = decomp_json
                    return ai_data
            
            elif df.empty:
                ai_data["df"] = df
                ai_data["warning"] = "Запрос синтаксически верен, но база вернула 0 строк (возможно, слишком строгие фильтры)."
                ai_data["decomposition"] = decomp_json
                return ai_data
                
            else:
                logging.info("✅ Успех! Данные получены.")
                ai_data["df"] = df
                ai_data["decomposition"] = decomp_json
                return ai_data

# Создаем глобальный инстанс оркестратора
orchestrator = GenBIOrchestrator()

def generate_dax_from_text(user_query: str) -> dict:
    """Точка входа для app.py"""
    return orchestrator.ask(user_query)

if __name__ == '__main__':
    queries = [
        'Продажи по видам номенклатуры',
        'Выручка в разрезе бизнес-регионов',
        'Покажи остатки на складах в штуках',
        'Топ-10 товаров по сумме реализации',
        'Выручка по месяцам только для контрагентов из Москвы',
        'Остатки номенклатуры с видом "Винт" на текущий момент',
        'Сравни продажи этого года с прошлым годом по месяцам',
        'Динамика валовой прибыли по кварталам',
        'Отклонение выручки к прошлому году в процентах по категориям товаров',
        'Рентабельность по менеджерам за последний месяц',
        'Доля продаж каждого товара в общем объеме (в процентах)',
        'Средний оплаченный счет в разрезе ответственных менеджеров',
        'Количество новых клиентов по месяцам',
        'Список неоплаченных счетов по контрагентам',
        'Остатки на складах, которые сейчас находятся в резерве (в кг)' 
    ]

    for query in queries:
        generate_dax_from_text(query)