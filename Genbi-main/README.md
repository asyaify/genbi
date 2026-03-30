# Запуск
1. Положить файл `OLAP_MK.bim` в корень проекта.
2. Установить зависимости: `pip install -r requirements.txt`
3. Поднять Qdrant: `docker-compose up -d`
4. Создать env: `cp .env.example .env` и заполнить значения.
5. Проиндексировать куб: `python indexer_qdrant.py`
6. Запустить интерфейс: `streamlit run app.py`

## Переменные окружения (.env)

```env
API_KEY=...

CUBE_URL=...
CUBE_USERNAME=...
CUBE_PASSWORD=...
CATALOG=...

APP_PASSWORD=...
```

Если значение пароля содержит символ `$`, для корректной работы `docker-compose` укажите его как `$$`.

Совместимость со старым env сохранена: `USERNAME/PASSWORD` и `SECRET_KEY` также читаются как fallback.

## Что умеет интерфейс

- Чат с генерацией DAX и выполнением в SSAS
- Табличный вывод результата
- Автопостроение графика (`bar` / `line`) с более информативным оформлением
- Граф запроса через Graphviz (меры, разрезы, фильтры, временной контекст)
- Ключевые наблюдения по данным (размер результата, базовая статистика)
- Мини executive summary рядом с графиком
- Экспорт отчёта в Excel (`.xlsx`), Word (`.docx`) и PDF (`.pdf`)
