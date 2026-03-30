# Genbi — Инструкция по запуску

## Требования

- **Python** 3.10+
- **Docker** (для Qdrant)
- Доступ к **SSAS Tabular** серверу (XMLA endpoint)
- API-ключ от **aitunnel.ru** (LLM + Embeddings)
- Файл модели **OLAP_MK.bim** (уже в репозитории)
- Шрифт **DejaVuSans** (для кириллицы в PDF-экспорте)

---

## 1. Клонируйте репозиторий

```bash
git clone https://github.com/asyaify/genbi.git
cd genbi/Genbi-main
```

---

## 2. Установите зависимости Python

```bash
pip install -r requirements.txt
```

Основные зависимости:
- `streamlit` — веб-интерфейс
- `openai` — взаимодействие с LLM API
- `qdrant-client` — векторный поиск (RAG)
- `plotly` — графики (6 типов)
- `pandas`, `numpy` — обработка данных
- `reportlab` — PDF-экспорт
- `python-docx` — Word-экспорт
- `requests-ntlm` — NTLM-авторизация для SSAS (опционально)
- `matplotlib` — условное форматирование таблиц

---

## 3. Установите шрифт для PDF (если нет)

```bash
sudo apt-get install -y fonts-dejavu-core
```

---

## 4. Создайте файл `.env`

Создайте файл `.env` в папке `Genbi-main/`:

```env
# ── LLM API ──
API_KEY=ваш_ключ_aitunnel

# ── SSAS / XMLA ──
CUBE_URL=https://ваш-сервер.ru/
CUBE_USERNAME=user@domain.ru
CUBE_PASSWORD=пароль
CATALOG=OLAP_Mk
AUTH_METHOD=basic          # basic | ntlm | auto

# ── Приложение ──
APP_PASSWORD=пароль_для_входа_в_UI
```

| Переменная      | Описание                                          |
|-----------------|---------------------------------------------------|
| `API_KEY`       | Ключ API aitunnel.ru (или `SECRET_KEY`)           |
| `CUBE_URL`      | URL XMLA-эндпойнта SSAS                           |
| `CUBE_USERNAME` | Логин для SSAS                                    |
| `CUBE_PASSWORD` | Пароль для SSAS                                   |
| `CATALOG`       | Имя каталога SSAS (например `OLAP_Mk`)            |
| `AUTH_METHOD`   | Метод авторизации: `basic`, `ntlm` или `auto`     |
| `APP_PASSWORD`  | Пароль для входа в веб-интерфейс Streamlit         |

---

## 5. Поднимите Qdrant (векторная БД)

```bash
docker-compose up -d
```

Qdrant будет доступен на `http://localhost:6333`.

Проверка:
```bash
curl http://localhost:6333/collections
```

---

## 6. Проиндексируйте модель куба

```bash
python indexer_qdrant.py
```

Скрипт:
- Парсит файл `OLAP_MK.bim`
- Создаёт 2 коллекции в Qdrant: `genbi_measures` (меры) и `genbi_columns` (столбцы)
- Генерирует embeddings через `text-embedding-3-small`

> Запускайте повторно только при изменении BIM-модели.

---

## 7. Запустите приложение

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

Или через скрипт:
```bash
bash start.sh
```

Приложение будет доступно по адресу: **http://localhost:8501**

---

## 8. Войдите в интерфейс

Введите пароль, указанный в `APP_PASSWORD`.

---

## Запуск тестов

```bash
python -m pytest tests/ -v
```

108 тестов покрывают: визуализацию, KPI, PDF/Word/Excel экспорт, декомпозицию запросов, pipeline оркестратора, парсинг SSAS-ответов.

---

## Структура проекта

```
Genbi-main/
├── app.py                 # Streamlit UI (основной файл)
├── genbi_agent.py         # Оркестратор: RAG → LLM → DAX → SSAS
├── decomposer_agent.py    # Агент декомпозиции запросов
├── cube_client.py         # XMLA-клиент для SSAS
├── indexer_qdrant.py      # Индексатор BIM → Qdrant
├── OLAP_MK.bim            # Модель данных SSAS Tabular
├── requirements.txt       # Python-зависимости
├── docker-compose.yml     # Qdrant контейнер
├── .env                   # Переменные окружения (не в git)
├── start.sh               # Скрипт запуска
├── tests/                 # Тесты (108 штук)
│   ├── test_app.py        # Тесты UI-функций
│   ├── test_genbi_agent.py# Тесты оркестратора
│   ├── test_decomposer.py # Тесты декомпозитора
│   └── test_cube_client.py# Тесты XMLA-клиента
└── qdrant_storage/        # Данные Qdrant (volume)
```

---

## Решение проблем

| Проблема                         | Решение                                                    |
|----------------------------------|------------------------------------------------------------|
| HTTP 401 от SSAS                 | Проверьте `CUBE_USERNAME`/`CUBE_PASSWORD`, попробуйте `AUTH_METHOD=basic` |
| Qdrant не готов                  | Запустите `docker-compose up -d`, затем `python indexer_qdrant.py`         |
| PDF без кириллицы                | Установите `fonts-dejavu-core`                              |
| Запрос возвращает 0 строк        | Данные в кубе за 2019–2022. Система автоматически подставляет актуальный год |
| `ImportError: matplotlib`        | `pip install matplotlib` (уже в requirements.txt)           |
