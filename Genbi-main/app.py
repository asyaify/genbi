import io
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from docx import Document
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from genbi_agent import generate_dax_from_text

# ── Регистрация кириллических шрифтов для PDF ──
_CYRILLIC_FONT = "DejaVuSans"
_CYRILLIC_FONT_BOLD = "DejaVuSans-Bold"
_FONT_REGISTERED = False

def _register_cyrillic_fonts():
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    try:
        pdfmetrics.registerFont(TTFont(_CYRILLIC_FONT, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont(_CYRILLIC_FONT_BOLD, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
        _FONT_REGISTERED = True
    except Exception:
        pass  # Fallback to Helvetica

def _pdf_font(bold: bool = False) -> str:
    if _FONT_REGISTERED:
        return _CYRILLIC_FONT_BOLD if bold else _CYRILLIC_FONT
    return "Helvetica-Bold" if bold else "Helvetica"

# ── Константы для smart-агрегации графиков ──
TOP_N_CHART = 15  # Показывать топ-N категорий, остальные — «Остальные»

CHART_TYPES_MAP = {
    "bar": "📊 Гистограмма",
    "line": "📈 Линейный",
    "pie": "🥧 Круговая",
    "treemap": "🗂️ Treemap",
    "scatter": "⚬ Точечная",
    "area": "📐 Область",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  -  %(message)s",
    handlers=[logging.StreamHandler()]
)

load_dotenv(interpolate=False)
APP_PASSWORD = os.getenv("APP_PASSWORD") or os.getenv("PASSWORD")

st.set_page_config(page_title="Genbi | AI BI Chat", page_icon="📊", layout="wide")


def apply_custom_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --genbi-accent: #0E6BA8;
                --genbi-soft: #D8EDF8;
                --genbi-ok: #1F8A70;
                --genbi-bg: #F6FBFF;
            }
            .stApp {
                background:
                    radial-gradient(circle at 10% 10%, var(--genbi-soft), transparent 40%),
                    radial-gradient(circle at 90% 0%, #FDEEDC, transparent 35%),
                    var(--genbi-bg);
            }
            h1, h2, h3 {
                letter-spacing: 0.3px;
            }
            /* KPI-карточки — компактные и аккуратные */
            [data-testid="stMetric"] {
                background: white;
                border: 1px solid #E2EDF5;
                border-radius: 12px;
                padding: 12px 16px;
                box-shadow: 0 2px 8px rgba(14, 107, 168, 0.06);
            }
            [data-testid="stMetricLabel"] {
                font-size: 0.8rem !important;
                color: #555;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.3rem !important;
            }
            /* Expander стиль */
            .streamlit-expanderHeader {
                font-size: 0.9rem;
            }
            /* Успешное сообщение */
            .stSuccess {
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="GenbiResult")
    output.seek(0)
    return output.read()


def dataframe_to_word_bytes(df: pd.DataFrame, title: str) -> bytes:
    doc = Document()
    doc.add_heading("Genbi отчёт", level=1)
    doc.add_paragraph(f"Сформировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(f"Запрос: {title}")

    rows, cols = df.shape
    doc.add_paragraph(f"Строк: {rows}, колонок: {cols}")

    table = doc.add_table(rows=1, cols=cols)
    table.style = "Table Grid"
    for col_idx, col_name in enumerate(df.columns):
        table.rows[0].cells[col_idx].text = str(col_name)

    for _, row in df.head(200).iterrows():
        cells = table.add_row().cells
        for col_idx, val in enumerate(row):
            cells[col_idx].text = "" if pd.isna(val) else str(val)

    data = io.BytesIO()
    doc.save(data)
    data.seek(0)
    return data.read()


def dataframe_to_pdf_bytes(df: pd.DataFrame, title: str, summary_text: str) -> bytes:
    _register_cyrillic_fonts()
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 40
    pdf.setFont(_pdf_font(bold=True), 14)
    pdf.drawString(40, y, "Genbi отчёт")
    y -= 20

    pdf.setFont(_pdf_font(), 9)
    pdf.drawString(40, y, f"Сформировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 16
    pdf.drawString(40, y, f"Запрос: {title[:120]}")
    y -= 16
    pdf.drawString(40, y, f"Строк: {len(df)}, колонок: {len(df.columns)}")
    y -= 20

    pdf.setFont(_pdf_font(bold=True), 10)
    pdf.drawString(40, y, "Executive summary")
    y -= 14
    pdf.setFont(_pdf_font(), 9)
    for line in summary_text.split("\n")[:8]:
        if y < 80:
            pdf.showPage()
            y = height - 40
            pdf.setFont(_pdf_font(), 9)
        pdf.drawString(40, y, line[:120])
        y -= 12

    y -= 8
    pdf.setFont(_pdf_font(bold=True), 10)
    pdf.drawString(40, y, "Таблица (первые 25 строк)")
    y -= 14
    pdf.setFont(_pdf_font(), 8)

    cols = [str(c)[:22] for c in df.columns]
    header = " | ".join(cols)
    pdf.drawString(40, y, header[:130])
    y -= 10

    for _, row in df.head(25).iterrows():
        if y < 60:
            pdf.showPage()
            y = height - 40
            pdf.setFont(_pdf_font(), 8)
        row_text = " | ".join([str(v)[:22] if not pd.isna(v) else "" for v in row.values])
        pdf.drawString(40, y, row_text[:130])
        y -= 10

    pdf.save()
    buffer.seek(0)
    return buffer.read()


def build_query_graph_dot(ai_response: dict) -> str:
    """Строит компактную DOT-схему декомпозиции запроса с данными, если они есть."""
    decomp = ai_response.get("decomposition", {}) or {}
    metrics = decomp.get("metrics", []) or []
    dimensions = decomp.get("dimensions", []) or []
    filters = decomp.get("filters", []) or []
    time_context = decomp.get("time_context", "") or "Нет"
    chart_type = ai_response.get("chart_type", "table")

    dot_lines = [
        "digraph G {",
        "rankdir=LR;",
        "node [shape=box, style=\"rounded,filled\", fontsize=10, color=\"#0E6BA8\", fillcolor=\"#EAF5FB\"];",
        "edge [color=\"#888888\"];",
        "q [label=\"Запрос\", shape=oval, fillcolor=\"#FDEEDC\", color=\"#CC7A00\"];",
    ]

    parts = []
    for idx, metric in enumerate(metrics[:4]):
        node = f"m{idx}"
        dot_lines.append(f"{node} [label=\"{str(metric).replace(chr(34), chr(39))}\"];")
        parts.append(node)

    for idx, dim in enumerate(dimensions[:4]):
        node = f"d{idx}"
        dot_lines.append(f"{node} [label=\"{str(dim).replace(chr(34), chr(39))}\", fillcolor=\"#D8EDF8\"];")
        parts.append(node)

    for idx, flt in enumerate(filters[:3]):
        node = f"f{idx}"
        dot_lines.append(
            f"{node} [label=\"{str(flt).replace(chr(34), chr(39))}\", fillcolor=\"#FFF1F1\", color=\"#B54747\"];"
        )
        parts.append(node)

    if time_context and time_context != "Нет":
        dot_lines.append(f"t [label=\"{time_context}\", fillcolor=\"#FFF7EA\", color=\"#C88300\"];")
        parts.append("t")

    result_label = f"Результат ({chart_type})"
    dot_lines.append(f"r [label=\"{result_label}\", fillcolor=\"#EAFBF0\", color=\"#1F8A70\"];")

    for p in parts:
        dot_lines.append(f"q -> {p};")
        dot_lines.append(f"{p} -> r;")

    dot_lines.append("}")
    return "\n".join(dot_lines)


# Слова-маркеры, указывающие, что метрика уже агрегирована (среднее, %, доля)
_RATE_KEYWORDS = ("средн", "доля", "процент", "%", "наценка", "рентаб", "отклонение", "коэффициент")


def _is_rate_metric(col_name: str) -> bool:
    """Определяет, является ли метрика относительной (среднее, %, доля)."""
    lower = col_name.lower()
    return any(kw in lower for kw in _RATE_KEYWORDS)


def build_kpi_cards(df: pd.DataFrame) -> list[dict]:
    """
    Умные KPI-карточки:
    - Для абсолютных метрик (выручка, кол-во): Итого + Макс
    - Для относительных (средний чек, %): Медиана + Диапазон
    - Уникальные категории первого текстового столбца
    """
    cards = []
    if df is None or df.empty:
        return cards

    df = _clean_df_for_display(df)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # Кол-во записей
    cards.append({"label": "Записей", "value": f"{len(df):,}", "delta": None})

    # Кол-во уникальных категорий (если есть текстовый столбец)
    if cat_cols:
        n_unique = df[cat_cols[0]].nunique()
        cards.append({"label": f"Уник. {cat_cols[0][:18]}", "value": f"{n_unique:,}", "delta": None})

    for col in num_cols[:2]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        short_name = col[:20]

        if _is_rate_metric(col):
            # Относительная метрика — медиана + диапазон
            cards.append({
                "label": f"Медиана: {short_name}",
                "value": _format_value_axis(series.median()),
                "delta": None,
            })
            cards.append({
                "label": f"Мин / Макс",
                "value": f"{_format_value_axis(series.min())} – {_format_value_axis(series.max())}",
                "delta": None,
            })
        else:
            # Абсолютная метрика — итого + лидер
            cards.append({
                "label": f"Итого: {short_name}",
                "value": _format_value_axis(series.sum()),
                "delta": None,
            })
            cards.append({
                "label": f"Макс: {short_name}",
                "value": _format_value_axis(series.max()),
                "delta": None,
            })

    return cards[:6]


def _clean_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Заменяет NaN/пустые значения в категориальных столбцах на '(не указано)'."""
    df = df.copy()
    cat_cols = df.select_dtypes(exclude=["number"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("(не указано)").replace("", "(не указано)")
        df[col] = df[col].astype(str).replace("nan", "(не указано)")
    return df


def _aggregate_top_n(df: pd.DataFrame, x_col: str, y_col: str, n: int = TOP_N_CHART) -> pd.DataFrame:
    """
    Агрегирует DataFrame: оставляет топ-N категорий по y_col, остальные сворачивает в «Остальные».
    Подход рекомендован для dashboards с большим числом категорий (>15).
    """
    if len(df) <= n:
        return df

    df_sorted = df.sort_values(y_col, ascending=False, key=lambda s: pd.to_numeric(s, errors="coerce"))
    top = df_sorted.head(n).copy()
    rest = df_sorted.iloc[n:]

    if not rest.empty:
        rest_row = {x_col: f"Остальные ({len(rest)})"}
        for col in df.columns:
            if col == x_col:
                continue
            numeric = pd.to_numeric(rest[col], errors="coerce")
            if not numeric.isna().all():
                rest_row[col] = numeric.sum()
            else:
                rest_row[col] = ""
        top = pd.concat([top, pd.DataFrame([rest_row])], ignore_index=True)

    return top


def _format_value_axis(value: float) -> str:
    """Форматирует числа: 1.2M, 540K и т. д."""
    abs_val = abs(value)
    if abs_val >= 1e9:
        return f"{value / 1e9:.1f} млрд"
    if abs_val >= 1e6:
        return f"{value / 1e6:.1f} млн"
    if abs_val >= 1e3:
        return f"{value / 1e3:.1f} тыс"
    return f"{value:,.0f}"


def build_insights(df: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    if df is None or df.empty:
        return insights

    df = _clean_df_for_display(df)
    insights.append(f"Строк: {len(df)}, колонок: {len(df.columns)}")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    if num_cols:
        first_num = num_cols[0]
        series = pd.to_numeric(df[first_num], errors="coerce").dropna()
        if not series.empty:
            insights.append(
                f"{first_num}: min={series.min():,.2f}, max={series.max():,.2f}, avg={series.mean():,.2f}"
            )

    if cat_cols:
        first_cat = cat_cols[0]
        top = df[first_cat].astype(str).value_counts(dropna=False).head(1)
        if not top.empty:
            insights.append(f"Топ категория по {first_cat}: {top.index[0]} ({int(top.iloc[0])} строк)")

    return insights


def build_executive_summary(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "Данные отсутствуют, сформировать вывод невозможно."

    df = _clean_df_for_display(df)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    lines = []

    # 1. Обзор данных
    lines.append(f"**Обзор данных:** {len(df)} строк, {len(df.columns)} колонок.")

    # 2. Качество данных
    total_cells = len(df) * len(df.columns)
    missing = df.isna().sum().sum()
    if missing > 0 and total_cells > 0:
        pct = missing / total_cells * 100
        lines.append(f"**Качество данных:** обнаружены пропуски — {int(missing)} ({pct:.1f}% ячеек).")

    # 3. Ключевые метрики (до 2 числовых столбцов)
    for col in num_cols[:2]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        short = col[:30]
        metric_parts = []

        if _is_rate_metric(col):
            metric_parts.append(
                f"**Метрика «{short}»:** медиана {_format_value_axis(series.median())}, "
                f"диапазон {_format_value_axis(series.min())} – {_format_value_axis(series.max())}."
            )
        else:
            metric_parts.append(
                f"**Метрика «{short}»:** итого {_format_value_axis(series.sum())}, "
                f"среднее {_format_value_axis(series.mean())}, макс {_format_value_axis(series.max())}."
            )

        # Выбросы (IQR)
        if len(series) >= 4:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
                if outliers > 0:
                    metric_parts.append(f"Выбросы: {outliers} значений за пределами нормы (IQR).")

        # Тренд (сравнение первой и второй половины)
        if len(series) > 2:
            first_half = series.iloc[:len(series) // 2].mean()
            second_half = series.iloc[len(series) // 2:].mean()
            if first_half > 0:
                change_pct = (second_half - first_half) / first_half * 100
                if abs(change_pct) > 5:
                    direction = "рост" if change_pct > 0 else "снижение"
                    metric_parts.append(f"Тренд: {direction} ~{abs(change_pct):.0f}% (1-я половина → 2-я).")

        lines.append("  \n".join(metric_parts))

    # 4. Лидеры по категориям с долями
    if cat_cols and num_cols:
        dim = cat_cols[0]
        metric_name = num_cols[0]
        grouped = (
            df[[dim, metric_name]]
            .assign(_m=pd.to_numeric(df[metric_name], errors="coerce"))
            .dropna(subset=["_m"])
            .groupby(dim, dropna=False)["_m"]
            .sum()
            .sort_values(ascending=False)
        )
        if not grouped.empty and grouped.sum() > 0:
            top3 = grouped.head(3)
            total = grouped.sum()
            leader_lines = []
            for k, v in top3.items():
                pct = v / total * 100
                leader_lines.append(f"- {k}: {_format_value_axis(v)} ({pct:.0f}%)")
            lines.append(f"**Лидеры по «{metric_name}»:**  \n" + "  \n".join(leader_lines))

            # Концентрация
            top3_share = top3.sum() / total * 100
            if top3_share > 70:
                lines.append(f"**Внимание:** высокая концентрация — топ-3 = {top3_share:.0f}% от общего объёма.")

    # 5. Рекомендация
    lines.append("**Рекомендация:** проверьте выбросы и динамику лидеров перед принятием решений.")
    return "\n\n".join(lines)


def build_chart(df: pd.DataFrame, chart_type: str, x_col: str = None,
                y_cols_override: list[str] = None,
                show_avg: bool = False, show_median: bool = False):
    """
    Строит Plotly-график с поддержкой множества типов визуализации и справочных линий.
    Типы: bar, line, pie, treemap, scatter, area.
    """
    if df is None or df.empty or len(df.columns) < 2 or chart_type == "table":
        return None

    df = _clean_df_for_display(df)
    num_cols_all = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols_all]
    if not num_cols_all:
        return None

    # Авто-определение осей, если не указаны
    x_axis = x_col or (cat_cols[0] if cat_cols else df.columns[0])
    if y_cols_override:
        y_main = y_cols_override[0]
        y_axis = y_cols_override[:2] if len(y_cols_override) > 1 else y_cols_override[0]
    else:
        y_main = num_cols_all[0]
        y_axis = num_cols_all[:2] if len(num_cols_all) > 1 else num_cols_all[0]

    fig = None

    if chart_type == "bar":
        plot_df = _aggregate_top_n(df, x_axis, y_main, TOP_N_CHART)
        plot_df = plot_df.sort_values(y_main, ascending=True,
                                      key=lambda s: pd.to_numeric(s, errors="coerce"))
        fig = px.bar(
            plot_df, y=x_axis, x=y_axis, orientation="h",
            title=f"{y_main} по {x_axis} (топ-{min(TOP_N_CHART, len(df))})",
            text_auto=".3s",
        )
        fig.update_layout(
            template="plotly_white", legend_title_text="Показатели",
            margin=dict(l=10, r=20, t=50, b=30), yaxis_title="", xaxis_title=y_main,
            height=max(400, min(TOP_N_CHART + 1, len(plot_df)) * 35 + 100), bargap=0.15,
        )
        fig.update_traces(textposition="outside", textfont_size=11)
        colors = ["#0E6BA8"] * len(plot_df)
        if len(df) > TOP_N_CHART and colors:
            colors[0] = "#C0C0C0"
        fig.update_traces(marker_color=colors, selector=dict(type="bar"))
        # Справочные линии (вертикальные для горизонтальных баров)
        if show_avg or show_median:
            series = pd.to_numeric(df[y_main], errors="coerce").dropna()
            if not series.empty:
                if show_avg:
                    fig.add_vline(x=series.mean(), line_dash="dash", line_color="#E74C3C",
                                  annotation_text=f"Средн: {_format_value_axis(series.mean())}")
                if show_median:
                    fig.add_vline(x=series.median(), line_dash="dot", line_color="#27AE60",
                                  annotation_text=f"Медиана: {_format_value_axis(series.median())}")

    elif chart_type == "line":
        sort_cols = [x_axis] if x_axis in df.columns else None
        source = df.sort_values(by=sort_cols) if sort_cols else df
        fig = px.line(source, x=x_axis, y=y_axis,
                      title=f"Динамика: {y_main} по {x_axis}", markers=True)
        fig.update_layout(
            template="plotly_white", legend_title_text="Показатели",
            margin=dict(l=30, r=20, t=50, b=30), xaxis_title=x_axis,
            yaxis_title="Значение", hovermode="x unified",
        )
        _add_reference_lines(fig, df, y_main, show_avg, show_median)

    elif chart_type == "pie":
        plot_df = _aggregate_top_n(df, x_axis, y_main, TOP_N_CHART)
        plot_df[y_main] = pd.to_numeric(plot_df[y_main], errors="coerce").fillna(0).clip(lower=0)
        fig = px.pie(plot_df, names=x_axis, values=y_main,
                     title=f"Распределение: {y_main} по {x_axis}", hole=0.35)
        fig.update_traces(textinfo="percent+label", textposition="outside")
        fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))

    elif chart_type == "treemap":
        plot_df = _aggregate_top_n(df, x_axis, y_main, TOP_N_CHART * 2)
        plot_df[y_main] = pd.to_numeric(plot_df[y_main], errors="coerce").fillna(0).clip(lower=0)
        fig = px.treemap(plot_df, path=[x_axis], values=y_main,
                         title=f"Treemap: {y_main} по {x_axis}",
                         color=y_main, color_continuous_scale="Blues")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))

    elif chart_type == "scatter":
        y_list = y_cols_override if y_cols_override and len(y_cols_override) >= 2 else num_cols_all[:2]
        if len(y_list) >= 2:
            fig = px.scatter(df, x=y_list[0], y=y_list[1],
                             color=x_axis if cat_cols else None,
                             title=f"Корреляция: {y_list[0]} vs {y_list[1]}",
                             hover_data=df.columns[:4].tolist())
        else:
            fig = px.scatter(df, x=x_axis, y=y_main, title=f"{y_main} по {x_axis}")
        fig.update_layout(template="plotly_white", margin=dict(l=30, r=20, t=50, b=30))
        ref_col = y_list[1] if len(y_list) >= 2 else y_main
        _add_reference_lines(fig, df, ref_col, show_avg, show_median)

    elif chart_type == "area":
        sort_cols = [x_axis] if x_axis in df.columns else None
        source = df.sort_values(by=sort_cols) if sort_cols else df
        fig = px.area(source, x=x_axis, y=y_axis,
                      title=f"Область: {y_main} по {x_axis}")
        fig.update_layout(
            template="plotly_white", legend_title_text="Показатели",
            margin=dict(l=30, r=20, t=50, b=30), hovermode="x unified",
        )
        _add_reference_lines(fig, df, y_main, show_avg, show_median)

    else:
        return None

    return fig


def _add_reference_lines(fig, df: pd.DataFrame, y_col: str,
                         show_avg: bool, show_median: bool) -> None:
    """Добавляет горизонтальные справочные линии (среднее / медиана) на график."""
    series = pd.to_numeric(df[y_col], errors="coerce").dropna()
    if series.empty:
        return
    if show_avg:
        fig.add_hline(y=series.mean(), line_dash="dash", line_color="#E74C3C",
                      annotation_text=f"Средн: {_format_value_axis(series.mean())}")
    if show_median:
        fig.add_hline(y=series.median(), line_dash="dot", line_color="#27AE60",
                      annotation_text=f"Медиана: {_format_value_axis(series.median())}")


def _style_dataframe(df: pd.DataFrame):
    """Условное форматирование: цветовой градиент по числовым столбцам."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        return df
    try:
        styled = df.style
        for col in num_cols[:4]:
            styled = styled.background_gradient(subset=[col], cmap="YlOrRd")
        return styled
    except ImportError:
        return df


def build_drilldown_suggestions(df: pd.DataFrame, original_query: str) -> list[str]:
    """Генерирует предложения для drill-down на основе данных."""
    suggestions = []
    if df is None or df.empty:
        return suggestions

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    if cat_cols and num_cols:
        main_cat = cat_cols[0]
        main_metric = num_cols[0]
        series = pd.to_numeric(df[main_metric], errors="coerce")
        grouped = df.assign(_m=series).groupby(main_cat)["_m"].sum().sort_values(ascending=False)
        if not grouped.empty:
            top = str(grouped.index[0])
            if top != "(не указано)":
                suggestions.append(f"Детализация по '{top}'")

    lower_q = original_query.lower() if original_query else ""
    if not any(kw in lower_q for kw in ("месяц", "квартал", "динамик", "тренд", "год")):
        suggestions.append("Показать динамику по месяцам")

    if cat_cols and len(cat_cols) > 1:
        suggestions.append(f"Разбивка по {cat_cols[1]}")
    elif num_cols and len(num_cols) > 1:
        suggestions.append(f"Сравнить {num_cols[0]} и {num_cols[1]}")

    return suggestions[:3]


def render_interactive_analysis(df: pd.DataFrame, default_chart_type: str,
                                 suffix: str, query: str = "") -> None:
    """Интерактивная панель анализа в стиле Grafana: тип графика, оси, пороговые линии, drill-down."""
    df_clean = _clean_df_for_display(df)
    num_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df_clean.columns if c not in num_cols]
    all_cols = df_clean.columns.tolist()

    if not num_cols:
        st.dataframe(_style_dataframe(df_clean), use_container_width=True)
        return

    # ── Панель управления ──
    chart_options = list(CHART_TYPES_MAP.keys())
    default_idx = chart_options.index(default_chart_type) if default_chart_type in chart_options else 0

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1, 1])
    with ctrl1:
        chart_type = st.selectbox(
            "Тип графика", options=chart_options,
            format_func=lambda x: CHART_TYPES_MAP[x],
            index=default_idx, key=f"ctype_{suffix}",
        )
    with ctrl2:
        default_x = cat_cols[0] if cat_cols else all_cols[0]
        x_col = st.selectbox(
            "Категория / Ось X", options=all_cols,
            index=all_cols.index(default_x), key=f"xcol_{suffix}",
        )
    with ctrl3:
        y_options = num_cols
        default_y = y_options[:2] if len(y_options) >= 2 else y_options[:1]
        y_cols = st.multiselect(
            "Значения / Ось Y", options=y_options,
            default=default_y, key=f"ycol_{suffix}",
        )

    # Справочные линии
    ref1, ref2 = st.columns(2)
    with ref1:
        show_avg = st.checkbox("📏 Линия среднего", key=f"avg_{suffix}")
    with ref2:
        show_median = st.checkbox("📐 Линия медианы", key=f"med_{suffix}")

    # ── График ──
    if y_cols:
        fig = build_chart(df_clean, chart_type, x_col, y_cols, show_avg, show_median)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"fig_{suffix}")
    else:
        st.info("Выберите хотя бы один показатель для оси Y")

    # ── Условно-форматированная таблица ──
    with st.expander(f"📋 Таблица данных ({len(df)} строк)"):
        st.dataframe(_style_dataframe(df_clean), use_container_width=True)

    # ── Drill-down предложения ──
    suggestions = build_drilldown_suggestions(df_clean, query)
    if suggestions:
        st.markdown("**🔍 Продолжить анализ:**")
        drill_cols = st.columns(len(suggestions))
        for i, sugg in enumerate(suggestions):
            with drill_cols[i]:
                if st.button(sugg[:45] + ("…" if len(sugg) > 45 else ""),
                             key=f"drill_{suffix}_{i}", use_container_width=True):
                    st.session_state["_prefill"] = sugg
                    st.rerun()


def render_export_buttons(df: pd.DataFrame, prompt: str, suffix: str, summary_text: str) -> None:
    if df is None or df.empty:
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="Скачать Excel (.xlsx)",
            data=dataframe_to_excel_bytes(df),
            file_name=f"genbi_report_{suffix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"xlsx_{suffix}",
        )

    with col2:
        st.download_button(
            label="Скачать Word (.docx)",
            data=dataframe_to_word_bytes(df, prompt),
            file_name=f"genbi_report_{suffix}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            key=f"docx_{suffix}",
        )

    with col3:
        st.download_button(
            label="Скачать PDF (.pdf)",
            data=dataframe_to_pdf_bytes(df, prompt, summary_text),
            file_name=f"genbi_report_{suffix}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"pdf_{suffix}",
        )


def _render_static_chart(df: pd.DataFrame, chart_type: str, suffix: str) -> None:
    """Статичный график для исторических сообщений (без виджетов — стабильный rerun)."""
    fig = build_chart(df, chart_type)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key=f"hist_fig_{suffix}")


def render_assistant_message(msg: dict, idx: int) -> None:
    """Рендерит историческое сообщение. Для стабильности — статичный график без виджетов."""
    st.markdown(msg.get("content", ""))

    if msg.get("warning"):
        st.warning(msg["warning"])

    if msg.get("dataframe") is not None and not msg["dataframe"].empty:
        df = msg["dataframe"]
        suffix = msg.get("suffix", f"history_{idx}")
        query = msg.get("prompt", "")
        chart_type = msg.get("chart_type", "bar")
        executive_summary = msg.get("executive_summary", "Сводка недоступна")

        # KPI-карточки
        kpi_cards = build_kpi_cards(df)
        if kpi_cards:
            kpi_cols = st.columns(len(kpi_cards))
            for i, card in enumerate(kpi_cards):
                with kpi_cols[i]:
                    st.metric(label=card["label"], value=card["value"], delta=card.get("delta"))

        # Статичный график (без интерактивных виджетов — они ломают историю при rerun)
        _render_static_chart(df, chart_type, suffix)

        with st.expander("Executive Summary"):
            st.markdown(executive_summary)

        with st.expander(f"📋 Таблица данных ({len(df)} строк)"):
            st.dataframe(df, use_container_width=True)

        render_export_buttons(df, query, suffix, executive_summary)

    if msg.get("graph_dot"):
        with st.expander("Схема декомпозиции запроса"):
            st.graphviz_chart(msg["graph_dot"])

    if msg.get("dax"):
        with st.expander("DAX запрос"):
            st.code(msg["dax"], language="dax")


def _render_current_message(msg: dict, idx: int) -> None:
    """Рендерит последнее сообщение с интерактивными контролами (выбор графика, осей и т.д.)."""
    st.markdown(msg.get("content", ""))

    if msg.get("warning"):
        st.warning(msg["warning"])

    if msg.get("dataframe") is not None and not msg["dataframe"].empty:
        df = msg["dataframe"]
        suffix = msg.get("suffix", f"current_{idx}")
        query = msg.get("prompt", "")
        chart_type = msg.get("chart_type", "bar")
        executive_summary = msg.get("executive_summary", "Сводка недоступна")

        st.success(f"✅ Данные получены: {len(df)} строк")

        # KPI-карточки
        kpi_cards = build_kpi_cards(df)
        if kpi_cards:
            kpi_cols = st.columns(len(kpi_cards))
            for i, card in enumerate(kpi_cards):
                with kpi_cols[i]:
                    st.metric(label=card["label"], value=card["value"], delta=card.get("delta"))

        # Интерактивная панель анализа
        render_interactive_analysis(df, chart_type, suffix, query)

        with st.expander("Executive Summary"):
            st.markdown(executive_summary)

        render_export_buttons(df, query, suffix, executive_summary)

        # Сохранить запрос
        if st.button("⭐ Сохранить запрос", key=f"save_{suffix}"):
            if query and query not in st.session_state.favorites:
                st.session_state.favorites.append(query)
                st.rerun()

    if msg.get("graph_dot"):
        with st.expander("Схема декомпозиции запроса"):
            st.graphviz_chart(msg["graph_dot"])

    if msg.get("dax"):
        with st.expander("📝 DAX"):
            st.code(msg["dax"], language="dax")


def main() -> None:
    apply_custom_theme()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "favorites" not in st.session_state:
        st.session_state.favorites = []

    if not APP_PASSWORD:
        st.error("Переменная APP_PASSWORD не задана в .env")
        st.stop()

    if not st.session_state.authenticated:
        st.title("Вход в Genbi")
        st.markdown("Введите пароль, заданный в APP_PASSWORD.")
        pwd = st.text_input("Пароль", type="password")
        if st.button("Войти", use_container_width=True):
            if pwd == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Неверный пароль")
        st.stop()

    with st.sidebar:
        st.markdown("## ⚙️ Управление")
        if st.button("🗑️ Очистить чат", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.markdown("#### 💡 Примеры запросов")
        examples = [
            "Выручка по бизнес-регионам",
            "Топ-10 товаров по сумме реализации",
            "Сравни продажи по месяцам с прошлым годом",
            "Остатки на складах в штуках",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state["_prefill"] = ex
                st.rerun()

        # Сохранённые запросы (избранное)
        if st.session_state.favorites:
            st.divider()
            st.markdown("#### ⭐ Сохранённые")
            for i, fav in enumerate(st.session_state.favorites):
                fc1, fc2 = st.columns([5, 1])
                with fc1:
                    if st.button(fav[:30] + ("…" if len(fav) > 30 else ""),
                                 key=f"favr_{i}", use_container_width=True):
                        st.session_state["_prefill"] = fav
                        st.rerun()
                with fc2:
                    if st.button("✕", key=f"favd_{i}"):
                        st.session_state.favorites.pop(i)
                        st.rerun()

        st.divider()
        st.caption("📥 Excel / Word / PDF экспорт доступен")

    st.title("📊 Genbi: OLAP куб в диалоге")
    st.caption("AI-генерация DAX · Анализ данных · Экспорт отчётов")

    # Определяем последнее сообщение с данными для интерактивного рендеринга
    _last_data_idx = None
    for _i in range(len(st.session_state.messages) - 1, -1, -1):
        if (st.session_state.messages[_i].get("role") == "assistant"
                and st.session_state.messages[_i].get("dataframe") is not None
                and not st.session_state.messages[_i]["dataframe"].empty):
            _last_data_idx = _i
            break

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg.get("role", "assistant")):
            if msg.get("role") == "assistant":
                if i == _last_data_idx:
                    _render_current_message(msg, i)
                else:
                    render_assistant_message(msg, i)
            else:
                st.markdown(msg.get("content", ""))

    # Поддержка prefill из sidebar
    prefill = st.session_state.pop("_prefill", None)
    prompt = st.chat_input("Задайте вопрос по данным куба...")
    prompt = prefill or prompt
    if not prompt:
        # Welcome state — показываем когда чат пуст
        if not st.session_state.messages:
            st.markdown("""\n---\n**Добро пожаловать!** Введите вопрос о данных на естественном языке.\n\nПримеры:\n- *Выручка в разрезе бизнес-регионов*\n- *Топ-10 товаров по сумме реализации*\n- *Сравни продажи этого года с прошлым годом по месяцам*\n\nВыберите пример из боковой панели или введите свой запрос.\n""")
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Анализирую куб и формирую ответ..."):
            try:
                ai_response = generate_dax_from_text(prompt)
            except Exception as e:
                st.error(f"Ошибка оркестратора: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Ошибка оркестратора: {str(e)}",
                })
                return

        if not ai_response or not isinstance(ai_response, dict):
            st.error("Некорректный ответ от оркестратора")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Некорректный ответ от оркестратора",
            })
            return

        dax_query = ai_response.get("dax", "")
        thought = ai_response.get("thought", "")
        df = ai_response.get("df")

        save_msg = {
            "role": "assistant",
            "content": f"**Анализ:** {thought}" if thought else "",
            "dax": dax_query,
            "dataframe": None,
            "chart_type": "bar",
            "graph_dot": None,
            "executive_summary": "",
            "warning": ai_response.get("warning", ""),
            "prompt": prompt,
            "suffix": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        if ai_response.get("error"):
            st.error(ai_response["error"])
            save_msg["content"] = f"Ошибка: {ai_response['error']}"
            st.session_state.messages.append(save_msg)
            return

        if isinstance(df, pd.DataFrame) and not df.empty:
            chart_type = ai_response.get("chart_type", "bar")
            if chart_type == "table":
                chart_type = "bar"
            save_msg["dataframe"] = df
            save_msg["chart_type"] = chart_type
            save_msg["executive_summary"] = build_executive_summary(df)
            save_msg["graph_dot"] = build_query_graph_dot(ai_response)
            st.session_state.messages.append(save_msg)
            st.rerun()
        else:
            if save_msg["warning"]:
                st.warning(save_msg["warning"])
            st.info("Результаты не получены. Проверьте DAX и фильтры.")
            if dax_query:
                with st.expander("📝 DAX"):
                    st.code(dax_query, language="dax")
            st.session_state.messages.append(save_msg)


if __name__ == "__main__":
    main()
