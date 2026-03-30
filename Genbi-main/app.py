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
            .genbi-kpi {
                background: white;
                border: 1px solid #E2EDF5;
                border-radius: 14px;
                padding: 12px;
                box-shadow: 0 6px 20px rgba(14, 107, 168, 0.08);
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


def build_kpi_cards(df: pd.DataFrame) -> list[dict]:
    """Формирует KPI-метрики для карточек: сумма, среднее, max, кол-во строк."""
    cards = []
    if df is None or df.empty:
        return cards

    df = _clean_df_for_display(df)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    cards.append({"label": "Строк данных", "value": f"{len(df):,}", "delta": None})

    for col in num_cols[:2]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        cards.append({
            "label": f"∑ {col}",
            "value": _format_value_axis(series.sum()),
            "delta": None,
        })
        cards.append({
            "label": f"⌀ {col}",
            "value": _format_value_axis(series.mean()),
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
    lines = [f"В результате {len(df)} строк и {len(df.columns)} колонок."]
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    if num_cols:
        main_metric = num_cols[0]
        metric = pd.to_numeric(df[main_metric], errors="coerce").dropna()
        if not metric.empty:
            lines.append(
                f"Ключевая метрика '{main_metric}': сумма {_format_value_axis(metric.sum())}, "
                f"среднее {_format_value_axis(metric.mean())}, максимум {_format_value_axis(metric.max())}."
            )
            if len(metric) > 1:
                growth = metric.iloc[-1] - metric.iloc[0]
                trend = "рост" if growth > 0 else "снижение" if growth < 0 else "без выраженного изменения"
                lines.append(
                    f"Тренд: {trend} ({_format_value_axis(growth)} между первой и последней точкой)."
                )

    if cat_cols:
        main_dim = cat_cols[0]
        top_items = df[main_dim].astype(str).value_counts(dropna=False).head(3)
        if not top_items.empty:
            formatted = ", ".join([f"{idx} ({int(val)})" for idx, val in top_items.items()])
            lines.append(f"Наиболее частые категории по '{main_dim}': {formatted}.")

    if num_cols and cat_cols:
        dim = cat_cols[0]
        metric_name = num_cols[0]
        grouped = (
            df[[dim, metric_name]]
            .assign(_metric=pd.to_numeric(df[metric_name], errors="coerce"))
            .dropna(subset=["_metric"])
            .groupby(dim, dropna=False)["_metric"]
            .sum()
            .sort_values(ascending=False)
            .head(3)
        )
        if not grouped.empty:
            leaders = ", ".join([f"{k}: {_format_value_axis(v)}" for k, v in grouped.items()])
            lines.append(f"Лидеры по сумме '{metric_name}': {leaders}.")

    lines.append("Рекомендуется проверить выбросы и динамику по лидирующим категориям перед принятием решений.")
    return "\n".join(lines)


def build_chart(df: pd.DataFrame, chart_type: str):
    """
    Строит Plotly-график с умной агрегацией:
    - bar: горизонтальная диаграмма (barh), топ-N + «Остальные», форматированные значения
    - line: линейный график с маркерами
    При >TOP_N_CHART категорий автоматически агрегирует.
    """
    if df is None or df.empty or len(df.columns) < 2 or chart_type == "table":
        return None

    df = _clean_df_for_display(df)
    y_cols = df.select_dtypes(include=["number"]).columns.tolist()
    x_cols = [c for c in df.columns if c not in y_cols]
    if not y_cols:
        return None

    x_axis = x_cols[0] if x_cols else df.columns[0]
    y_main = y_cols[0]
    y_axis = y_cols[:2] if len(y_cols) > 1 else y_cols[0]

    if chart_type == "bar":
        # Агрегация: Топ-N + «Остальные»
        plot_df = _aggregate_top_n(df, x_axis, y_main, TOP_N_CHART)
        plot_df = plot_df.sort_values(y_main, ascending=True, key=lambda s: pd.to_numeric(s, errors="coerce"))

        # Горизонтальная bar chart (лучше читаются длинные подписи)
        fig = px.bar(
            plot_df,
            y=x_axis,
            x=y_axis,
            orientation="h",
            title=f"{', '.join(y_cols[:2])} по {x_axis} (топ-{min(TOP_N_CHART, len(df))})",
            text_auto=".3s",
        )
        fig.update_layout(
            template="plotly_white",
            legend_title_text="Показатели",
            margin=dict(l=10, r=20, t=50, b=30),
            yaxis_title="",
            xaxis_title=y_main,
            height=max(400, min(TOP_N_CHART + 1, len(plot_df)) * 35 + 100),
            bargap=0.15,
        )
        fig.update_traces(
            textposition="outside",
            textfont_size=11,
        )
        # Подсвечиваем «Остальные» серым
        colors = ["#0E6BA8"] * len(plot_df)
        if len(df) > TOP_N_CHART and len(colors) > 0:
            colors[0] = "#C0C0C0"  # первый элемент (самый нижний в sorted) = Остальные
        fig.update_traces(marker_color=colors, selector=dict(type="bar"))

    elif chart_type == "line":
        sort_cols = [x_axis] if x_axis in df.columns else None
        source = df.sort_values(by=sort_cols) if sort_cols else df
        fig = px.line(
            source,
            x=x_axis,
            y=y_axis,
            title=f"Динамика: {', '.join(y_cols[:2])} по {x_axis}",
            markers=True,
        )
        fig.update_layout(
            template="plotly_white",
            legend_title_text="Показатели",
            margin=dict(l=30, r=20, t=50, b=30),
            xaxis_title=x_axis,
            yaxis_title="Значение",
            hovermode="x unified",
        )
    else:
        return None

    return fig


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


def render_assistant_message(msg: dict, idx: int) -> None:
    st.markdown(msg.get("content", ""))

    if msg.get("warning"):
        st.warning(msg["warning"])

    if msg.get("dataframe") is not None and not msg["dataframe"].empty:
        # KPI-карточки
        kpi_cards = build_kpi_cards(msg["dataframe"])
        if kpi_cards:
            kpi_cols = st.columns(len(kpi_cards))
            for i, card in enumerate(kpi_cards):
                with kpi_cols[i]:
                    st.metric(label=card["label"], value=card["value"], delta=card.get("delta"))

        # График + Executive Summary
        chart = msg.get("chart")
        executive_summary = msg.get("executive_summary", "Сводка недоступна")
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True, key=f"chart_history_{idx}")
            with st.expander("Executive Summary"):
                st.info(executive_summary)
        else:
            with st.expander("Executive Summary"):
                st.info(executive_summary)

        # Таблица данных — в expander
        with st.expander(f"Таблица данных ({len(msg['dataframe'])} строк)"):
            st.dataframe(msg["dataframe"], use_container_width=True)

        suffix = msg.get("suffix", f"history_{idx}")
        render_export_buttons(msg["dataframe"], msg.get("prompt", "history"), suffix, executive_summary)

    if msg.get("graph_dot"):
        with st.expander("Схема декомпозиции запроса"):
            st.graphviz_chart(msg["graph_dot"])

    if msg.get("dax"):
        with st.expander("DAX запрос"):
            st.code(msg["dax"], language="dax")


def main() -> None:
    apply_custom_theme()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

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
        st.caption("✅ Экспорт доступен для результатов")

    st.title("📊 Genbi: OLAP куб в диалоге")
    st.markdown("AI-генерация DAX, анализ данных и экспорт")

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg.get("role", "assistant")):
            if msg.get("role") == "assistant":
                render_assistant_message(msg, i)
            else:
                st.markdown(msg.get("content", ""))

    prompt = st.chat_input("Пример: Топ менеджеров по продажам")
    if not prompt:
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
            "content": "",
            "dax": dax_query,
            "dataframe": None,
            "chart": None,
            "graph_dot": None,
            "executive_summary": "",
            "warning": ai_response.get("warning", ""),
            "prompt": prompt,
            "suffix": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        if thought:
            st.markdown(f"**Анализ:** {thought}")
            save_msg["content"] = f"Анализ: {thought}"

        if ai_response.get("error"):
            st.error(ai_response["error"])
            save_msg["content"] = f"Ошибка: {ai_response['error']}"
            st.session_state.messages.append(save_msg)
            return

        if save_msg["warning"]:
            st.warning(save_msg["warning"])

        if isinstance(df, pd.DataFrame) and not df.empty:
            save_msg["dataframe"] = df
            st.success(f"Данные получены: {len(df)} строк")

            # KPI-карточки
            kpi_cards = build_kpi_cards(df)
            if kpi_cards:
                kpi_cols = st.columns(len(kpi_cards))
                for i, card in enumerate(kpi_cards):
                    with kpi_cols[i]:
                        st.metric(label=card["label"], value=card["value"], delta=card.get("delta"))

            chart_type = ai_response.get("chart_type", "table")
            chart_fig = build_chart(df.copy(), chart_type)
            executive_summary = build_executive_summary(df)
            save_msg["executive_summary"] = executive_summary

            # График на всю ширину
            if chart_fig is not None:
                save_msg["chart"] = chart_fig
                st.plotly_chart(chart_fig, use_container_width=True, key=f"chart_live_{save_msg['suffix']}")
                with st.expander("Executive Summary"):
                    st.info(executive_summary)
            else:
                with st.expander("Executive Summary"):
                    st.info(executive_summary)

            # Таблица данных — в expander чтобы не загромождать
            with st.expander(f"Таблица данных ({len(df)} строк)"):
                st.dataframe(df, use_container_width=True)

            # Insights
            insights = build_insights(df)
            if insights:
                st.markdown("**Ключевые наблюдения**")
                for item in insights:
                    st.write(f"- {item}")

            graph_dot = build_query_graph_dot(ai_response)
            save_msg["graph_dot"] = graph_dot
            with st.expander("Схема декомпозиции запроса"):
                st.graphviz_chart(graph_dot)

            render_export_buttons(df, prompt, save_msg["suffix"], executive_summary)
        else:
            st.info("Результаты не получены. Проверьте DAX и фильтры.")

        if dax_query:
            with st.expander("📝 DAX"):
                st.code(dax_query, language="dax")

        st.session_state.messages.append(save_msg)


if __name__ == "__main__":
    main()
