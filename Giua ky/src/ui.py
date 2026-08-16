"""Thành phần giao diện dùng chung cho mọi trang của dashboard."""

import pandas as pd
import streamlit as st

from src.config import APP_ICON, APP_TITLE, RISK_FREE_ANNUAL, TICKER
from src.pipeline import Analysis, load_analysis

CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
  h1 {font-size: 1.9rem !important; font-weight: 700;}
  h2 {font-size: 1.35rem !important; margin-top: 1.6rem;}
  h3 {font-size: 1.08rem !important;}
  [data-testid="stMetricValue"] {font-size: 1.5rem;}
  [data-testid="stMetricLabel"] {font-size: 0.82rem; color: #64748b;}
  .note {background:#f8fafc; border-left:4px solid #2563eb; padding:0.85rem 1.1rem;
         border-radius:6px; font-size:0.92rem; line-height:1.6; margin:0.6rem 0 1rem 0;}
  .note b {color:#1e3a8a;}
  .formula {background:#f1f5f9; border-radius:6px; padding:0.7rem 1rem; font-size:0.95rem;
            text-align:center; margin:0.5rem 0 1rem 0;}
  .interp li {margin-bottom: 0.45rem; line-height:1.6;}
  .src {color:#64748b; font-size:0.8rem; margin-top:0.4rem;}
</style>
"""

PERIODS = {
    "Toàn bộ lịch sử": None,
    "15 năm gần nhất": 15,
    "10 năm gần nhất": 10,
    "5 năm gần nhất": 5,
    "3 năm gần nhất": 3,
    "1 năm gần nhất": 1,
}


def page_config(title: str) -> None:
    st.set_page_config(page_title=f"{title} — {APP_TITLE}", page_icon=APP_ICON,
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Đang nạp dữ liệu và ước lượng mô hình...")
def _cached_analysis(ticker: str, rf: float, start: str | None) -> Analysis:
    return load_analysis(ticker, rf, start=start)


def sidebar() -> Analysis:
    """Vẽ thanh điều khiển bên trái và trả về đối tượng phân tích tương ứng."""
    with st.sidebar:
        st.markdown(f"## {APP_ICON} {APP_TITLE}")
        st.caption("Đồ án giữa kỳ — Nghiên cứu khoa học")
        st.divider()

        period = st.selectbox("Giai đoạn phân tích", list(PERIODS.keys()), index=0,
                              help="Mọi mô hình CAPM, APT, Monte Carlo đều ước lượng "
                                   "lại theo giai đoạn được chọn.")
        rf = st.slider("Lãi suất phi rủi ro (%/năm)", 0.0, 12.0, RISK_FREE_ANNUAL * 100, 0.25,
                       help="Dùng làm Rf trong CAPM, APT và tỷ số Sharpe. Tham chiếu: "
                            "lợi suất trái phiếu chính phủ kỳ hạn 10 năm.") / 100

        years = PERIODS[period]
        start = None
        if years:
            start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")

        a = _cached_analysis(TICKER, rf, start)

        st.divider()
        st.metric(f"Giá {a.ticker}", f"{a.last_price:,.2f}",
                  f"{a.returns.iloc[-1] * 100:+.2f}%")
        st.caption(
            f"Phiên gần nhất: **{a.last_date:%d/%m/%Y}**  \n"
            f"Dữ liệu: {len(a.prices):,} phiên từ {a.prices.index[0]:%d/%m/%Y}"
        )
        st.divider()
        st.caption(
            "**Nguồn dữ liệu**  \n"
            "• Giá cổ phiếu: vnstock (đã điều chỉnh)  \n"
            "• Chỉ số VNINDEX: DNSE entrade  \n"
            "• Hồ sơ doanh nghiệp: vnstock  \n"
            "• Nhân tố vĩ mô: Yahoo Finance"
        )
        if st.button("🔄 Tải lại dữ liệu từ nguồn", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    return a


# Khối hiển thị nhỏ
def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def formula(text: str) -> None:
    st.markdown(f'<div class="formula">{text}</div>', unsafe_allow_html=True)


def interpretation(items: list[str], title: str = "Diễn giải kết quả") -> None:
    st.markdown(f"#### {title}")
    st.markdown(
        '<ul class="interp">' + "".join(f"<li>{m}</li>" for m in items) + "</ul>",
        unsafe_allow_html=True,
    )


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    """Hàng thẻ số liệu: mỗi phần tử là (nhãn, giá trị, chênh lệch)."""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col:
            st.metric(label, value, delta)
