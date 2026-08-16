"""Dashboard Quản lý Đầu tư — điểm khởi động ứng dụng Streamlit."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import APP_ICON, APP_TITLE, TICKER  # noqa: E402
from src.ui import CSS  # noqa: E402

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": f"{APP_TITLE} — phân tích cổ phiếu {TICKER} bằng CAPM, APT "
                         "và mô phỏng Monte Carlo."},
)
st.markdown(CSS, unsafe_allow_html=True)

PAGES = [
    st.Page("pages/p1_overview.py", title="Tổng quan", icon="🏠", default=True),
    st.Page("pages/p2_company.py", title="Thông tin cổ phiếu", icon="🏢"),
    st.Page("pages/p3_risk.py", title="Rủi ro và hiệu quả", icon="⚠️"),
    st.Page("pages/p4_capm.py", title="Mô hình CAPM", icon="📐"),
    st.Page("pages/p5_apt.py", title="Mô hình APT", icon="🧩"),
    st.Page("pages/p6_montecarlo.py", title="Mô phỏng Monte Carlo", icon="🎲"),
    st.Page("pages/p7_allocation.py", title="Quản lý đầu tư", icon="💼"),
    st.Page("pages/p8_chatbot.py", title="Chatbot hỏi đáp", icon="🤖"),
]

st.navigation(PAGES).run()
