# Ứng dụng minh hoạ — giao dịch thuật toán trên cổ phiếu FPT.

import streamlit as st

from src.config import APP_ICON, APP_TITLE

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

TRANG = [
    st.Page("trang/p1_tong_quan.py", title="Tổng quan", icon="🏠", default=True),
    st.Page("trang/p2_du_lieu.py", title="Dữ liệu và phân tích khám phá", icon="📊"),
    st.Page("trang/p3_dac_trung.py", title="Bộ đặc trưng", icon="🧬"),
    st.Page("trang/p4_chia_du_lieu.py", title="Chia dữ liệu chống rò rỉ", icon="✂️"),
    st.Page("trang/p5_baseline.py", title="Baseline", icon="📐"),
    st.Page("trang/p6_mo_hinh_de_xuat.py", title="Mô hình đề xuất ATFN", icon="🤖"),
    st.Page("trang/p7_nhip_giao_dich.py", title="Nhịp giao dịch", icon="⏱️"),
    st.Page("trang/p8_ablation.py", title="Thí nghiệm Ablation", icon="🔬"),
    st.Page("trang/p9_backtest.py", title="Backtest tương tác", icon="🎛️"),
    st.Page("trang/p10_ket_qua_cuoi.py", title="Kết quả trên tập kiểm tra", icon="🏁"),
    st.Page("trang/p11_bien_luan.py", title="Biện luận và hỏi đáp", icon="💬"),
]

st.navigation(TRANG).run()
