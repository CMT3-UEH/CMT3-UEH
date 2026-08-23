# Trang 10 — kết quả trên tập kiểm tra và độ tin cậy thống kê.

import streamlit as st

from src.config import SHARPE_TARGET
from src.evaluation.metrics import cumulative_return, drawdown_series
from src.ui import (
    bang_dep,
    canh_bao_thieu,
    ghi_chu,
    khoi_tao_trang,
    kho,
    muc_tieu_sharpe,
    so,
    the_chi_so,
)
from src.viz import cot_so_sanh, drawdown, duong_tai_san, khoang_tin_cay

khoi_tao_trang("Kết quả trên tập kiểm tra",
               "Chạy một lần, cấu hình đã đóng băng từ trước")

k = kho()

if canh_bao_thieu("kết quả test", "python -m src.experiments.run_final"):
    st.stop()

st.info(
    "Những điều đã công bố **trước** khi chạy tập kiểm tra: mô hình đề xuất là "
    "**ATFN-ABCD** (không phải bậc nào tốt nhất trên tập kiểm tra), toàn bộ baseline và bậc "
    "ablation đều được báo cáo không lọc bỏ dòng nào, và con số quyết định là Sharpe sau phí."
)

bang = k.bang["kết quả test"]
bang_dep(bang)

st.plotly_chart(
    cot_so_sanh(bang, "Sharpe", "Sharpe trên tập kiểm tra", nguong=SHARPE_TARGET),
    width="stretch",
)

ten = next((c for c in bang.index if c.startswith("ATFN-ABCD")), None)
if ten:
    muc_tieu_sharpe(float(bang.loc[ten, "Sharpe"]))
    moc = "A1 · Mua và nắm giữ FPT"
    if moc in bang.index:
        chenh = bang.loc[ten, "Sharpe"] - bang.loc[moc, "Sharpe"]
        the_chi_so({
            "Sharpe mô hình đề xuất": so(bang.loc[ten, "Sharpe"]),
            "Sharpe mua và nắm giữ": so(bang.loc[moc, "Sharpe"]),
            "Chênh lệch": ("+" if chenh >= 0 else "") + so(chenh),
            "Chỉ tiêu đề bài": so(SHARPE_TARGET, 1),
        })

if k.co_bang("độ tin cậy test"):
    st.divider()
    st.subheader("Sharpe đó đáng tin tới đâu")
    tc = k.bang["độ tin cậy test"]
    bang_dep(tc)
    if {"KTC 95% dưới", "KTC 95% trên"}.issubset(tc.columns):
        st.plotly_chart(khoang_tin_cay(tc, "Sharpe kèm khoảng tin cậy bootstrap theo khối"),
                        width="stretch")
    st.markdown(
        """
Ba cột cần đọc kỹ:

* **Sai số chuẩn** — Sharpe ước lượng trên 1.404 phiên có sai số quanh 0,43 đơn vị. Chênh lệch
  nhỏ hơn con số này giữa hai chiến lược không kết luận được gì.
* **Ngưỡng do dò tìm** — mức Sharpe mà một chiến lược hoàn toàn vô dụng vẫn đạt được chỉ nhờ đã
  thử nhiều cấu hình. Đây là cái phải trừ đi trước khi nói mô hình có giá trị.
* **Deflated Sharpe** — xác suất Sharpe thật vượt ngưỡng đó. Dưới 0,95 thì chưa đủ cơ sở
  tuyên bố chiến lược có kỹ năng.
"""
    )

st.divider()
st.subheader("Đường tăng trưởng vốn trên tập kiểm tra")
chon = st.multiselect(
    "Chọn chiến lược",
    list(k.tin_hieu.columns),
    default=[c for c in (ten, "A1 · Mua và nắm giữ FPT",
                         "A3 · Nắm giữ điều tiết biến động")
             if c and c in k.tin_hieu.columns],
)
if chon:
    nav, dd = {}, {}
    for t in chon:
        r = k.chay(t, "test").returns.iloc[1:]
        nav[t] = cumulative_return(r)
        dd[t] = drawdown_series(r)
    st.plotly_chart(duong_tai_san(nav, log=False), width="stretch")
    st.plotly_chart(drawdown(dd), width="stretch")

if k.co_bang("độ vững"):
    st.divider()
    st.subheader("Đổi giả định thì kết quả còn đứng được không")
    bang_dep(k.bang["độ vững"])
    ghi_chu(
        "Dòng *phí ×0* chỉ để tham chiếu, không phải kết quả có thể công bố. "
        "Dòng đáng chú ý nhất là *trễ thực thi 2 phiên*: nếu Sharpe không giảm khi làm chậm "
        "tín hiệu thì cần nghi ngờ có rò rỉ thông tin ở đâu đó."
    )
