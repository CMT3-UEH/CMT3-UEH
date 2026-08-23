# Trang 1 — tổng quan bài toán và mốc tham chiếu.

import streamlit as st

from src.config import SHARPE_TARGET, TICKER
from src.formatting import pct
from src.ui import (
    bang_dep,
    ghi_chu,
    kho,
    khoi_tao_trang,
    nguyen,
    so,
    thanh_ben_trang_thai,
    the_chi_so,
)
from src.viz import duong_tai_san

khoi_tao_trang(
    "Giao dịch thuật toán trên cổ phiếu FPT",
    "Đồ án cuối kỳ — mô hình học sâu tự quyết nhịp giao dịch",
)
thanh_ben_trang_thai()

k = kho()
gia = k.ds

st.markdown(
    """
Bài toán: mỗi phiên quyết định phân bổ bao nhiêu phần vốn vào cổ phiếu **FPT**,
phần còn lại giữ tiền mặt hưởng lãi suất phi rủi ro.

Điểm khác biệt của đề tài: **nhịp giao dịch không được cài sẵn mà do mô hình tự
quyết**. Số lệnh mỗi năm và thời gian nắm giữ trung bình là kết quả nghiên cứu,
không phải siêu tham số.
"""
)

the_chi_so({
    "Số phiên dùng được": nguyen(len(gia)),
    "Khoảng thời gian": f"{gia.index.min():%m/%Y} – {gia.index.max():%m/%Y}",
    "Số đặc trưng": nguyen(len(k.features)),
    "Chỉ tiêu Sharpe": f"≥ {so(SHARPE_TARGET, 1)}",
})

st.subheader("Quy ước của bài toán")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Đầu vào**")
    st.markdown(
        "Đặc trưng tính đến hết giá đóng cửa phiên `t`, cửa sổ nhìn lại 60 phiên. "
        "VNINDEX và bốn nhân tố vĩ mô là biến ngoại sinh, không giao dịch."
    )
with c2:
    st.markdown("**Đầu ra**")
    st.markdown(
        "Vị thế `w[t+1] ∈ [0, 1]` — tỷ trọng vốn nắm FPT. "
        "Phần `1 − w` giữ tiền mặt và **có hưởng lãi**."
    )
with c3:
    st.markdown("**Thực thi**")
    st.markdown(
        "Tín hiệu ở `close(t)` → lệnh khớp ở `open(t+1)`. "
        "Phí 0,15%/chiều, thuế bán 0,1%, trượt giá 0,05%, ràng buộc T+2,5."
    )

st.divider()

st.subheader(f"Cổ phiếu {TICKER}")
r = gia["close"].pct_change().dropna()
nav = (1 + r).cumprod()
st.plotly_chart(duong_tai_san({f"Mua và nắm giữ {TICKER}": nav}, log=True),
                width="stretch")

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("**Thống kê toàn kỳ**")
    bang_dep(k.tom_tat_tai_san("all"))
with c2:
    if k.co_bang("mốc mua và giữ"):
        st.markdown("**Mốc mua và nắm giữ theo từng giai đoạn, đã trừ phí**")
        bang_dep(k.bang["mốc mua và giữ"])
        ghi_chu(
            "Đây là mốc mà mọi mô hình phải vượt qua. Nếu chiến lược thuật toán "
            "không thắng nổi việc mua rồi để yên thì kiến trúc mạng không có ý nghĩa gì."
        )

if k.co_bang("chi phí theo nhịp"):
    st.divider()
    st.subheader("Vì sao nhịp giao dịch là vấn đề trung tâm")
    b = k.bang["chi phí theo nhịp"]
    bang_dep(b)
    if "Chi phí bào mòn mỗi năm" in b.columns:
        cao, thap = b["Chi phí bào mòn mỗi năm"].max(), b["Chi phí bào mòn mỗi năm"].min()
        st.markdown(
            f"Mọi dòng đều nắm giữ khoảng một nửa thời gian và có cùng lợi suất "
            f"trước phí, nên chênh lệch giữa các dòng đúng bằng cái giá của việc "
            f"giao dịch dày hơn: **{pct(cao, 2, vi=True)}/năm** khi đảo vị thế mỗi "
            f"phiên, so với **{pct(thap, 2, vi=True)}/năm** khi đảo mỗi quý."
        )
        st.info(
            "Trên cổ phiếu này, giao dịch theo ngày gần như chắc chắn thua trừ khi "
            "tín hiệu cực mạnh. Vì vậy việc để mô hình tự chọn nhịp không phải ý "
            "tưởng trang trí mà là điều kiện cần để chiến lược tồn tại."
        )

st.divider()
st.caption(
    "Chuyển trang bằng thanh bên. Trang nào thiếu dữ liệu sẽ báo đúng lệnh cần chạy "
    "để sinh ra dữ liệu đó."
)
