# Trang 3 — bộ đặc trưng và kiểm định tính nhân quả.

import pandas as pd
import streamlit as st

from src.ui import bang_dep, ghi_chu, khoi_tao_trang, kho, nguyen, the_chi_so

khoi_tao_trang("Bộ đặc trưng",
               "106 biến, 9 nhóm, tất cả đã qua kiểm định tính nhân quả")

k = kho()
nhom = k.nhom_dac_trung

the_chi_so({
    "Tổng số đặc trưng": nguyen(len(k.features)),
    "Số nhóm": nguyen(len(nhom)),
    "Cửa sổ nhìn lại": "60 phiên",
    "Nhãn": "1, 5, 10, 20 phiên",
})

st.subheader("Nguyên tắc bất di bất dịch")
st.markdown(
    """
Mọi đặc trưng chỉ được tính từ thông tin **có sẵn tính đến giá đóng cửa phiên đó**.
Nguyên tắc này không được để ở dạng lời hứa mà được kiểm bằng thực nghiệm:

> Cắt chuỗi giá tại một phiên ngẫu nhiên, tính lại toàn bộ đặc trưng chỉ trên phần quá khứ.
> Nếu giá trị tại phiên đó khác với giá trị tính trên toàn chuỗi thì đặc trưng đã nhìn thấy tương lai.

Cả 106 đặc trưng đều đạt phép kiểm này trên 25 điểm cắt ngẫu nhiên.
Xem `src/features/builder.py::assert_causal`.

Hai chỗ từng suýt rò rỉ và đã được xử lý riêng:

* **Đặc trưng khung tuần và khung tháng** — thanh tuần gán nhãn ở ngày cuối tuần nhưng chứa cả
  dữ liệu giữa tuần. Mọi chuỗi khung lớn đều bị đẩy lùi một kỳ trước khi ghép về lịch ngày.
* **Nhân tố vĩ mô** — S&P 500 và hàng hoá đóng cửa sau giờ giao dịch Việt Nam, nên giá "cùng ngày"
  thực chất là thông tin của hôm sau. Cả nhóm bị trễ một phiên.
"""
)

st.divider()
st.subheader("Các nhóm đặc trưng")
st.caption(
    "Tên hiển thị ở đây giữ nguyên mã dùng trong mã nguồn (`mom_20`, `rel_beta_60`, ...) "
    "để đối chiếu trực tiếp với `src/features/`. Tiền tố cho biết nhóm: `mom`/`ma`/`rev` "
    "là giá, `vol` biến động, `liq` thanh khoản, `ms` vi cấu trúc, `wk`/`mo` khung tuần và "
    "tháng, `rel` quan hệ với thị trường, `reg` chế độ, `mac` vĩ mô, `cal` lịch."
)

for ten, cot in nhom.items():
    with st.expander(f"{ten} — {len(cot)} biến"):
        st.code(", ".join(cot), language=None)

st.divider()
st.subheader("Phân phối của một đặc trưng")
chon = st.selectbox("Chọn đặc trưng", k.features,
                    index=k.features.index("mom_20") if "mom_20" in k.features else 0)

c1, c2 = st.columns([2, 1])
with c1:
    import plotly.graph_objects as go
    from src.config import COLOR_PRIMARY

    s = k.ds[chon].dropna()
    fig = go.Figure(go.Histogram(x=s.values, nbinsx=80, marker_color=COLOR_PRIMARY))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                      template="plotly_white",
                      xaxis_title=chon, yaxis_title="Số phiên")
    st.plotly_chart(fig, width="stretch")
with c2:
    st.markdown("**Thống kê mô tả**")
    bang_dep(pd.DataFrame({chon: k.ds[chon].describe()}))

st.divider()
st.subheader("Tương quan giữa các đặc trưng trong một nhóm")
nhom_chon = st.selectbox("Chọn nhóm", list(nhom.keys()))
cot = nhom[nhom_chon]
if len(cot) > 1:
    bang_dep(k.ds[cot].corr())
    ghi_chu(
        "Tương quan cao trong cùng một nhóm là bình thường và không phải vấn đề với "
        "mô hình có chính quy hoá. Nó chỉ thành vấn đề khi đọc độ quan trọng đặc trưng: "
        "hai biến gần trùng nhau sẽ chia đôi công lao cho nhau."
    )
