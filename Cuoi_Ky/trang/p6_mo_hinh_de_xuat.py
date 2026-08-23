# Trang 6 — kiến trúc và kết quả của mô hình đề xuất ATFN.

import streamlit as st

from src.evaluation.metrics import cumulative_return
from src.ui import (
    bang_dep, canh_bao_thieu, chon_phan, ghi_chu, khoi_tao_trang, kho,
    phan_tram, so, the_chi_so,
)
from src.viz import duong_tai_san, gia_va_vi_the

khoi_tao_trang("Mô hình đề xuất — ATFN",
               "Adaptive-Tempo Fusion Network: mạng hợp nhất đa nhịp tự điều tiết tần suất")

k = kho()

st.subheader("Kiến trúc")
st.code(
    """
        đặc trưng FPT + VNINDEX + vĩ mô, cửa sổ 60 phiên
                          │
   ┌──────────────────────┴──────────────────────┐
   │  (A) Bộ mã hoá đa phân giải                  │  TCN giãn nở ∥ GRU
   │      TCN 4 lớp, độ giãn 1-2-4-8              │  trường tiếp nhận phủ 60 phiên
   └──────────────────────┬──────────────────────┘
                          │
   ┌──────────────────────┴──────────────────────┐
   │  (B) Đầu ra đa tầm dự báo + cổng chọn nhịp     │  dự báo ở h = 1, 5, 10, 20
   │      softmax trên 4 tầm dự báo                 │  → chọn chu kỳ nắm giữ
   └──────────────────────┬──────────────────────┘
                          │
   ┌──────────────────────┴──────────────────────┐
   │  (C) Cổng theo chế độ thị trường             │  3 chuyên gia + hệ số tỷ trọng
   │      đọc riêng 16 đặc trưng chế độ           │  → thu hẹp vị thế khi bất lợi
   └──────────────────────┬──────────────────────┘
                          │
   ┌──────────────────────┴──────────────────────┐
   │  (D) Vùng không giao dịch, ngưỡng τ học được │  chỉ đổi vị thế khi |Δw| > τ
   │      mất mát = −Sharpe + λ₁·turnover + λ₂    │  triển khai tuần tự, có đạo hàm
   └──────────────────────┬──────────────────────┘
                          │
                trung bình 5 hạt giống
""",
    language=None,
)

st.markdown(
    """
Điểm kỹ thuật đáng chú ý nhất nằm ở thành phần (D): **vùng không giao dịch được triển khai
tuần tự ngay bên trong lô huấn luyện**, không phải áp sau khi huấn luyện xong. Vị thế phiên `t`
phụ thuộc vị thế phiên `t−1`, nên chỉ có triển khai theo đường đi thì đạo hàm mới truyền được
tới ngưỡng `τ`. Đây cũng là lý do lô huấn luyện bắt buộc là **đoạn phiên liên tiếp**, không
xáo trộn: Sharpe của một tập ngày rời rạc lấy ngẫu nhiên trong mười tám năm không phải Sharpe
của chiến lược nào cả.
"""
)

if canh_bao_thieu("ablation valid", "python -m src.experiments.run_atfn"):
    st.stop()

st.divider()
st.subheader("Kết quả trên tập kiểm định")
bang = k.bang["ablation valid"]
bang_dep(bang)

ten_de_xuat = next((c for c in k.tin_hieu.columns if c.startswith("ATFN-ABCD")), None)
if ten_de_xuat:
    kq = k.chay(ten_de_xuat, "valid")
    nhip = kq.tempo()
    the_chi_so({
        "Lệnh mỗi năm": so(nhip["Số lần khớp lệnh mỗi năm"], 1),
        "Phiên giữa hai lệnh": so(nhip["Số phiên giữa hai lệnh"], 1),
        "Tỷ lệ thời gian có vị thế": phan_tram(nhip["Tỷ lệ thời gian có vị thế"], 1),
        "Tổng chi phí giao dịch": phan_tram(nhip["Tổng chi phí giao dịch"]),
    })

    st.divider()
    st.subheader("Vị thế theo thời gian")
    part = chon_phan("Giai đoạn", ["valid", "test"], khoa="phan_p6")
    kq2 = k.chay(ten_de_xuat, part)
    st.plotly_chart(
        gia_va_vi_the(k.gia(part), kq2.weights,
                      f"Giá FPT và vị thế mô hình — tập {part}"),
        width="stretch",
    )
    ghi_chu(
        "Dải dưới là tỷ trọng vốn thực tế nắm giữ sau khi đã qua vùng không giao dịch "
        "và mọi ràng buộc thị trường, không phải tín hiệu thô của mạng."
    )

    st.subheader("Đường tăng trưởng vốn so với mua và nắm giữ")
    moc = "A1 · Mua và nắm giữ FPT"
    nav = {ten_de_xuat: cumulative_return(kq2.returns.iloc[1:])}
    if moc in k.tin_hieu.columns:
        nav[moc] = cumulative_return(k.chay(moc, part).returns.iloc[1:])
    st.plotly_chart(duong_tai_san(nav, log=False), width="stretch")
