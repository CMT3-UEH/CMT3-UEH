# Trang 7 — nhịp giao dịch do mô hình tự quyết. Trọng tâm nghiên cứu của đề tài.

import pandas as pd
import streamlit as st

from src.backtest.engine import BacktestConfig
from src.backtest.costs import BASE_COST
from src.evaluation.metrics import annual_return, sharpe_ratio
from src.ui import (
    bang_dep, chon_phan, ghi_chu, khoi_tao_trang, kho, nguyen, phan_tram, so,
    the_chi_so,
)
from src.viz import cong_nhip, phan_phoi_nam_giu

khoi_tao_trang("Nhịp giao dịch",
               "Tần suất giao dịch là kết quả của mô hình, không phải tham số cài sẵn")

k = kho()

st.markdown(
    """
Câu hỏi trung tâm của đồ án: **có thể để mô hình tự quyết nên giao dịch dày hay thưa không?**
Câu trả lời là có, bằng bốn cơ chế bổ trợ nhau.

| Cơ chế | Cách hoạt động | Hệ quả với tần suất |
|---|---|---|
| Phạt vòng quay vị thế trong hàm mất mát | Mỗi lần đổi vị thế đều bị phạt ngay trong lúc tối ưu | Kết quả nảy sinh, không phải tham số |
| Vùng không giao dịch học được | Chỉ vào lệnh khi lệch quá ngưỡng `τ` do mạng sinh ra theo từng phiên | Mô hình tự chọn độ rộng vùng không giao dịch |
| Đầu ra đa tầm dự báo + cổng nhịp | Dự báo đồng thời ở 1, 5, 10, 20 phiên; cổng học nên tin tầm dự báo nào | Chu kỳ nắm giữ đổi theo phiên |
| Cổng chế độ thị trường | Thu hẹp tỷ trọng nắm giữ khi trạng thái bất lợi | Số phiên đứng ngoài tự điều chỉnh |
"""
)

ten = next((c for c in k.tin_hieu.columns if c.startswith("ATFN-ABCD")), None)
if ten is None:
    st.warning("Chưa có mô hình đề xuất. Chạy `python -m src.experiments.run_atfn`.")
    st.stop()

st.divider()
st.subheader("Nhịp mà mô hình đã chọn")
part = chon_phan("Giai đoạn", ["valid", "test"], mac_dinh=1, khoa="phan_p7")
kq = k.chay(ten, part)
nhip = kq.tempo()

the_chi_so({
    "Lệnh mỗi năm": so(nhip["Số lần khớp lệnh mỗi năm"], 1),
    "Phiên giữa hai lệnh": so(nhip["Số phiên giữa hai lệnh"], 1),
    "Thời gian nắm giữ trung bình": f"{so(nhip['Thời gian nắm giữ trung bình (phiên)'], 0)} phiên",
    "Vòng quay vị thế mỗi năm": so(nhip["Vòng quay vị thế mỗi năm"], 2),
})
the_chi_so({
    "Số lần vào vị thế": nguyen(nhip["Số lần vào vị thế"]),
    "Tỷ lệ thời gian có vị thế": phan_tram(nhip["Tỷ lệ thời gian có vị thế"], 1),
    "Tổng chi phí giao dịch": phan_tram(nhip["Tổng chi phí giao dịch"]),
    "Phiên bị ràng buộc chặn lại": nguyen(nhip["Số phiên bị ràng buộc chặn lại"]),
}, so_cot=4)

c1, c2 = st.columns([1, 1])
with c1:
    st.plotly_chart(phan_phoi_nam_giu(kq.holding_periods(),
                                      "Phân phối độ dài mỗi lần nắm giữ"),
                    width="stretch")
with c2:
    if k.nhip is not None:
        cot_nhip = [c for c in k.nhip.columns if c.startswith("h=")]
        if cot_nhip:
            tb = k.nhip[cot_nhip].reindex(k.d.index(part)).dropna().mean()
            st.markdown("**Trọng số trung bình của cổng chọn nhịp**")
            bang_dep(pd.DataFrame({"Trọng số": tb}))
            ghi_chu(
                "Tầm dự báo nào có trọng số cao nhất chính là chu kỳ nắm giữ mà mô hình "
                "đang tin tưởng ở giai đoạn đó."
            )

if k.nhip is not None:
    cot_nhip = [c for c in k.nhip.columns if c.startswith("h=")]
    if cot_nhip:
        st.divider()
        st.subheader("Cổng chọn nhịp theo thời gian")
        idx = k.d.index(part)
        cong = k.nhip[cot_nhip].reindex(idx).dropna()
        st.plotly_chart(
            cong_nhip(cong, k.ds.loc[cong.index, "close"]),
            width="stretch",
        )
        ghi_chu(
            "Dải màu bên dưới là trọng số cổng, cộng lại bằng 1 ở mỗi phiên. "
            "Khi mô hình chuyển trọng số sang tầm dự báo dài, nó đang chuyển sang nhịp chậm."
        )

    if "nguong" in k.nhip.columns and k.nhip["nguong"].notna().any():
        st.subheader("Ngưỡng vùng không giao dịch mà mô hình học được")
        ng = k.nhip["nguong"].reindex(k.d.index(part)).dropna()
        the_chi_so({
            "Trung bình": so(ng.mean()),
            "Trung vị": so(ng.median()),
            "Nhỏ nhất": so(ng.min()),
            "Lớn nhất": so(ng.max()),
        })
        ghi_chu(
            "Ngưỡng lớn nghĩa là mô hình đòi hỏi tín hiệu phải lệch nhiều mới chịu vào lệnh. "
            "Đây chính là đại lượng quyết định tần suất giao dịch, và nó do mạng sinh ra "
            "theo từng phiên chứ không phải một hằng số ta đặt."
        )

st.divider()
st.subheader("Đổi mức phí giả định — nhịp có dịch chuyển theo không?")
st.markdown(
    "Nếu cơ chế hoạt động đúng như thiết kế, chi phí cao hơn phải khiến chiến lược giao dịch "
    "thưa hơn. Bảng dưới đo trực tiếp điều đó."
)

he_so = st.select_slider("Hệ số nhân chi phí giao dịch",
                         [0.0, 0.5, 1.0, 2.0, 4.0], value=1.0)
idx = k.d.index(part)
hang = {}
for h in (0.0, 1.0, 2.0, 4.0):
    kq2 = k.chay(ten, part, BacktestConfig(cost=BASE_COST.scaled(h)))
    n2 = kq2.tempo()
    hang[f"phí ×{h:g}"] = {
        "Lệnh mỗi năm": n2["Số lần khớp lệnh mỗi năm"],
        "Phiên giữa hai lệnh": n2["Số phiên giữa hai lệnh"],
        "Sharpe": sharpe_ratio(kq2.returns.iloc[1:]),
        "CAGR": annual_return(kq2.returns.iloc[1:]),
        "Tổng chi phí giao dịch": n2["Tổng chi phí giao dịch"],
    }
bang_dep(pd.DataFrame(hang).T)

st.warning(
    "Đọc bảng này cho đúng: **số lệnh không đổi theo phí** vì tín hiệu đã được huấn luyện "
    "xong với một mức phí cố định và ở đây ta chỉ tính lại chi phí trên cùng tín hiệu đó. "
    "Muốn thấy mô hình thật sự giãn nhịp theo chi phí thì phải huấn luyện lại ở từng mức "
    "phí — đó là thí nghiệm riêng, không phải thứ đọc được từ bảng này."
)

if k.co_bang("ablation nhịp"):
    st.divider()
    st.subheader("Ablation về nhịp — trên tập kiểm tra")
    st.markdown(
        "Cùng một vị thế mục tiêu do mô hình sinh ra, chỉ đổi cách quyết định khi nào được "
        "phép đổi vị thế. Giữ nguyên tín hiệu nên chênh lệch giữa các dòng đúng bằng đóng "
        "góp của riêng cơ chế chọn nhịp."
    )
    bang_dep(k.bang["ablation nhịp"])
