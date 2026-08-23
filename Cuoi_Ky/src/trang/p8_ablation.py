# Trang 8 — thí nghiệm ablation, cộng dồn và theo nhóm đặc trưng.

import pandas as pd
import streamlit as st

from src.ui import bang_dep, canh_bao_thieu, ghi_chu, khoi_tao_trang, kho
from src.viz import thac_nuoc_ablation

khoi_tao_trang("Thí nghiệm Ablation",
               "Mỗi thành phần đóng góp bao nhiêu, đo bằng số chứ không bằng lời")

k = kho()

if canh_bao_thieu("ablation valid", "python -m src.experiments.run_atfn"):
    st.stop()

bang = k.bang["ablation valid"]

st.subheader("Ablation cộng dồn")
st.markdown(
    """
Bốn bậc, mỗi bậc thêm đúng một thành phần. Bậc thứ tư được **tách riêng khỏi bậc thứ năm**
có chủ đích: thành phần D gồm hai nửa — đổi hàm mất mát sang Sharpe có phí, và thêm vùng
không giao dịch. Gộp chung thì không biết nửa nào có tác dụng.
"""
)

thu_tu = [c for c in (
    "ATFN-A · TCN+GRU",
    "ATFN-AB · +đa tầm dự báo & cổng nhịp",
    "ATFN-ABC · +cổng chế độ",
    "ATFN-ABC+L · +mất mát Sharpe có phí",
    "ATFN-ABCD · +vùng không giao dịch",
) if c in bang.index]

if thu_tu:
    b = bang.loc[thu_tu]
    bang_dep(b)
    st.plotly_chart(thac_nuoc_ablation(b, "Sharpe",
                                       "Đóng góp Sharpe của từng thành phần"),
                    width="stretch")

    goc = b["Sharpe"].iloc[0]
    dong = []
    for ten in thu_tu:
        dong.append({
            "Cấu hình": ten,
            "Sharpe": b.loc[ten, "Sharpe"],
            "Δ so với bậc A": b.loc[ten, "Sharpe"] - goc,
            "Lệnh mỗi năm": b.loc[ten, "Lệnh mỗi năm"],
            "Phiên giữa hai lệnh": b.loc[ten, "Phiên giữa hai lệnh"],
        })
    bang_dep(pd.DataFrame(dong).set_index("Cấu hình"))

st.warning(
    "Đọc bảng ablation cho đúng: sai số chuẩn của Sharpe trên tập kiểm định vào khoảng "
    "0,59 đơn vị. Một chênh lệch nhỏ hơn con số đó **không** chứng minh được thành phần "
    "tương ứng có tác dụng. Phần nào không cải thiện được báo cáo trung thực chứ không bị "
    "gỡ khỏi bảng."
)

if k.co_bang("ablation nhịp"):
    st.divider()
    st.subheader("Ablation về nhịp giao dịch")
    st.markdown(
        "Giữ nguyên vị thế mục tiêu của mô hình, chỉ đổi cơ chế quyết định khi nào được "
        "phép đổi vị thế. Đây là bảng chứng minh trực tiếp cho luận điểm trung tâm của đề tài."
    )
    bang_dep(k.bang["ablation nhịp"])

st.divider()
st.subheader("Dao động giữa các hạt giống")
from src.config import REPORT_DIR

duong = REPORT_DIR / "09_hat_giong_atfn.csv"
if duong.exists():
    hg = pd.read_csv(duong)
    tom = hg.groupby("mô hình")["Sharpe kiểm định"].agg(["mean", "std", "min", "max"])
    tom.columns = ["Trung bình", "Độ lệch chuẩn", "Thấp nhất", "Cao nhất"]
    bang_dep(tom)
    ghi_chu(
        "Nếu độ lệch giữa các hạt giống lớn hơn khoảng cách giữa các cấu hình ablation thì bảng "
        "xếp hạng bậc không có ý nghĩa, và cách báo cáo trung thực duy nhất là ghi trung "
        "bình kèm độ lệch chứ không ghi một con số."
    )
