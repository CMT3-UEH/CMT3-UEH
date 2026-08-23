# Trang 5 — baseline nhóm A và nhóm B, kèm cảnh báo về độ tin cậy của bảng xếp hạng.

import pandas as pd
import streamlit as st

from src.evaluation.metrics import cumulative_return, drawdown_series
from src.ui import bang_dep, canh_bao_thieu, ghi_chu, khoi_tao_trang, kho, so
from src.viz import cot_so_sanh, drawdown, duong_tai_san

khoi_tao_trang("Baseline", "Mô hình cổ điển, dựa trên quy tắc, máy học và học sâu")

k = kho()

if canh_bao_thieu("baseline valid", "python -m src.experiments.run_baselines"):
    st.stop()

st.subheader("Nhóm A — cổ điển và dựa trên quy tắc, nhóm B — máy học và học sâu")
bang = k.bang["baseline valid"]
if k.co_bang("học sâu valid"):
    bang = pd.concat([bang, k.bang["học sâu valid"]])
bang = bang.sort_values("Sharpe", ascending=False)
bang_dep(bang)

st.plotly_chart(cot_so_sanh(bang, "Sharpe", "Sharpe trên tập kiểm định"),
                width="stretch")

st.divider()
st.subheader("Bảng xếp hạng này đáng tin tới đâu")

if k.co_bang("độ tin cậy valid"):
    tc = k.bang["độ tin cậy valid"]
    bang_dep(tc)
    st.error(
        f"**Không một chiến lược nào có t-stat vượt 2.** Sai số chuẩn trung bình của "
        f"Sharpe trên tập kiểm định là **{so(tc['Sai số chuẩn'].mean())}**, trong khi "
        f"cả bảng xếp hạng chỉ trải rộng "
        f"**{so(tc['Sharpe'].max() - tc['Sharpe'].min())}** đơn vị. Toàn bộ bảng nằm "
        f"gọn trong hai sai số chuẩn — không cấu hình nào phân biệt được với cấu hình nào."
    )
    st.markdown(
        """
Đây không phải một chi tiết kỹ thuật nhỏ. Nó có ba hệ quả trực tiếp cho cả đồ án:

1. **Không được chọn siêu tham số theo một cửa sổ kiểm định duy nhất.** Kiểm định CSCV cho
   xác suất quá khớp khi backtest quanh 52–57%, tức chọn kiểu đó còn kém hơn tung đồng xu.
   Vì vậy phần chọn cấu hình đã chuyển sang **kiểm định cuốn chiếu trên 15 cửa sổ liên tiếp**,
   chấm theo Sharpe gộp của toàn bộ đoạn ngoài mẫu.
2. **Không dò tìm kiến trúc học sâu.** Kích thước mọi mô hình nhóm B được cố định trước theo
   quy mô dữ liệu. Dò kiến trúc trên nền nhiễu này chỉ là xếp hạng may rủi, lại còn thổi
   phồng số phép thử khiến Deflated Sharpe về sau mất ý nghĩa.
3. **Mọi con số cuối cùng phải đi kèm khoảng tin cậy**, không được báo cáo trần trụi.
"""
    )

if k.co_bang("ứng viên tiến dần"):
    st.divider()
    st.subheader("Chọn cấu hình bằng kiểm định cuốn chiếu")
    st.markdown(
        "Mỗi cấu hình được chấm trên 15 cửa sổ liên tiếp trong vùng huấn luyện và kiểm định. "
        "Tiêu chí chọn là **Sharpe gộp của toàn bộ đoạn ngoài mẫu** (khoảng 1.890 phiên, "
        "sai số chuẩn ≈ 0,37) chứ không phải trung bình Sharpe của từng cửa sổ 126 phiên "
        "vốn dao động rất mạnh."
    )
    bang_dep(k.bang["ứng viên tiến dần"].head(20))
    ghi_chu(
        "Cột *Tỷ lệ cửa sổ dương* chỉ để báo cáo, không dùng để chọn. Thêm nó vào tiêu chí "
        "chọn là thêm một nút vặn nữa để dò, mà chính việc dò quá nhiều mới là thứ đang cần khống chế."
    )

st.divider()
st.subheader("So sánh trực quan trên tập kiểm định")
chon = st.multiselect(
    "Chọn chiến lược để vẽ",
    list(k.tin_hieu.columns),
    default=[c for c in ("A1 · Mua và nắm giữ FPT", "A3 · Nắm giữ điều tiết biến động")
             if c in k.tin_hieu.columns],
)
if chon:
    nav, dd = {}, {}
    for ten in chon:
        r = k.chay(ten, "valid").returns.iloc[1:]
        nav[ten] = cumulative_return(r)
        dd[ten] = drawdown_series(r)
    st.plotly_chart(duong_tai_san(nav, log=False), width="stretch")
    st.plotly_chart(drawdown(dd), width="stretch")
