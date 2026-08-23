# Trang 9 — backtest tương tác. Người dùng đổi giả định và thấy kết quả đổi theo.

import pandas as pd
import streamlit as st

from src.backtest.constraints import Constraints
from src.backtest.costs import CostModel
from src.backtest.engine import BacktestConfig
from src.config import FEE_RATE, SELL_TAX, SLIPPAGE
from src.evaluation.metrics import cumulative_return, drawdown_series, summary_table
from src.ui import bang_dep, chon_phan, ghi_chu, khoi_tao_trang, kho, muc_tieu_sharpe
from src.viz import drawdown, duong_tai_san, sharpe_truot

khoi_tao_trang("Backtest tương tác",
               "Đổi giả định chi phí và ràng buộc rồi xem kết quả đổi theo")

k = kho()
if k.tin_hieu.empty:
    st.warning("Chưa có chiến lược nào. Chạy các giai đoạn trong `src/experiments/`.")
    st.stop()

with st.sidebar:
    st.subheader("Giả định backtest")
    # Điểm cơ bản (1 điểm = 0,01%) thay vì phần trăm: Streamlit vẽ nhãn thanh trượt
    # theo định dạng Anh và không cho đổi, thang số nguyên né hẳn vấn đề đó.
    phi = st.slider("Phí mỗi chiều (điểm cơ bản)", 0, 50, round(FEE_RATE * 1e4), 5) / 1e4
    thue = st.slider("Thuế khi bán (điểm cơ bản)", 0, 30, round(SELL_TAX * 1e4), 5) / 1e4
    truot = st.slider("Trượt giá mỗi chiều (điểm cơ bản)", 0, 30,
                      round(SLIPPAGE * 1e4), 1) / 1e4
    tre = st.slider("Trễ thực thi (phiên)", 1, 5, 1)
    lai_tien = st.checkbox("Tiền mặt hưởng lãi phi rủi ro", value=True)
    t_cong = st.checkbox("Áp dụng chu kỳ thanh toán T+2,5", value=True)
    von_trieu = st.select_slider(
        "Vốn giả định", ["100 triệu", "500 triệu", "1 tỷ", "5 tỷ",
                         "10 tỷ", "50 tỷ", "100 tỷ"],
        value="1 tỷ",
    )
    von = {"100 triệu": 0.1, "500 triệu": 0.5, "1 tỷ": 1, "5 tỷ": 5,
           "10 tỷ": 10, "50 tỷ": 50, "100 tỷ": 100}[von_trieu]
    lo_toi_thieu = st.slider("Quy mô lệnh tối thiểu (điểm cơ bản vốn)",
                             0, 500, 50, 50) / 1e4

cfg = BacktestConfig(
    cost=CostModel(fee=phi, sell_tax=thue, slippage=truot),
    constraints=Constraints(capital=von * 1e9, enforce_settlement=t_cong,
                            min_trade=lo_toi_thieu),
    exec_lag=tre,
    cash_earns_rf=lai_tien,
)

c1, c2 = st.columns([2, 1])
with c1:
    chon = st.multiselect(
        "Chiến lược", list(k.tin_hieu.columns),
        default=[c for c in k.tin_hieu.columns if c.startswith("ATFN-ABCD")][:1]
        or list(k.tin_hieu.columns)[:1],
    )
with c2:
    part = chon_phan("Giai đoạn", ["test", "valid", "train", "all"],
                     dang="selectbox", khoa="phan_p9")

if not chon:
    st.info("Chọn ít nhất một chiến lược.")
    st.stop()

nav, dd, r_all, hang = {}, {}, {}, {}
for ten in chon:
    kq = k.chay(ten, part, cfg)
    r = kq.returns.iloc[1:]
    nav[ten] = cumulative_return(r)
    dd[ten] = drawdown_series(r)
    r_all[ten] = r
    nhip = kq.tempo()
    hang[ten] = {
        **summary_table(r, label=ten)[ten].to_dict(),
        "Lệnh mỗi năm": nhip["Số lần khớp lệnh mỗi năm"],
        "Phiên giữa hai lệnh": nhip["Số phiên giữa hai lệnh"],
        "Tỷ lệ thời gian có vị thế": nhip["Tỷ lệ thời gian có vị thế"],
        "Tổng chi phí giao dịch": nhip["Tổng chi phí giao dịch"],
    }

bang = pd.DataFrame(hang).T
if len(chon) == 1 and "Sharpe" in bang.columns:
    muc_tieu_sharpe(float(bang["Sharpe"].iloc[0]))

st.plotly_chart(duong_tai_san(nav, log=False), width="stretch")
st.plotly_chart(drawdown(dd), width="stretch")
st.plotly_chart(sharpe_truot(r_all, 252), width="stretch")

st.subheader("Bảng chỉ tiêu")
bang_dep(bang)

ghi_chu(
    "Mọi con số ở đây được tính lại từ đầu bằng đúng bộ máy backtest dùng cho phần thí "
    "nghiệm, không phải đọc từ bảng đã lưu. Kéo phí lên và xem Sharpe tụt là cách nhanh "
    "nhất để thấy chi phí giao dịch quan trọng tới mức nào trên cổ phiếu này."
)
