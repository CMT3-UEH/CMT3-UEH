"""Trang tổng quan: bức tranh nhanh về cổ phiếu và toàn bộ kết quả phân tích."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import viz
from src.analytics import metrics as M
from src.config import COLOR_DOWN, COLOR_UP
from src.formatting import num, pct
from src.ui import metric_row, note, sidebar

a = sidebar()

st.title(f"🏠 Tổng quan — {a.ticker}")
st.caption(
    f"Dữ liệu từ {a.prices.index[0]:%d/%m/%Y} đến {a.last_date:%d/%m/%Y} "
    f"({len(a.prices):,} phiên giao dịch) · Giá đã điều chỉnh cổ tức và chia tách"
)

# Thẻ số liệu chính
metric_row([
    ("Giá đóng cửa", f"{a.last_price:,.2f}", f"{a.returns.iloc[-1] * 100:+.2f}% phiên"),
    ("Lợi suất năm (CAGR)", pct(a.mu_annual), f"VNINDEX {pct(M.annual_return(a.benchmark_returns))}"),
    ("Độ biến động năm", pct(a.sigma_annual), f"VNINDEX {pct(M.annual_volatility(a.benchmark_returns))}"),
    ("Tỷ số Sharpe", num(M.sharpe_ratio(a.returns, a.rf_annual)), None),
    ("Beta (CAPM)", num(a.capm.beta), f"R² {pct(a.capm.r_squared, 0)}"),
    ("Sụt giảm tối đa", pct(M.max_drawdown(a.returns)), None),
])

note(
    f"Trong giai đoạn khảo sát, <b>{a.ticker}</b> mang lại <b>{pct(a.mu_annual)}/năm</b> "
    f"với độ biến động <b>{pct(a.sigma_annual)}</b>. Beta <b>{num(a.capm.beta)}</b> cho thấy "
    f"cổ phiếu {'ít nhạy hơn' if a.capm.beta < 1 else 'nhạy hơn'} thị trường, và alpha Jensen "
    f"<b>{pct(a.capm.alpha_annual)}/năm</b> "
    f"({'có' if a.capm.p_alpha < 0.05 else 'chưa có'} ý nghĩa thống kê ở mức 5%)."
)

# Biểu đồ giá
st.markdown("## Diễn biến giá và khối lượng")
c1, c2 = st.columns([3, 1])
with c2:
    show_bb = st.checkbox("Hiện dải Bollinger", value=True)
    show_vol = st.checkbox("Hiện khối lượng", value=True)
    log_scale = st.checkbox("Trục giá theo thang log", value=True,
                            help="Với chuỗi giá dài nhiều năm, thang log giúp so sánh "
                                 "mức tăng trưởng theo tỷ lệ phần trăm chứ không theo "
                                 "số tuyệt đối.")
fig = viz.candlestick(a.tech, a.ticker, show_bb=show_bb)
if log_scale:
    fig.update_yaxes(type="log")
with c1:
    st.plotly_chart(fig, use_container_width=True)
if show_vol:
    st.plotly_chart(viz.volume_chart(a.tech), use_container_width=True)

# So sánh với thị trường
st.markdown("## So sánh với thị trường")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(
        viz.growth_comparison(a.returns, a.benchmark_returns, a.ticker),
        use_container_width=True,
    )
with col2:
    yearly = a.yearly_returns() * 100
    yearly_b = ((1 + a.benchmark_returns).resample("YE").prod() - 1) * 100
    f = go.Figure()
    f.add_trace(go.Bar(x=yearly.index.year, y=yearly.values, name=a.ticker,
                       marker_color=[COLOR_UP if v >= 0 else COLOR_DOWN for v in yearly.values]))
    f.add_trace(go.Scatter(x=yearly_b.index.year, y=yearly_b.values, name="VNINDEX",
                           mode="markers", marker=dict(size=9, color="#334155", symbol="diamond")))
    f.update_layout(**viz.LAYOUT, title="Lợi suất từng năm (%)", yaxis_title="%")
    st.plotly_chart(f, use_container_width=True)

growth = 1 + M.total_return(a.returns)
years = (a.prices.index[-1] - a.prices.index[0]).days / 365.25
note(
    f"100 triệu đồng đầu tư vào {a.ticker} đầu kỳ và giữ tới nay trở thành "
    f"<b>{growth * 100:,.0f} triệu đồng</b> — gấp <b>{growth:,.1f} lần</b> sau "
    f"{years:.1f} năm. Cùng khoảng thời gian, gửi tiết kiệm ở mức "
    f"{pct(a.rf_annual, 1)}/năm chỉ cho <b>{(1 + a.rf_annual) ** years * 100:,.0f} triệu</b>."
)

# Bảng chỉ tiêu
st.markdown("## Bảng chỉ tiêu tổng hợp")
tbl = pd.concat([a.summary, a.benchmark_summary], axis=1)
show = tbl.copy().astype(object)
for idx in tbl.index:
    is_pct = any(k in idx for k in
                 ("Lợi suất", "biến động", "giảm", "Tỷ lệ", "VaR", "CVaR"))
    for col in tbl.columns:
        v = tbl.loc[idx, col]
        show.loc[idx, col] = pct(v) if is_pct else num(v)
st.dataframe(show, use_container_width=True)

with st.expander("Các chỉ tiêu này có ý nghĩa gì?"):
    st.markdown(
        "- **Lợi suất tích luỹ** — tổng mức sinh lời của cả giai đoạn.\n"
        "- **CAGR** — lợi suất kép bình quân mỗi năm, đã tính hiệu ứng lãi nhập gốc.\n"
        "- **Độ biến động** — độ lệch chuẩn lợi suất năm hoá, thước đo rủi ro tổng thể.\n"
        "- **Sharpe** — lợi suất vượt trội trên mỗi đơn vị rủi ro; càng cao càng hiệu quả.\n"
        "- **Sortino** — như Sharpe nhưng chỉ phạt biến động giảm.\n"
        "- **Sụt giảm tối đa** — khoản lỗ lớn nhất tính từ đỉnh tới đáy.\n"
        "- **Calmar** — lợi suất năm chia cho mức sụt giảm tối đa.\n"
        "- **VaR/CVaR 95%** — ngưỡng lỗ và mức lỗ trung bình trong 5% phiên xấu nhất.\n"
        "- **Skewness/Kurtosis** — độ lệch và độ nhọn của phân phối lợi suất; kurtosis "
        "dương lớn nghĩa là đuôi dày, biến cố cực đoan xảy ra nhiều hơn phân phối chuẩn."
    )

st.markdown("## Nội dung các trang phân tích")
c = st.columns(4)
cards = [
    ("🏢 Thông tin cổ phiếu", "Hồ sơ doanh nghiệp, cơ cấu cổ đông, P/E, P/B, ROE và "
                              "các chỉ báo kỹ thuật."),
    ("⚠️ Rủi ro và hiệu quả", "Phân phối lợi suất, VaR/CVaR, sụt giảm từ đỉnh, hiệu "
                               "quả theo từng giai đoạn khủng hoảng."),
    ("📐 Mô hình CAPM", "Hồi quy beta và alpha Jensen, đường SML, beta trượt theo "
                        "thời gian."),
    ("🧩 Mô hình APT", "Mô hình đa nhân tố vĩ mô: thị trường, tỷ giá, dầu, vàng, "
                       "chứng khoán Mỹ."),
    ("🎲 Monte Carlo", "Mô phỏng 10.000 kịch bản giá bằng ba phương pháp, tính VaR "
                       "toàn kỳ và xác suất thua lỗ."),
    ("💼 Quản lý đầu tư", "Đường phân bổ vốn, tiêu chí Kelly, định cỡ vị thế, so sánh "
                          "DCA với mua một lần."),
    ("🤖 Chatbot hỏi đáp", "52 câu hỏi gợi ý được trả lời bằng số liệu tính trực tiếp "
                           "từ dữ liệu."),
    ("📄 Báo cáo", "Toàn bộ hình và bảng trong dashboard được xuất sang báo cáo Word "
                   "kèm slide thuyết trình."),
]
for i, (title, desc) in enumerate(cards):
    with c[i % 4]:
        st.markdown(f"**{title}**")
        st.caption(desc)
