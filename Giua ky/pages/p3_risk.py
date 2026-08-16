"""Trang rủi ro và hiệu quả đầu tư: phân phối lợi suất, VaR/CVaR, drawdown, giai đoạn khủng hoảng."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import viz
from src.analytics import metrics as M
from src.config import COLOR_DOWN, COLOR_PRIMARY
from src.formatting import money_vnd as vnd, num, pct
from src.ui import formula, metric_row, note, sidebar

a = sidebar()

st.title("⚠️ Rủi ro và hiệu quả đầu tư")
st.caption("Đo lường rủi ro theo ba góc nhìn: độ biến động, rủi ro đuôi và sụt giảm từ đỉnh")

metric_row([
    ("Độ biến động năm", pct(a.sigma_annual), None),
    ("Sharpe", num(M.sharpe_ratio(a.returns, a.rf_annual)), None),
    ("Sortino", num(M.sortino_ratio(a.returns, a.rf_annual)), None),
    ("VaR 95% một phiên", pct(M.var_historical(a.returns, 0.95)), None),
    ("CVaR 95% một phiên", pct(M.cvar_historical(a.returns, 0.95)), None),
    ("Sụt giảm tối đa", pct(M.max_drawdown(a.returns)), None),
])

tab1, tab2, tab3, tab4 = st.tabs([
    "Phân phối lợi suất & VaR", "Sụt giảm từ đỉnh", "Ổn định theo thời gian",
    "Các giai đoạn thị trường",
])

with tab1:
    st.markdown("### Phân phối lợi suất theo phiên")
    level = st.select_slider("Mức tin cậy", options=[0.90, 0.95, 0.99], value=0.95,
                             format_func=lambda v: f"{v:.0%}")
    var_h = M.var_historical(a.returns, level)
    cvar_h = M.cvar_historical(a.returns, level)
    var_p = M.var_parametric(a.returns, level)

    st.plotly_chart(viz.return_distribution(a.returns, var_h, cvar_h, level),
                    use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        rows = {
            f"VaR {level:.0%} — phương pháp lịch sử": pct(var_h),
            f"VaR {level:.0%} — phương pháp tham số": pct(var_p),
            f"CVaR {level:.0%} — lịch sử": pct(cvar_h),
            "Số phiên vượt ngưỡng VaR": f"{int((a.returns < var_h).sum()):,} / {len(a.returns):,}",
            "Phiên giảm mạnh nhất": pct(float(a.returns.min())),
            "Phiên tăng mạnh nhất": pct(float(a.returns.max())),
            "Độ lệch (Skewness)": num(float(a.returns.skew())),
            "Độ nhọn (Kurtosis)": num(float(a.returns.kurtosis())),
        }
        st.dataframe(pd.DataFrame({"Giá trị": rows}), use_container_width=True)
    with c2:
        formula("VaR<sub>α</sub> = phân vị (1−α) của phân phối lợi suất")
        formula("CVaR<sub>α</sub> = E[ R | R ≤ VaR<sub>α</sub> ]")
        note(
            f"Với danh mục <b>100 triệu đồng</b>: ngưỡng lỗ trong một phiên ở mức tin cậy "
            f"{level:.0%} là <b>{vnd(abs(var_h) * 100e6)}</b>. Nếu rơi vào nhóm "
            f"{(1 - level):.0%} phiên xấu nhất, mức lỗ trung bình là "
            f"<b>{vnd(abs(cvar_h) * 100e6)}</b>."
        )
        kurt = float(a.returns.kurtosis())
        note(
            f"Độ nhọn <b>{kurt:.2f}</b> (phân phối chuẩn bằng 0) cho thấy lợi suất "
            f"{a.ticker} có <b>đuôi dày</b>: biến cố cực đoan xảy ra thường xuyên hơn giả "
            "định chuẩn. Đây là lý do VaR tham số cho con số lạc quan hơn VaR lịch sử, và là "
            "lý do phần mô phỏng có thêm phương án phân phối Student-t."
        )

with tab2:
    dd = M.drawdown_series(a.returns)
    st.plotly_chart(viz.drawdown_chart(dd, a.ticker), use_container_width=True)

    trough = dd.idxmin()
    peak = a.prices.loc[:trough].idxmax()
    after = dd.loc[trough:]
    recovered = after[after >= -1e-9]
    rec_date = recovered.index[0] if len(recovered) else None

    c1, c2 = st.columns([1, 1])
    with c1:
        rows = {
            "Sụt giảm sâu nhất": pct(dd.min()),
            "Đỉnh trước đó": f"{peak:%d/%m/%Y} — {a.prices.loc[peak]:,.2f}",
            "Đáy": f"{trough:%d/%m/%Y} — {a.prices.loc[trough]:,.2f}",
            "Thời gian rơi": f"{(trough - peak).days:,} ngày",
            "Ngày lấy lại đỉnh cũ": f"{rec_date:%d/%m/%Y}" if rec_date is not None else "Chưa hồi phục",
            "Thời gian hồi phục": f"{(rec_date - trough).days:,} ngày" if rec_date is not None else "—",
            "Mức tăng cần để hoà vốn": pct(1 / (1 + dd.min()) - 1),
            "Sụt giảm hiện tại": pct(dd.iloc[-1]),
        }
        st.dataframe(pd.DataFrame({"Giá trị": rows}), use_container_width=True)
    with c2:
        worst = dd.nsmallest(300)
        thresholds = [-0.1, -0.2, -0.3, -0.5]
        rows2 = {f"Số phiên sụt giảm quá {abs(t):.0%}": f"{int((dd < t).sum()):,} phiên "
                 f"({(dd < t).mean() * 100:.1f}%)" for t in thresholds}
        st.dataframe(pd.DataFrame({"Thống kê": rows2}), use_container_width=True)
        note(
            "Toán học của việc hoà vốn rất khắc nghiệt: lỗ 50% cần lãi 100% mới về mức cũ, "
            f"lỗ {abs(dd.min()) * 100:.0f}% cần lãi <b>{pct(1 / (1 + dd.min()) - 1, 0)}</b>. "
            "Đó là lý do quản trị mức sụt giảm quan trọng không kém việc tìm kiếm lợi nhuận."
        )

with tab3:
    st.markdown("### Rủi ro và hiệu quả có ổn định theo thời gian không?")
    max_window = max(63, min(504, len(a.returns) // 2))
    window = st.slider("Cửa sổ trượt (phiên)", 63, max_window, min(252, max_window), 21,
                       help="252 phiên ≈ 1 năm giao dịch. Giới hạn trên được đặt bằng một "
                            "nửa số phiên của giai đoạn đang chọn để ước lượng còn ý nghĩa.")

    roll_vol = a.returns.rolling(window).std(ddof=1) * np.sqrt(252)
    roll_sharpe = M.rolling_metric(a.returns, window, "sharpe", a.rf_annual)

    f = go.Figure()
    f.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol * 100, name="Độ biến động (%)",
                           line=dict(color=COLOR_DOWN)))
    f.add_hline(y=a.sigma_annual * 100, line_dash="dash", line_color="#94a3b8",
                annotation_text=f"Trung bình toàn kỳ {a.sigma_annual * 100:.1f}%")
    f.update_layout(**viz.LAYOUT, title=f"Độ biến động trượt {window} phiên", yaxis_title="%")
    st.plotly_chart(f, use_container_width=True)

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, name="Sharpe trượt",
                            line=dict(color=COLOR_PRIMARY)))
    f2.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    f2.add_hline(y=1, line_dash="dash", line_color="#16a34a", annotation_text="Mức tốt = 1")
    f2.update_layout(**viz.LAYOUT, title=f"Tỷ số Sharpe trượt {window} phiên")
    st.plotly_chart(f2, use_container_width=True)

    note(
        f"Độ biến động dao động từ <b>{pct(float(roll_vol.min()))}</b> đến "
        f"<b>{pct(float(roll_vol.max()))}</b>, còn Sharpe trượt đi từ "
        f"<b>{num(float(roll_sharpe.min()))}</b> tới <b>{num(float(roll_sharpe.max()))}</b>. "
        "Sự dao động lớn này cho thấy các chỉ tiêu tính trên toàn kỳ chỉ là giá trị bình quân; "
        "rủi ro thực tế mà nhà đầu tư gặp phải phụ thuộc mạnh vào thời điểm tham gia thị trường."
    )

with tab4:
    st.markdown("### Hiệu quả qua các giai đoạn thị trường đặc biệt")
    events = [
        ("Khủng hoảng tài chính toàn cầu", "2007-10-01", "2009-02-28"),
        ("Hồi phục hậu khủng hoảng", "2009-03-01", "2009-12-31"),
        ("Giai đoạn tích luỹ", "2012-01-01", "2016-12-31"),
        ("Sóng tăng 2017–2018", "2017-01-01", "2018-04-09"),
        ("Covid-19 lao dốc", "2020-01-20", "2020-03-31"),
        ("Bùng nổ hậu Covid", "2020-04-01", "2021-12-31"),
        ("Siết tín dụng và trái phiếu", "2022-04-01", "2022-11-15"),
        ("Ba năm gần nhất", str(a.prices.index[-1] - pd.Timedelta(days=365 * 3))[:10],
         str(a.prices.index[-1])[:10]),
    ]
    rows = {}
    for label, s, e in events:
        p = a.prices.loc[s:e]
        b = a.benchmark.loc[s:e]
        if len(p) < 2:
            continue
        r_stock = float(p.iloc[-1] / p.iloc[0] - 1)
        r_bench = float(b.iloc[-1] / b.iloc[0] - 1) if len(b) > 1 else np.nan
        r = M.to_returns(p)
        rows[label] = {
            "Giai đoạn": f"{p.index[0]:%m/%Y} – {p.index[-1]:%m/%Y}",
            a.ticker: pct(r_stock),
            "VNINDEX": pct(r_bench),
            "Chênh lệch": pct(r_stock - r_bench),
            "Biến động": pct(M.annual_volatility(r)),
            "Sụt giảm sâu nhất": pct(M.max_drawdown(r)),
        }
    st.dataframe(pd.DataFrame(rows).T, use_container_width=True)

    note(
        "Bảng này cho thấy rủi ro không phân bố đều theo thời gian. Cùng một cổ phiếu, "
        "nhà đầu tư mua năm 2007 và nhà đầu tư mua năm 2012 có trải nghiệm hoàn toàn khác "
        "nhau. Đây là lập luận quan trọng khi biện luận kết quả mô hình: mọi ước lượng "
        "CAPM, APT hay Monte Carlo đều phụ thuộc vào giai đoạn dữ liệu được chọn."
    )

    st.markdown("### Lợi suất theo tháng")
    st.plotly_chart(viz.monthly_heatmap(M.monthly_return_table(a.returns)),
                    use_container_width=True)
