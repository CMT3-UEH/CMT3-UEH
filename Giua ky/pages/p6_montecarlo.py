"""Trang mô phỏng Monte Carlo: dự báo phân phối giá, VaR toàn kỳ, xác suất đạt mục tiêu."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import viz
from src.analytics.montecarlo import compare_methods
from src.formatting import money_vnd as vnd, pct
from src.ui import formula, metric_row, note, sidebar

a = sidebar()

st.title("🎲 Mô phỏng Monte Carlo")
st.caption("Mô tả phân phối các kết cục có thể xảy ra thay vì đưa ra một con số dự báo duy nhất")

# Tham số mô phỏng
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        horizon_label = st.selectbox("Kỳ hạn mô phỏng",
                                     ["3 tháng", "6 tháng", "1 năm", "2 năm", "3 năm", "5 năm"],
                                     index=2)
        horizon = {"3 tháng": 63, "6 tháng": 126, "1 năm": 252,
                   "2 năm": 504, "3 năm": 756, "5 năm": 1260}[horizon_label]
    with c2:
        method_label = st.selectbox("Mô hình sinh dữ liệu",
                                    ["GBM (phân phối chuẩn)", "Student-t (đuôi dày)",
                                     "Bootstrap lịch sử"], index=0)
        method = {"GBM (phân phối chuẩn)": "gbm", "Student-t (đuôi dày)": "t",
                  "Bootstrap lịch sử": "bootstrap"}[method_label]
    with c3:
        n_sims = st.select_slider("Số kịch bản", [1000, 5000, 10000, 20000], value=10000)
    with c4:
        seed = st.number_input("Hạt giống ngẫu nhiên", 0, 9999, 42,
                               help="Cố định hạt giống để kết quả tái lập được — "
                                    "yêu cầu bắt buộc của một nghiên cứu khoa học.")

sim = a.simulate(horizon, int(n_sims), method, int(seed))
stats = sim.stats()

mu_d = float(a.log_returns.mean())
sd_d = float(a.log_returns.std(ddof=1))

metric_row([
    ("Giá hiện tại", f"{a.last_price:,.2f}", None),
    ("Trung vị mô phỏng", f"{stats['Trung vị']:,.2f}",
     f"{(stats['Trung vị'] / a.last_price - 1) * 100:+.1f}%"),
    ("Phân vị 5%", f"{stats['Phân vị 5%']:,.2f}",
     f"{(stats['Phân vị 5%'] / a.last_price - 1) * 100:+.1f}%"),
    ("Phân vị 95%", f"{stats['Phân vị 95%']:,.2f}",
     f"{(stats['Phân vị 95%'] / a.last_price - 1) * 100:+.1f}%"),
    ("Xác suất thua lỗ", pct(stats["Xác suất lỗ"], 1), None),
    (f"VaR 95% ({horizon_label})", pct(stats["VaR 95% (toàn kỳ)"]), None),
])

tab1, tab2, tab3, tab4 = st.tabs([
    "Quỹ đạo mô phỏng", "Phân phối kết quả", "So sánh ba phương pháp", "Xác suất đạt mục tiêu",
])

with tab1:
    st.plotly_chart(viz.mc_fan(sim, a.ticker, sample=80), use_container_width=True)
    col1, col2 = st.columns([1, 1.2])
    with col1:
        formula({
            "gbm": "S<sub>t+1</sub> = S<sub>t</sub> · exp[(μ − σ²/2)Δt + σ√Δt · ε],  "
                   "ε ~ N(0, 1)",
            "t": "S<sub>t+1</sub> = S<sub>t</sub> · exp[(μ − σ²/2)Δt + σ√Δt · ε],  "
                 "ε ~ Student-t(4) đã chuẩn hoá",
            "bootstrap": "r<sub>t</sub> lấy mẫu có hoàn lại theo khối 5 phiên từ lợi suất "
                         "lịch sử; S<sub>t+1</sub> = S<sub>t</sub> · exp(r<sub>t</sub>)",
        }[method])
        rows = {
            "Lợi suất log trung bình (ngày)": f"{mu_d:.5f}",
            "Độ lệch chuẩn (ngày)": f"{sd_d:.5f}",
            "Lợi suất kỳ vọng (năm)": pct(mu_d * 252),
            "Độ biến động (năm)": pct(sd_d * np.sqrt(252)),
            "Số phiên lịch sử dùng ước lượng": f"{len(a.log_returns):,}",
            "Số kịch bản mô phỏng": f"{int(n_sims):,}",
            "Số bước mỗi kịch bản": f"{horizon:,} phiên",
        }
        st.dataframe(pd.DataFrame({"Giá trị": rows}), use_container_width=True)
    with col2:
        note(
            f"Biểu đồ vẽ 80 quỹ đạo mẫu trong tổng số {int(n_sims):,} kịch bản. Ba đường đậm "
            "là phân vị 5%, trung vị và phân vị 95% tính trên toàn bộ kịch bản.<br><br>"
            "Điều đáng chú ý là <b>hình quạt mở rộng theo căn bậc hai của thời gian</b>: "
            "độ bất định tăng dần nhưng chậm hơn tuyến tính. Đây là nền tảng toán học cho "
            "nhận định rằng đầu tư dài hạn giảm rủi ro tương đối."
        )
        note(
            f"Khoảng tin cậy 90% cho giá {a.ticker} sau {horizon_label}: "
            f"<b>{stats['Phân vị 5%']:,.1f} – {stats['Phân vị 95%']:,.1f}</b> nghìn đồng. "
            "Khoảng rất rộng này chính là thông điệp quan trọng nhất của mô phỏng: với độ "
            f"biến động {pct(a.sigma_annual)}/năm, dự báo điểm cho giá cổ phiếu là vô nghĩa."
        )

with tab2:
    st.plotly_chart(viz.mc_histogram(sim, a.ticker), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        show = {k: (f"{v:,.2f}" if "Giá" in k or "vị" in k else pct(v))
                for k, v in stats.items()}
        st.dataframe(pd.DataFrame({"Giá trị": show}), use_container_width=True)
    with col2:
        var, cvar = stats["VaR 95% (toàn kỳ)"], stats["CVaR 95% (toàn kỳ)"]
        capital = st.number_input("Vốn đầu tư (triệu đồng)", 10, 10000, 100, 10)
        cap = capital * 1e6
        rows = {
            "Giá trị kỳ vọng cuối kỳ": vnd(cap * (1 + stats["Lợi suất kỳ vọng"])),
            "Trung vị": vnd(cap * (stats["Trung vị"] / a.last_price)),
            "Ngưỡng lỗ 5% xấu nhất (VaR)": vnd(cap * abs(var)),
            "Lỗ trung bình khi xấu (CVaR)": vnd(cap * abs(cvar)),
            "Kịch bản tệ nhất": vnd(cap * abs(stats["Lỗ tệ nhất"])),
            "Xác suất mất tiền": pct(stats["Xác suất lỗ"], 1),
        }
        st.dataframe(pd.DataFrame({"Giá trị": rows}), use_container_width=True)
        note(
            "Câu hỏi quan trọng dành cho nhà đầu tư không phải là <i>kỳ vọng lãi bao nhiêu</i>, "
            f"mà là <b>có chịu nổi khoản lỗ {vnd(cap * abs(cvar))} hay không</b> mà vẫn giữ "
            "được kỷ luật, không bán tháo ở đáy."
        )

with tab3:
    cmp = compare_methods(a.last_price, a.log_returns, horizon, min(int(n_sims), 10000), int(seed))
    show = cmp.copy().astype(object)
    for i in cmp.index:
        for c in cmp.columns:
            v = cmp.loc[i, c]
            show.loc[i, c] = pct(v) if any(k in i for k in
                ("Lợi suất", "Xác suất", "VaR", "CVaR", "Lỗ", "Lãi")) else f"{v:,.2f}"
    st.dataframe(show, use_container_width=True)

    f = go.Figure()
    for col, color in zip(cmp.columns, ["#2563eb", "#f59e0b", "#16a34a"]):
        f.add_trace(go.Bar(name=col, x=["VaR 95%", "CVaR 95%", "Xác suất lỗ"],
                           y=[cmp.loc["VaR 95% (toàn kỳ)", col] * 100,
                              cmp.loc["CVaR 95% (toàn kỳ)", col] * 100,
                              -cmp.loc["Xác suất lỗ", col] * 100],
                           marker_color=color))
    f.update_layout(**viz.LAYOUT, title="Chỉ tiêu rủi ro theo từng phương pháp (%)",
                    barmode="group")
    st.plotly_chart(f, use_container_width=True)

    st.markdown(
        "#### Ba giả định khác nhau về phân phối\n\n"
        "| Phương pháp | Giả định | Ưu điểm | Nhược điểm |\n"
        "|---|---|---|---|\n"
        "| **GBM** | Lợi suất log phân phối chuẩn | Chuẩn mực lý thuyết, có lời giải giải "
        "tích, dễ giải thích | Đánh giá thấp xác suất biến cố cực đoan |\n"
        "| **Student-t** | Cú sốc theo phân phối t, bậc tự do 4 | Phản ánh đuôi dày của thị "
        "trường thực | Phải chọn bậc tự do, vẫn giả định độc lập |\n"
        "| **Bootstrap** | Lấy mẫu theo khối 5 phiên từ lịch sử | Không giả định phân phối, "
        "giữ được hiệu ứng cụm biến động | Chỉ tái hiện những gì đã xảy ra trong quá khứ |\n"
    )
    var_row = cmp.loc["VaR 95% (toàn kỳ)"]
    worst_row = cmp.loc["Lỗ tệ nhất"]
    note(
        f"Ở mức tin cậy 95% — vùng đuôi gần — ba phương pháp cho kết quả rất sát nhau "
        f"({', '.join(pct(v) for v in var_row)}). Khác biệt chỉ lộ rõ ở vùng đuôi xa: kịch bản "
        f"tệ nhất lần lượt là {', '.join(pct(v) for v in worst_row)}.<br><br>"
        f"Nguyên nhân là biên độ dao động của sàn HOSE giới hạn các phiên cực đoan, khiến độ "
        f"nhọn của {a.ticker} chỉ <b>{float(a.returns.kurtosis()):.2f}</b> — thấp hơn nhiều so "
        "với cổ phiếu quốc tế. Vì vậy phần đuôi gần của phân phối thực tế không dày hơn phân "
        "phối chuẩn, và ưu thế của mô hình đuôi dày chỉ thể hiện ở kịch bản khủng hoảng."
    )

with tab4:
    st.markdown("### Xác suất đạt các mức giá mục tiêu")
    col1, col2 = st.columns([1, 1])
    with col1:
        target = st.slider("Giá mục tiêu (nghìn đồng)",
                           float(a.last_price * 0.3), float(a.last_price * 3.0),
                           float(a.last_price * 1.2), 0.5)
        p = sim.prob_above(target)
        st.metric(f"Xác suất giá ≥ {target:,.1f} sau {horizon_label}", pct(p, 1))
        st.caption(f"Tương ứng mức {'tăng' if target > a.last_price else 'giảm'} "
                   f"{abs(target / a.last_price - 1) * 100:.1f}% so với giá hiện tại.")
    with col2:
        levels = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        rows = {
            f"≥ {m:.0%} giá hiện tại ({a.last_price * m:,.1f})": pct(sim.prob_above(a.last_price * m), 1)
            for m in levels
        }
        st.dataframe(pd.DataFrame({"Xác suất": rows}), use_container_width=True)

    st.markdown("### Xác suất thua lỗ theo thời gian nắm giữ")
    rows = {}
    for label, h in [("3 tháng", 63), ("6 tháng", 126), ("1 năm", 252),
                     ("2 năm", 504), ("3 năm", 756), ("5 năm", 1260)]:
        s = a.simulate(h, 5000, method, int(seed))
        st_ = s.stats()
        rows[label] = {
            "Xác suất lỗ": st_["Xác suất lỗ"] * 100,
            "Lợi suất kỳ vọng": st_["Lợi suất kỳ vọng"] * 100,
            "VaR 95%": st_["VaR 95% (toàn kỳ)"] * 100,
        }
    df = pd.DataFrame(rows).T

    f = go.Figure()
    f.add_trace(go.Bar(x=df.index, y=df["Xác suất lỗ"], name="Xác suất lỗ (%)",
                       marker_color="#dc2626"))
    f.add_trace(go.Scatter(x=df.index, y=df["Lợi suất kỳ vọng"], name="Lợi suất kỳ vọng (%)",
                           mode="lines+markers", yaxis="y2", line=dict(color="#16a34a", width=3)))
    f.update_layout(**viz.LAYOUT, title="Thời gian nắm giữ càng dài, xác suất lỗ càng giảm",
                    yaxis=dict(title="Xác suất lỗ (%)"),
                    yaxis2=dict(title="Lợi suất kỳ vọng (%)", overlaying="y", side="right"))
    st.plotly_chart(f, use_container_width=True)

    note(
        f"Xác suất thua lỗ giảm từ <b>{df.loc['3 tháng', 'Xác suất lỗ']:.1f}%</b> "
        f"(3 tháng) xuống <b>{df.loc['5 năm', 'Xác suất lỗ']:.1f}%</b> (5 năm). "
        "Nguyên nhân toán học: lợi suất kỳ vọng tích luỹ tăng tuyến tính theo thời gian "
        "(μ·T) trong khi độ lệch chuẩn chỉ tăng theo căn bậc hai (σ·√T), nên tỷ lệ "
        "tín hiệu trên nhiễu cải thiện dần. Đây là lập luận định lượng cho chiến lược "
        "đầu tư dài hạn — với điều kiện lợi suất kỳ vọng thực sự dương."
    )
