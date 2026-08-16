"""Trang thông tin cổ phiếu: hồ sơ doanh nghiệp, định giá, cổ đông, chỉ báo kỹ thuật."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import viz
from src.analytics import technical as TA
from src.formatting import money_vnd as vnd, num, pct
from src.ui import metric_row, note, sidebar

a = sidebar()

st.title(f"🏢 Thông tin cổ phiếu {a.ticker}")


def ov(field, default=None):
    if a.overview is None or a.overview.empty or field not in a.overview.columns:
        return default
    v = a.overview.iloc[0][field]
    return default if pd.isna(v) else v


# Hồ sơ doanh nghiệp
name = ov("organ_name", a.ticker)
st.markdown(f"### {name}")
st.caption(f"Mã chứng khoán **{a.ticker}** · Ngành **{ov('sector', '—')}** · "
           f"Nhóm chỉ số **{ov('com_group_code', '—')}**")

metric_row([
    ("Giá hiện tại", f"{a.last_price:,.2f}", f"{a.returns.iloc[-1] * 100:+.2f}%"),
    ("Vốn hoá", vnd(ov("market_cap")), None),
    ("Số cổ phiếu lưu hành", f"{ov('issue_share', 0) / 1e6:,.0f} triệu", None),
    ("Sở hữu nước ngoài", pct(ov("foreigner_percentage"), 1),
     f"trần {pct(ov('maximum_foreign_percentage'), 0)}"),
    ("Cao nhất 52 tuần", f"{ov('highest_price1_year', 0) / 1000:,.1f}", None),
    ("Thấp nhất 52 tuần", f"{ov('lowest_price1_year', 0) / 1000:,.1f}", None),
])

profile = str(ov("company_profile", "") or "")
profile = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", profile)).strip()
if profile:
    with st.expander("Giới thiệu doanh nghiệp", expanded=True):
        st.write(profile)

# Chỉ số tài chính
st.markdown("## Chỉ số tài chính và định giá")

if a.ratio is not None and not a.ratio.empty:
    ratio = a.ratio.copy().sort_values(["year", "quarter"])
    latest = ratio.iloc[-1]
    period = f"{int(latest['year'])} · quý {int(latest['quarter'])}"

    metric_row([
        ("P/E", num(latest.get("pe")), None),
        ("P/B", num(latest.get("pb")), None),
        ("ROE", pct(latest.get("roe"), 1), None),
        ("ROA", pct(latest.get("roa"), 1), None),
        ("Biên lợi nhuận ròng", pct(latest.get("after_tax_profit_margin"), 1), None),
        ("Nợ/Vốn chủ sở hữu", num(latest.get("debt_to_equity")), None),
    ])
    st.caption(f"Số liệu kỳ gần nhất: {period}")

    tab1, tab2, tab3 = st.tabs(["Định giá theo thời gian", "Khả năng sinh lời", "Bảng số liệu"])

    # Nguồn dữ liệu dùng quý 5 để ký hiệu số liệu cả năm
    ratio["ky"] = [
        f"{int(y)} — cả năm" if int(q) >= 5 else f"{int(y)}Q{int(q)}"
        for y, q in zip(ratio["year"], ratio["quarter"])
    ]

    with tab1:
        f = go.Figure()
        for col, label in (("pe", "P/E"), ("pb", "P/B")):
            if col in ratio:
                f.add_trace(go.Scatter(x=ratio["ky"], y=ratio[col], name=label, mode="lines+markers"))
        f.update_layout(**viz.LAYOUT, title="Hệ số định giá qua các kỳ", yaxis_title="Lần")
        st.plotly_chart(f, use_container_width=True)
        note(
            "<b>P/E</b> cho biết nhà đầu tư trả bao nhiêu đồng cho mỗi đồng lợi nhuận một năm "
            "của doanh nghiệp; <b>P/B</b> so giá thị trường với giá trị sổ sách. Hệ số cao "
            "phản ánh kỳ vọng tăng trưởng, đồng thời cũng là mức rủi ro nếu kỳ vọng đó không "
            "thành hiện thực."
        )

    with tab2:
        f = go.Figure()
        for col, label in (("roe", "ROE"), ("roa", "ROA"),
                           ("after_tax_profit_margin", "Biên lợi nhuận ròng"),
                           ("gross_margin", "Biên lợi nhuận gộp")):
            if col in ratio:
                f.add_trace(go.Scatter(x=ratio["ky"], y=ratio[col] * 100, name=label,
                                       mode="lines+markers"))
        f.update_layout(**viz.LAYOUT, title="Hiệu quả sinh lời qua các kỳ (%)", yaxis_title="%")
        st.plotly_chart(f, use_container_width=True)
        note(
            "<b>ROE</b> đo lợi nhuận tạo ra trên mỗi đồng vốn chủ sở hữu — chỉ tiêu cốt lõi "
            "đánh giá chất lượng doanh nghiệp. ROE duy trì cao và ổn định qua nhiều chu kỳ "
            "thường đi kèm mức định giá P/B cao."
        )

    with tab3:
        cols = [c for c in ("ky", "pe", "pb", "ps", "ev_to_ebitda", "roe", "roa",
                            "gross_margin", "after_tax_profit_margin", "debt_to_equity",
                            "current_ratio", "dividend_yield") if c in ratio.columns]
        show = ratio[cols].rename(columns={
            "ky": "Kỳ", "pe": "P/E", "pb": "P/B", "ps": "P/S", "ev_to_ebitda": "EV/EBITDA",
            "roe": "ROE", "roa": "ROA", "gross_margin": "Biên gộp",
            "after_tax_profit_margin": "Biên ròng", "debt_to_equity": "Nợ/VCSH",
            "current_ratio": "Thanh toán hiện hành", "dividend_yield": "Tỷ suất cổ tức",
        }).set_index("Kỳ").round(3)
        st.dataframe(show.iloc[::-1], use_container_width=True, height=380)
else:
    st.info("Chưa lấy được dữ liệu chỉ số tài chính từ nguồn.")

# Cổ đông
st.markdown("## Cơ cấu cổ đông")
if a.shareholders is not None and not a.shareholders.empty:
    # Tỷ lệ sở hữu được tính lại từ số cổ phiếu chia cho tổng số cổ phiếu lưu
    # hành, vì tỷ lệ do nguồn dữ liệu cung cấp không nhất quán với số lượng.
    sh = a.shareholders.copy()
    issued = ov("issue_share")
    if issued:
        sh["share_own_percent"] = sh["quantity"] / float(issued)
    sh = sh.sort_values("quantity", ascending=False).head(10)
    col1, col2 = st.columns([1.2, 1])
    with col1:
        show = pd.DataFrame({
            "Cổ đông": sh["share_holder"].values,
            "Số cổ phiếu": [f"{v:,.0f}" for v in sh["quantity"]],
            "Tỷ lệ": [pct(v, 2) for v in sh["share_own_percent"]],
        })
        st.dataframe(show, use_container_width=True, hide_index=True)
    with col2:
        others = max(0.0, 1 - float(sh["share_own_percent"].sum()))
        f = go.Figure(go.Pie(
            labels=list(sh["share_holder"].str[:28]) + ["Cổ đông khác"],
            values=list(sh["share_own_percent"]) + [others],
            hole=0.45, textinfo="percent",
        ))
        f.update_layout(**viz.LAYOUT, title="Tỷ lệ sở hữu", showlegend=False)
        st.plotly_chart(f, use_container_width=True)

    room = ov("maximum_foreign_percentage", 0) - ov("foreigner_percentage", 0)
    note(
        f"Mười cổ đông lớn nhất nắm <b>{pct(sh['share_own_percent'].sum())}</b> vốn điều lệ. "
        f"Room ngoại còn lại khoảng <b>{pct(room)}</b> — "
        + ("gần cạn, dòng vốn ngoại khó gia tăng sở hữu."
           if room < 0.03 else "vẫn còn dư địa cho nhà đầu tư nước ngoài.")
    )
else:
    st.info("Chưa lấy được dữ liệu cổ đông.")

# Phân tích kỹ thuật
st.markdown("## Chỉ báo kỹ thuật")
sig = TA.signal_summary(a.tech)
cols = st.columns(len(sig))
for col, (k, v) in zip(cols, sig.items()):
    with col:
        st.markdown(f"**{k}**")
        st.caption(v)

f1, f2 = viz.rsi_macd(a.tech.tail(500))
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(f1, use_container_width=True)
with c2:
    st.plotly_chart(f2, use_container_width=True)

note(
    "Các chỉ báo kỹ thuật mô tả hành vi giá gần đây chứ không dự báo tương lai. "
    "Trong đề tài này chúng đóng vai trò bổ trợ: cung cấp bối cảnh thị trường khi đọc "
    "kết quả của các mô hình định lượng CAPM, APT và Monte Carlo."
)
