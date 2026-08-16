"""Trang mô hình CAPM: ước lượng beta, alpha Jensen, đường SML và beta trượt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src import viz
from src.analytics.capm import estimate_capm, interpret, rolling_beta
from src.formatting import num, pct
from src.ui import formula, interpretation, metric_row, note, sidebar

a = sidebar()
c = a.capm

st.title("📐 Mô hình định giá tài sản vốn (CAPM)")
st.caption(f"Hồi quy lợi suất vượt trội của {a.ticker} theo lợi suất vượt trội của VNINDEX, "
           f"{c.n_obs:,} quan sát ngày")

formula(
    "R<sub>i</sub> − R<sub>f</sub> = α + β · (R<sub>m</sub> − R<sub>f</sub>) + ε"
)

metric_row([
    ("Beta (β)", num(c.beta), f"t = {c.t_beta:,.1f}"),
    ("Alpha năm (Jensen)", pct(c.alpha_annual), f"p = {c.p_alpha:.3f}"),
    ("R²", pct(c.r_squared, 1), None),
    ("Lợi suất kỳ vọng CAPM", pct(c.expected_return), None),
    ("Lợi suất thực tế (số học)", pct(c.realized_arithmetic),
     f"{(c.realized_arithmetic - c.expected_return) * 100:+.2f} điểm %"),
    ("Treynor", num(c.treynor), None),
])

tab1, tab2, tab3, tab4 = st.tabs([
    "Kết quả hồi quy", "Đường thị trường chứng khoán", "Beta theo thời gian",
    "Kiểm định và hạn chế",
])

with tab1:
    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.plotly_chart(viz.capm_scatter(c, a.ticker), use_container_width=True)
    with col2:
        st.markdown("#### Bảng kết quả ước lượng")
        res = pd.DataFrame({
            "Hệ số": [c.alpha_daily, c.beta],
            "Sai số chuẩn": [float(c.model.bse["const"]), float(c.model.bse["ex_market"])],
            "t-stat": [c.t_alpha, c.t_beta],
            "p-value": [c.p_alpha, c.p_beta],
        }, index=["α (chặn)", "β (hệ số góc)"]).round(4)
        st.dataframe(res, use_container_width=True)
        st.caption(
            "Sai số chuẩn tính theo Newey–West (HAC, 5 độ trễ) để khắc phục hiện tượng "
            "tự tương quan và phương sai thay đổi thường gặp ở chuỗi lợi suất tài chính."
        )
        rows = {
            "Số quan sát": f"{c.n_obs:,}",
            "R²": pct(c.r_squared, 2),
            "Rủi ro phi hệ thống (năm)": pct(c.resid_std_annual),
            "Tỷ trọng rủi ro hệ thống": pct(c.systematic_share, 1),
            "Lợi suất thị trường (năm)": pct(c.market_return_annual),
            "Lãi suất phi rủi ro": pct(c.rf_annual),
        }
        st.dataframe(pd.DataFrame({"Giá trị": rows}), use_container_width=True)

    interpretation(interpret(c, a.ticker))

with tab2:
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.plotly_chart(viz.sml_chart(c, a.ticker), use_container_width=True)
    with col2:
        formula("E(R<sub>i</sub>) = R<sub>f</sub> + β<sub>i</sub> · [E(R<sub>m</sub>) − R<sub>f</sub>]")
        st.markdown(
            f"Thay số cho **{a.ticker}**:\n\n"
            f"- Lãi suất phi rủi ro R<sub>f</sub> = **{pct(c.rf_annual)}**\n"
            f"- Lợi suất thị trường E(R<sub>m</sub>) = **{pct(c.market_return_annual)}**\n"
            f"- Phần bù rủi ro thị trường = **{pct(c.market_return_annual - c.rf_annual)}**\n"
            f"- Beta = **{num(c.beta)}**\n\n"
            f"→ Lợi suất đòi hỏi = **{pct(c.expected_return)}/năm**",
            unsafe_allow_html=True,
        )
        above = c.realized_return > c.expected_return
        note(
            f"{a.ticker} nằm <b>{'phía trên' if above else 'phía dưới'}</b> đường SML: thực tế "
            f"đạt {pct(c.realized_return)} so với mức đòi hỏi {pct(c.expected_return)}. "
            + ("Theo lý thuyết, cổ phiếu nằm trên SML đang bị định giá thấp so với rủi ro hệ "
               "thống mà nhà đầu tư gánh chịu."
               if above else
               "Theo lý thuyết, cổ phiếu nằm dưới SML chưa đền bù đủ cho rủi ro hệ thống.")
        )
        st.caption(
            "Lưu ý: kết luận dựa trên dữ liệu quá khứ và giả định beta ổn định. "
            "Đây là phân tích học thuật, không phải khuyến nghị đầu tư."
        )

with tab3:
    max_window = max(63, min(504, len(a.returns) // 2))
    window = st.slider("Cửa sổ ước lượng beta (phiên)", 63, max_window,
                       min(126, max_window), 21,
                       help="Giới hạn trên bằng một nửa số phiên của giai đoạn đang chọn, "
                            "để chuỗi beta trượt luôn có đủ quan sát.")
    rb = rolling_beta(a.returns, a.benchmark_returns, window)
    if rb.empty:
        st.warning("Giai đoạn được chọn quá ngắn so với cửa sổ ước lượng. "
                   "Hãy giảm cửa sổ hoặc mở rộng giai đoạn phân tích ở thanh bên.")
        st.stop()
    st.plotly_chart(viz.rolling_beta_chart(rb, c.beta, window), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        rows = {
            "Beta toàn kỳ": num(c.beta),
            "Beta hiện tại": num(float(rb.iloc[-1])),
            "Beta cao nhất": f"{rb.max():.2f} ({rb.idxmax():%m/%Y})",
            "Beta thấp nhất": f"{rb.min():.2f} ({rb.idxmin():%m/%Y})",
            "Độ lệch chuẩn của beta": num(float(rb.std())),
            "Tỷ lệ thời gian beta > 1": pct(float((rb > 1).mean()), 1),
        }
        st.dataframe(pd.DataFrame({"Giá trị": rows}), use_container_width=True)
    with col2:
        note(
            f"Beta của {a.ticker} dao động từ <b>{rb.min():.2f}</b> đến <b>{rb.max():.2f}</b> "
            "tuỳ giai đoạn. Việc beta không cố định là hạn chế quan trọng của CAPM tĩnh: "
            "dùng một con số beta duy nhất cho toàn bộ lịch sử sẽ che giấu việc mức độ nhạy "
            "cảm với thị trường của doanh nghiệp thay đổi theo cấu trúc kinh doanh và theo "
            "trạng thái thị trường."
        )
        st.caption("Gợi ý: đổi 'Giai đoạn phân tích' ở thanh bên để so sánh beta giữa các chu kỳ.")

with tab4:
    st.markdown("### So sánh ước lượng qua các giai đoạn con")
    sub_periods = [
        ("Toàn bộ mẫu", None, None),
        ("2007–2009 (khủng hoảng)", "2007-01-01", "2009-12-31"),
        ("2010–2015", "2010-01-01", "2015-12-31"),
        ("2016–2019", "2016-01-01", "2019-12-31"),
        ("2020–2022 (Covid)", "2020-01-01", "2022-12-31"),
        ("2023 đến nay", "2023-01-01", None),
    ]
    rows = {}
    for label, s, e in sub_periods:
        r = a.returns.loc[s:e]
        rm = a.benchmark_returns.loc[s:e]
        if len(r) < 60:
            continue
        try:
            res = estimate_capm(r, rm, a.rf_annual)
        except Exception:
            continue
        rows[label] = {
            "Số phiên": f"{res.n_obs:,}",
            "Beta": num(res.beta),
            "Alpha năm": pct(res.alpha_annual),
            "p-value α": num(res.p_alpha, 3),
            "R²": pct(res.r_squared, 1),
        }
    st.dataframe(pd.DataFrame(rows).T, use_container_width=True)

    note(
        "Beta và alpha thay đổi rõ rệt giữa các giai đoạn — đây là bằng chứng thực nghiệm cho "
        "thấy giả định tham số cố định của CAPM không được thoả mãn với dữ liệu thực. "
        "Khi trình bày kết quả, cần luôn nêu rõ khoảng thời gian ước lượng."
    )

    st.markdown("### Kiểm định phần dư của mô hình")
    resid = pd.Series(c.model.resid)
    from scipy import stats as sps

    jb_stat, jb_p = sps.jarque_bera(resid)[:2]
    lb = None
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    except Exception:
        pass
    rows = {
        "Trung bình phần dư": f"{resid.mean():.2e}",
        "Độ lệch chuẩn phần dư (năm)": pct(c.resid_std_annual),
        "Độ lệch (Skewness)": num(float(resid.skew())),
        "Độ nhọn (Kurtosis)": num(float(resid.kurtosis())),
        "Jarque–Bera": f"{jb_stat:,.0f} (p = {jb_p:.3g})",
    }
    if lb is not None:
        rows["Ljung–Box (10 độ trễ)"] = (
            f"{lb['lb_stat'].iloc[0]:,.1f} (p = {lb['lb_pvalue'].iloc[0]:.3g})"
        )
    st.dataframe(pd.DataFrame({"Giá trị": rows}), use_container_width=True)

    st.markdown("#### Hạn chế của mô hình CAPM trong bài toán này")
    st.markdown(
        f"1. **Phần dư không phân phối chuẩn** — kiểm định Jarque–Bera bác bỏ giả thuyết "
        f"chuẩn (p < 0,001), độ nhọn {float(resid.kurtosis()):.1f} cho thấy đuôi dày. "
        "Sai số chuẩn thông thường sẽ không đáng tin, nên đề tài dùng ước lượng HAC.\n"
        f"2. **Beta không ổn định** — thay đổi từ {rolling_beta(a.returns, a.benchmark_returns, 126).min():.2f} "
        f"đến {rolling_beta(a.returns, a.benchmark_returns, 126).max():.2f} theo thời gian.\n"
        f"3. **R² chỉ {pct(c.r_squared, 0)}** — một nhân tố thị trường không giải thích được "
        "phần lớn biến động của cổ phiếu đơn lẻ.\n"
        "4. **VNINDEX là đại diện chưa hoàn hảo cho danh mục thị trường** — chỉ số này tính "
        "theo vốn hoá và tập trung vào một số mã lớn, chưa bao gồm trái phiếu, bất động sản "
        "hay vốn con người như lý thuyết đòi hỏi.\n"
        "5. **Lãi suất phi rủi ro được giả định cố định**, trong khi thực tế lãi suất Việt Nam "
        "biến động mạnh qua các thời kỳ.\n\n"
        "Những hạn chế này chính là động cơ để chuyển sang mô hình đa nhân tố APT ở trang sau."
    )
