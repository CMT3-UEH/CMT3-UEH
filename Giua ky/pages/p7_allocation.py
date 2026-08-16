"""Trang quản lý đầu tư: đường phân bổ vốn, Kelly, định cỡ vị thế, DCA và backtest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import viz
from src.analytics import allocation as AL
from src.formatting import money_vnd as vnd, num, pct
from src.ui import formula, note, sidebar

a = sidebar()

st.title("💼 Quản lý đầu tư")
st.caption(
    "Từ kết quả phân tích rủi ro — lợi nhuận đến quyết định thực tế: phân bổ bao nhiêu vốn, "
    "mua bao nhiêu cổ phiếu, giải ngân theo cách nào"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Đường phân bổ vốn", "Định cỡ vị thế", "Backtest phân bổ", "DCA và thời gian nắm giữ",
])

with tab1:
    st.markdown("### Phân bổ giữa cổ phiếu và tài sản phi rủi ro")
    formula("y* = [E(R) − R<sub>f</sub>] / (A · σ²)")

    col1, col2 = st.columns([1, 1.4])
    with col1:
        A = st.slider("Mức ngại rủi ro A", 1.0, 10.0, 3.0, 0.5,
                      help="A càng lớn càng ngại rủi ro. Nhà đầu tư cá nhân điển hình "
                           "thường được giả định A trong khoảng 2–4.")
        leverage = st.checkbox("Cho phép vay ký quỹ (y > 100%)", value=False)
        opt = AL.optimal_allocation(a.mu_arithmetic, a.sigma_annual, A, a.rf_annual, leverage)

        st.metric("Tỷ trọng tối ưu vào cổ phiếu", pct(opt.y_optimal, 1))
        rows = {
            "Lợi suất kỳ vọng — trung bình số học": pct(a.mu_arithmetic),
            "Lợi suất kép thực tế (CAGR)": pct(a.mu_annual),
            "Độ biến động của cổ phiếu": pct(a.sigma_annual),
            "Lãi suất phi rủi ro": pct(a.rf_annual),
            "Phần bù rủi ro": pct(a.mu_arithmetic - a.rf_annual),
            "— Danh mục hoàn chỉnh —": "",
            "Tỷ trọng cổ phiếu (y*)": pct(opt.y_optimal, 1),
            "Tỷ trọng tiền gửi": pct(1 - opt.y_optimal, 1),
            "Lợi suất kỳ vọng": pct(opt.expected_return),
            "Độ biến động": pct(opt.volatility),
            "Sharpe": num(opt.sharpe),
            "Mức hữu dụng U": num(opt.utility, 4),
        }
        st.dataframe(pd.DataFrame({"Giá trị": rows}), use_container_width=True)
    with col2:
        cal = AL.capital_allocation_line(a.mu_arithmetic, a.sigma_annual, a.rf_annual,
                                         1.5 if leverage else 1.0)
        st.plotly_chart(
            viz.cal_chart(cal, opt, a.mu_arithmetic, a.sigma_annual, a.rf_annual, a.ticker),
            use_container_width=True,
        )

    st.markdown("#### Tỷ trọng tối ưu theo từng mức ngại rủi ro")
    rows = {}
    for A_ in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        o = AL.optimal_allocation(a.mu_arithmetic, a.sigma_annual, A_, a.rf_annual, leverage)
        rows[f"A = {A_:g}"] = {
            "Tỷ trọng cổ phiếu": pct(o.y_optimal, 1),
            "Tỷ trọng tiền gửi": pct(1 - o.y_optimal, 1),
            "Lợi suất kỳ vọng": pct(o.expected_return),
            "Độ biến động": pct(o.volatility),
            "Hữu dụng U": num(o.utility, 4),
        }
    st.dataframe(pd.DataFrame(rows).T, use_container_width=True)

    kelly = AL.kelly_fraction(a.mu_arithmetic - a.rf_annual, a.sigma_annual)
    note(
        f"Với mức ngại rủi ro A = {A:g} đang chọn, lý thuyết khuyến nghị đặt "
        f"<b>{pct(opt.y_optimal, 0)}</b> vốn vào {a.ticker}. Tiêu chí Kelly cho con số "
        f"<b>{pct(kelly, 0)}</b> — tương đương trường hợp A = 1, tức gần như không ngại rủi ro, "
        f"nên thực hành phổ biến là dùng nửa Kelly ≈ <b>{pct(kelly / 2, 0)}</b>.<br><br>"
        "<b>Hai lưu ý quan trọng khi đọc con số này.</b> Thứ nhất, lợi suất kỳ vọng dùng ở đây "
        f"là <b>trung bình số học</b> ({pct(a.mu_arithmetic)}/năm) chứ không phải lợi suất kép "
        f"({pct(a.mu_annual)}/năm) — công thức hữu dụng trung bình – phương sai đòi hỏi như vậy. "
        "Thứ hai, tham số μ và σ được ước lượng trên <b>toàn bộ mẫu lịch sử</b>, tức bao gồm cả "
        "thông tin mà nhà đầu tư trong quá khứ chưa thể biết; đây là minh hoạ lý thuyết, không "
        "phải khuyến nghị có thể thực hiện được tại thời điểm trong quá khứ.<br><br>"
        "Ngoài ra đây là bài toán chỉ có một tài sản rủi ro. Kết quả không nói rằng nên dồn "
        f"{pct(opt.y_optimal, 0)} tài sản vào một cổ phiếu, mà nói rằng <i>nếu</i> chỉ được chọn "
        "giữa cổ phiếu này và tiền gửi thì tỷ lệ đó tối ưu hoá hữu dụng. Đa dạng hoá sang nhiều "
        f"mã sẽ loại bỏ phần rủi ro riêng ({pct(1 - a.capm.systematic_share, 0)} tổng rủi ro) mà "
        "không phải hy sinh lợi suất kỳ vọng."
    )

with tab2:
    st.markdown("### Mua bao nhiêu cổ phiếu là hợp lý?")
    col1, col2 = st.columns([1, 1])
    with col1:
        capital_m = st.number_input("Vốn đầu tư (triệu đồng)", 10, 100000, 100, 10)
        risk_pct = st.slider("Rủi ro tối đa mỗi lệnh (% vốn)", 0.5, 10.0, 2.0, 0.5) / 100
        atr_mult = st.slider("Khoảng cắt lỗ (số lần ATR)", 1.0, 5.0, 2.0, 0.5)

        capital = capital_m * 1e6
        price_vnd = a.last_price * 1000
        atr = float(a.tech["ATR14"].iloc[-1]) * 1000
        stop_dist = atr_mult * atr
        ps = AL.position_size_by_risk(capital, price_vnd, stop_dist, risk_pct)

        st.metric("Số cổ phiếu nên mua", f"{ps['shares']:,}")
        rows = {
            "Giá hiện tại": f"{price_vnd:,.0f} đồng",
            "ATR(14)": f"{atr:,.0f} đồng ({atr / price_vnd * 100:.2f}% giá)",
            "Khoảng cắt lỗ": f"{stop_dist:,.0f} đồng ({stop_dist / price_vnd * 100:.2f}%)",
            "Giá cắt lỗ": f"{ps['stop_price']:,.0f} đồng",
            "Số tiền rủi ro tối đa": vnd(ps["risk_amount"]),
            "Giá trị vị thế": vnd(ps["value"]),
            "Tỷ trọng trên tổng vốn": pct(ps["weight"], 1),
        }
        st.dataframe(pd.DataFrame({"Giá trị": rows}), use_container_width=True)
    with col2:
        st.markdown("#### So sánh hai nguyên tắc định cỡ")
        opt3 = a.allocation(3.0)
        by_weight_shares = int(capital * opt3.y_optimal / price_vnd / 100) * 100
        cmp = pd.DataFrame({
            "Theo rủi ro mỗi lệnh": {
                "Nguyên tắc": f"Mất tối đa {risk_pct:.1%} vốn nếu chạm cắt lỗ",
                "Số cổ phiếu": f"{ps['shares']:,}",
                "Giá trị": vnd(ps["value"]),
                "Tỷ trọng": pct(ps["weight"], 1),
            },
            "Theo lý thuyết danh mục": {
                "Nguyên tắc": "Tối ưu hữu dụng trung bình – phương sai (A = 3)",
                "Số cổ phiếu": f"{by_weight_shares:,}",
                "Giá trị": vnd(by_weight_shares * price_vnd),
                "Tỷ trọng": pct(opt3.y_optimal, 1),
            },
        })
        st.dataframe(cmp, use_container_width=True)
        note(
            "Hai nguyên tắc trả lời hai câu hỏi khác nhau: <b>định cỡ theo rủi ro</b> kiểm "
            "soát khoản lỗ tối đa của một lệnh cụ thể, còn <b>tỷ trọng danh mục</b> tối ưu "
            "quan hệ lợi nhuận – rủi ro dài hạn. Thực hành thận trọng là lấy giá trị nhỏ hơn "
            "giữa hai con số.<br><br>"
            "ATR (Average True Range) được dùng làm thước đo biên độ dao động thực tế, giúp "
            "đặt cắt lỗ đủ xa để không bị quét bởi nhiễu thường ngày."
        )

with tab3:
    st.markdown("### Backtest danh mục pha trộn cổ phiếu và tiền gửi")
    st.caption("Tái cân bằng hằng tháng, phí giao dịch 0,15% mỗi chiều tính trên phần phải "
               "mua bán để đưa tỷ trọng về mức mục tiêu")

    navs, table = AL.compare_allocations(a.returns, (0.0, 0.25, 0.5, 0.75, 1.0), a.rf_annual)
    st.plotly_chart(viz.nav_comparison(navs), use_container_width=True)

    show = table.copy().astype(object)
    for i in table.index:
        show.loc[i, "Lợi suất năm"] = pct(table.loc[i, "Lợi suất năm"])
        show.loc[i, "Độ biến động"] = pct(table.loc[i, "Độ biến động"])
        show.loc[i, "Sharpe"] = num(table.loc[i, "Sharpe"])
        show.loc[i, "Sụt giảm tối đa"] = pct(table.loc[i, "Sụt giảm tối đa"])
        show.loc[i, "NAV cuối kỳ"] = f"{table.loc[i, 'NAV cuối kỳ']:,.2f} lần"
    st.dataframe(show, use_container_width=True)

    sharpes = table["Sharpe"].dropna()
    note(
        f"<b>Sharpe gần như không đổi</b> giữa các mức phân bổ (dao động "
        f"{sharpes.min():.3f}–{sharpes.max():.3f}) — đúng như lý thuyết đường phân bổ vốn: "
        "mọi danh mục kết hợp cùng một tài sản rủi ro với tài sản phi rủi ro đều nằm trên "
        "cùng một đường thẳng, nên có cùng tỷ số lợi nhuận trên rủi ro. Thay đổi tỷ trọng "
        "chỉ là <i>chọn điểm</i> trên đường đó.<br><br>"
        f"Khác biệt thực sự nằm ở <b>mức sụt giảm tối đa</b>: danh mục toàn cổ phiếu từng mất "
        f"<b>{pct(table.loc['100% cổ phiếu', 'Sụt giảm tối đa'])}</b>, trong khi danh mục 50% "
        f"chỉ mất <b>{pct(table.loc['50% cổ phiếu', 'Sụt giảm tối đa'])}</b>. Con số này quyết "
        "định nhà đầu tư có trụ được qua khủng hoảng hay bán tháo ở đáy."
    )

with tab4:
    st.markdown("### Mua một lần hay chia nhỏ hàng tháng?")
    monthly = st.number_input("Số tiền đầu tư mỗi tháng (triệu đồng)", 1, 500, 10, 1)
    df = AL.dca_vs_lumpsum(a.prices, monthly * 1e6)

    f = go.Figure()
    f.add_trace(go.Scatter(x=df.index, y=df["DCA hàng tháng"] / 1e6, name="DCA hàng tháng",
                           line=dict(color="#2563eb", width=2)))
    f.add_trace(go.Scatter(x=df.index, y=df["Mua một lần"] / 1e6, name="Mua một lần",
                           line=dict(color="#16a34a", width=2)))
    f.add_trace(go.Scatter(x=df.index, y=df["Vốn đã giải ngân (DCA)"] / 1e6,
                           name="Vốn đã giải ngân (DCA)",
                           line=dict(color="#94a3b8", width=1.5, dash="dash")))
    f.update_layout(**viz.LAYOUT, title="Giá trị danh mục theo hai chiến lược (triệu đồng)",
                    yaxis_title="Triệu đồng")
    f.update_yaxes(type="log")
    st.plotly_chart(f, use_container_width=True)

    dca_final = float(df["DCA hàng tháng"].iloc[-1])
    ls_final = float(df["Mua một lần"].iloc[-1])
    invested_dca = float(df["Vốn đã giải ngân (DCA)"].iloc[-1])
    invested_ls = float(df["Vốn đã giải ngân (mua một lần)"].iloc[-1])
    ret_dca = dca_final / invested_dca - 1
    ret_ls = ls_final / invested_ls - 1
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(pd.DataFrame({
            "Bình quân giá (DCA)": {
                "Vốn giải ngân": vnd(invested_dca),
                "Giá trị cuối kỳ": vnd(dca_final),
                "Bội số vốn": f"{dca_final / invested_dca:,.2f} lần",
                "Lợi nhuận trên vốn": pct(ret_dca),
            },
            "Mua một lần": {
                "Vốn giải ngân": vnd(invested_ls),
                "Giá trị cuối kỳ": vnd(ls_final),
                "Bội số vốn": f"{ls_final / invested_ls:,.2f} lần",
                "Lợi nhuận trên vốn": pct(ret_ls),
            },
        }), use_container_width=True)
    with col2:
        note(
            "Hai chiến lược giải ngân số vốn khác nhau nên phải so bằng <b>bội số vốn của "
            "chính chiến lược đó</b>, không được lấy giá trị cuối kỳ của bên này chia cho vốn "
            "của bên kia.<br><br>"
            f"Theo cách tính đó, <b>{'bình quân giá' if ret_dca > ret_ls else 'mua một lần'}</b> "
            f"cho hiệu quả cao hơn: {pct(ret_dca)} so với {pct(ret_ls)}. "
            + ("Nguyên nhân nằm ở điểm xuất phát của mẫu: phương án mua một lần giải ngân toàn "
               "bộ ngay trước khủng hoảng 2008 nên mất hơn một thập kỷ mới hoà vốn, trong khi "
               "DCA vẫn tiếp tục mua được ở vùng giá thấp."
               if ret_dca > ret_ls else
               "Với tài sản có xu hướng tăng dài hạn, đưa vốn vào thị trường sớm thường thắng "
               "về giá trị tuyệt đối.")
            + "<br><br>Dù kết quả thế nào, phương án mua một lần vẫn giả định nhà đầu tư có sẵn "
              "toàn bộ vốn ngay từ đầu — điều hiếm đúng với người sống bằng thu nhập hằng tháng."
        )

    st.markdown("### Xác suất có lãi theo thời gian nắm giữ (dữ liệu lịch sử thực tế)")
    rows = {}
    for label, d in [("1 tháng", 21), ("3 tháng", 63), ("6 tháng", 126), ("1 năm", 252),
                     ("2 năm", 504), ("3 năm", 756), ("5 năm", 1260)]:
        if len(a.prices) <= d:
            continue
        fwd = (a.prices.shift(-d) / a.prices - 1).dropna()
        rows[label] = {
            "Xác suất có lãi": float((fwd > 0).mean()) * 100,
            "Lợi suất trung bình": float(fwd.mean()) * 100,
            "Trung vị": float(fwd.median()) * 100,
            "Xấu nhất": float(fwd.min()) * 100,
            "Tốt nhất": float(fwd.max()) * 100,
        }
    hist = pd.DataFrame(rows).T

    f = go.Figure()
    f.add_trace(go.Bar(x=hist.index, y=hist["Xác suất có lãi"], name="Xác suất có lãi (%)",
                       marker_color="#16a34a",
                       text=[f"{v:.0f}%" for v in hist["Xác suất có lãi"]],
                       textposition="outside"))
    f.add_hline(y=50, line_dash="dot", line_color="#94a3b8")
    f.update_layout(**viz.LAYOUT, title="Nắm giữ càng lâu, xác suất có lãi càng cao",
                    yaxis_title="%")
    st.plotly_chart(f, use_container_width=True)
    st.dataframe(hist.round(1), use_container_width=True)

    note(
        "Khác với trang Monte Carlo (mô phỏng), bảng này tính trên <b>toàn bộ dữ liệu lịch sử "
        "có thật</b>: với mỗi phiên trong quá khứ, giả sử mua vào rồi giữ đúng khoảng thời "
        "gian tương ứng và ghi nhận kết quả. Hai cách tiếp cận cho kết luận giống nhau về "
        "chiều hướng, củng cố độ tin cậy của nhận định."
    )
