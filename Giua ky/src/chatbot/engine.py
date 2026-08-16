"""Bộ trả lời của chatbot: mỗi câu hỏi gợi ý ứng với một hàm tính toán thật."""

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analytics import allocation as AL
from src.analytics import metrics as M
from src.analytics import technical as TA
from src.analytics.apt import FACTOR_LABELS
from src.chatbot.questions import QUESTIONS, render
from src.config import TRADING_DAYS
from src.formatting import money_vnd as vnd
from src.formatting import num, pct
from src.pipeline import Analysis


@dataclass
class Answer:
    text: str
    table: pd.DataFrame | None = None
    figure: object | None = None
    followups: list[str] = field(default_factory=list)


# Định dạng
def _ov(a: Analysis, field_name: str, default=None):
    """Lấy một trường trong bảng hồ sơ doanh nghiệp."""
    if a.overview is None or a.overview.empty or field_name not in a.overview.columns:
        return default
    v = a.overview.iloc[0][field_name]
    return default if pd.isna(v) else v


def _latest_ratio(a: Analysis) -> pd.Series | None:
    if a.ratio is None or a.ratio.empty:
        return None
    df = a.ratio.copy()
    sort_cols = [c for c in ("year", "quarter") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df.iloc[-1]


def _period_return(a: Analysis, start: str, end: str) -> float:
    s = a.prices.loc[start:end]
    if len(s) < 2:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[0] - 1)


# Nhóm: doanh nghiệp
def co_profile(a: Analysis) -> Answer:
    name = _ov(a, "organ_name", a.ticker)
    sector = _ov(a, "sector", "—")
    profile = str(_ov(a, "company_profile", "") or "")
    profile = re.sub(r"<[^>]+>", " ", profile)
    profile = re.sub(r"\s+", " ", profile).strip()
    if len(profile) > 900:
        profile = profile[:900].rsplit(".", 1)[0] + "."
    txt = f"**{name}** (mã {a.ticker}) — ngành **{sector}**.\n\n"
    txt += profile if profile else "Chưa lấy được phần mô tả doanh nghiệp từ nguồn dữ liệu."
    return Answer(txt, followups=["co_marketcap", "co_valuation", "co_shareholders"])


def co_listing(a: Analysis) -> Answer:
    first = a.prices.index[0]
    years = (a.prices.index[-1] - first).days / 365.25
    listing = _ov(a, "listing_date")
    txt = (
        f"Dữ liệu giá của **{a.ticker}** trong hệ thống bắt đầu từ **{first:%d/%m/%Y}** — "
        f"đây cũng là phiên giao dịch đầu tiên của cổ phiếu trên sàn HOSE.\n\n"
        f"- Độ dài lịch sử: **{years:.1f} năm** ({len(a.prices):,} phiên)\n"
        f"- Phiên gần nhất: **{a.last_date:%d/%m/%Y}**\n"
        f"- Nhóm chỉ số: **{_ov(a, 'com_group_code', 'VNINDEX')}**"
    )
    if listing:
        txt += f"\n- Ngày niêm yết theo hồ sơ: **{listing}**"
    return Answer(txt, followups=["px_now", "px_cagr", "rk_crisis"])


def co_marketcap(a: Analysis) -> Answer:
    mc = _ov(a, "market_cap")
    shares = _ov(a, "issue_share")
    ff = _ov(a, "free_float_percentage")
    rows = {
        "Vốn hoá thị trường": vnd(mc) if mc else "—",
        "Số cổ phiếu lưu hành": f"{shares:,.0f} cổ phiếu" if shares else "—",
        "Giá hiện tại": f"{a.last_price:,.2f} nghìn đồng",
        "KLGD bình quân 1 tháng": f"{_ov(a, 'average_match_volume1_month', float('nan')):,.0f} cổ phiếu",
        "GTGD bình quân 1 tháng": vnd(_ov(a, "average_match_value1_month")),
        "Tỷ lệ cổ phiếu tự do chuyển nhượng": pct(ff) if ff else "—",
    }
    txt = (
        f"**{a.ticker}** có vốn hoá **{vnd(mc)}**"
        + (f", tương ứng {shares / 1e6:,.0f} triệu cổ phiếu lưu hành." if shares else ".")
        + "\n\nĐây là chỉ tiêu phản ánh quy mô doanh nghiệp trên thị trường, bằng giá "
          "cổ phiếu nhân số lượng cổ phiếu đang lưu hành."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["co_valuation", "co_foreign"])


def co_shareholders(a: Analysis) -> Answer:
    if a.shareholders is None or a.shareholders.empty:
        return Answer("Chưa có dữ liệu cổ đông cho mã này.")
    df = a.shareholders.copy()
    df = df.sort_values("share_own_percent", ascending=False).head(10)
    tbl = pd.DataFrame({
        "Cổ đông": df["share_holder"].values,
        "Số cổ phiếu": [f"{v:,.0f}" for v in df["quantity"]],
        "Tỷ lệ sở hữu": [pct(v) for v in df["share_own_percent"]],
    })
    top = df.iloc[0]
    txt = (
        f"Cổ đông lớn nhất của **{a.ticker}** là **{top['share_holder']}** với "
        f"**{pct(top['share_own_percent'])}** vốn điều lệ. "
        f"Mười cổ đông lớn nhất nắm tổng cộng **{pct(df['share_own_percent'].sum())}**."
    )
    return Answer(txt, tbl, followups=["co_foreign", "co_marketcap"])


def co_foreign(a: Analysis) -> Answer:
    cur = _ov(a, "foreigner_percentage")
    mx = _ov(a, "maximum_foreign_percentage")
    state = _ov(a, "state_percentage")
    txt = (
        f"Nhà đầu tư nước ngoài đang nắm **{pct(cur)}** vốn của {a.ticker}, "
        f"trần cho phép là **{pct(mx)}**."
    )
    if cur is not None and mx is not None:
        room = mx - cur
        txt += (
            f"\n\nRoom ngoại còn lại khoảng **{pct(room)}** vốn điều lệ. "
            + ("Room gần cạn — dòng vốn ngoại khó mua thêm nhiều, đây thường là yếu tố "
               "hỗ trợ giá nhưng cũng làm giảm thanh khoản với khối ngoại."
               if room < 0.03 else
               "Room vẫn còn dư địa cho dòng vốn ngoại.")
        )
    if state:
        txt += f"\n\nSở hữu nhà nước: **{pct(state)}**."
    return Answer(txt, followups=["co_shareholders", "co_marketcap"])


def co_valuation(a: Analysis) -> Answer:
    r = _latest_ratio(a)
    if r is None:
        return Answer("Chưa có dữ liệu chỉ số tài chính cho mã này.")
    fields = [
        ("pe", "P/E — giá trên lợi nhuận mỗi cổ phiếu", "lần"),
        ("pb", "P/B — giá trên giá trị sổ sách", "lần"),
        ("ps", "P/S — giá trên doanh thu", "lần"),
        ("ev_to_ebitda", "EV/EBITDA", "lần"),
        ("roe", "ROE — lợi nhuận trên vốn chủ", "%"),
        ("roa", "ROA — lợi nhuận trên tổng tài sản", "%"),
        ("gross_margin", "Biên lợi nhuận gộp", "%"),
        ("after_tax_profit_margin", "Biên lợi nhuận sau thuế", "%"),
        ("debt_to_equity", "Nợ trên vốn chủ sở hữu", "lần"),
        ("dividend_yield", "Tỷ suất cổ tức", "%"),
    ]
    rows = {}
    for key, label, unit in fields:
        if key in r.index and pd.notna(r[key]):
            v = float(r[key])
            rows[label] = pct(v) if unit == "%" else f"{v:,.2f} lần"
    period = f"{int(r['year'])} quý {int(r['quarter'])}" if "year" in r.index else "kỳ gần nhất"
    pe, pb, roe = r.get("pe"), r.get("pb"), r.get("roe")
    txt = (
        f"Theo dữ liệu **{period}**, {a.ticker} giao dịch ở mức "
        f"**P/E {pe:,.1f} lần** và **P/B {pb:,.1f} lần**, với **ROE {pct(roe, 1)}**.\n\n"
        "P/E cho biết nhà đầu tư trả bao nhiêu đồng cho mỗi đồng lợi nhuận; "
        "ROE đo khả năng sinh lời trên vốn chủ sở hữu. ROE cao đi kèm P/B cao thường "
        "phản ánh thị trường sẵn sàng trả giá cho chất lượng lợi nhuận."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["co_growth", "capm_fair"])


def co_growth(a: Analysis) -> Answer:
    if a.ratio is None or a.ratio.empty:
        return Answer("Chưa có dữ liệu chỉ số tài chính theo thời gian.")
    df = a.ratio.copy()
    cols = [c for c in ("year", "quarter", "pe", "pb", "roe", "roa", "gross_margin",
                        "after_tax_profit_margin", "debt_to_equity") if c in df.columns]
    df = df[cols].sort_values(["year", "quarter"]).tail(12)
    show = df.rename(columns={
        "year": "Năm", "quarter": "Quý", "pe": "P/E", "pb": "P/B", "roe": "ROE",
        "roa": "ROA", "gross_margin": "Biên gộp",
        "after_tax_profit_margin": "Biên ròng", "debt_to_equity": "Nợ/VCSH",
    }).round(3)
    first_roe, last_roe = df["roe"].iloc[0], df["roe"].iloc[-1]
    trend = "cải thiện" if last_roe > first_roe else "giảm"
    txt = (
        f"Trong {len(df)} kỳ gần nhất, ROE của {a.ticker} đi từ **{pct(first_roe, 1)}** "
        f"xuống **{pct(last_roe, 1)}**" if last_roe < first_roe else
        f"Trong {len(df)} kỳ gần nhất, ROE của {a.ticker} đi từ **{pct(first_roe, 1)}** "
        f"lên **{pct(last_roe, 1)}**"
    )
    txt += f" — xu hướng hiệu quả sinh lời đang **{trend}**."
    return Answer(txt, show.set_index("Năm"), followups=["co_valuation"])


# Nhóm: giá và hiệu quả
def px_now(a: Analysis) -> Answer:
    chg1 = a.returns.iloc[-1]
    hi = _ov(a, "highest_price1_year")
    lo = _ov(a, "lowest_price1_year")
    txt = (
        f"Giá đóng cửa phiên **{a.last_date:%d/%m/%Y}** của {a.ticker} là "
        f"**{a.last_price:,.2f} nghìn đồng/cổ phiếu** "
        f"({'tăng' if chg1 >= 0 else 'giảm'} {pct(abs(chg1))} so với phiên trước)."
    )
    if hi and lo:
        pos = (a.last_price * 1000 - lo) / (hi - lo)
        txt += (
            f"\n\nTrong 52 tuần, giá dao động từ **{lo / 1000:,.1f}** đến "
            f"**{hi / 1000:,.1f}** nghìn đồng; mức hiện tại nằm ở **{pct(pos, 0)}** "
            "biên độ đó tính từ đáy."
        )
    return Answer(txt, followups=["px_change", "tc_trend", "co_valuation"])


def px_change(a: Analysis) -> Answer:
    periods = [("1 tuần", 5), ("1 tháng", 21), ("3 tháng", 63), ("6 tháng", 126),
               ("1 năm", 252), ("3 năm", 756), ("5 năm", 1260)]
    rows = {}
    for label, d in periods:
        v = a.price_change(d)
        b = float(a.benchmark.iloc[-1] / a.benchmark.iloc[-1 - d] - 1) if len(a.benchmark) > d else np.nan
        rows[label] = {a.ticker: pct(v), "VNINDEX": pct(b),
                       "Chênh lệch": pct(v - b) if not np.isnan(v) and not np.isnan(b) else "—"}
    tbl = pd.DataFrame(rows).T
    m1, y1 = a.price_change(21), a.price_change(252)
    txt = (
        f"So với 1 tháng trước, {a.ticker} **{'tăng' if m1 >= 0 else 'giảm'} {pct(abs(m1))}**; "
        f"so với 1 năm trước **{'tăng' if y1 >= 0 else 'giảm'} {pct(abs(y1))}**."
    )
    return Answer(txt, tbl, followups=["px_vs_index", "px_best_worst"])


def px_cagr(a: Analysis) -> Answer:
    years = (a.prices.index[-1] - a.prices.index[0]).days / 365.25
    rows = {
        "Lợi suất tích luỹ toàn kỳ": pct(M.total_return(a.returns)),
        "Lợi suất kép hằng năm (CAGR)": pct(a.mu_annual),
        "Độ biến động hằng năm": pct(a.sigma_annual),
        "Tỷ số Sharpe": num(M.sharpe_ratio(a.returns, a.rf_annual)),
        "Số năm dữ liệu": f"{years:.1f} năm",
    }
    txt = (
        f"Trong **{years:.1f} năm** ({a.prices.index[0]:%m/%Y} – {a.last_date:%m/%Y}), "
        f"{a.ticker} đạt lợi suất kép **{pct(a.mu_annual)}/năm**, tức 1 đồng vốn ban đầu "
        f"trở thành **{(1 + M.total_return(a.returns)):,.2f} đồng**.\n\n"
        f"Để so sánh: VNINDEX cùng kỳ đạt **{pct(M.annual_return(a.benchmark_returns))}/năm** "
        f"và lãi suất gửi ngân hàng giả định trong mô hình là **{pct(a.rf_annual)}/năm**."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["px_vs_index", "px_invest_100"])


def px_vs_index(a: Analysis) -> Answer:
    tbl = pd.concat([a.summary, a.benchmark_summary], axis=1)
    fmt = tbl.copy()
    for idx in fmt.index:
        for col in fmt.columns:
            v = tbl.loc[idx, col]
            fmt.loc[idx, col] = pct(v) if any(k in idx for k in
                ("Lợi suất", "biến động", "giảm", "Tỷ lệ", "VaR", "CVaR")) else num(v)
    diff = a.mu_annual - M.annual_return(a.benchmark_returns)
    txt = (
        f"{a.ticker} đạt **{pct(a.mu_annual)}/năm** so với **"
        f"{pct(M.annual_return(a.benchmark_returns))}/năm** của VNINDEX — "
        f"vượt trội **{pct(diff)} mỗi năm**.\n\n"
        f"Nhưng đi kèm rủi ro cao hơn: biến động **{pct(a.sigma_annual)}** so với "
        f"**{pct(M.annual_volatility(a.benchmark_returns))}** của chỉ số. "
        f"Sau khi điều chỉnh rủi ro, Sharpe của {a.ticker} là "
        f"**{num(M.sharpe_ratio(a.returns, a.rf_annual))}** còn của VNINDEX là "
        f"**{num(M.sharpe_ratio(a.benchmark_returns, a.rf_annual))}**."
    )
    return Answer(txt, fmt, followups=["capm_alpha", "rk_sharpe"])


def px_best_worst(a: Analysis) -> Answer:
    yearly = a.yearly_returns()
    yearly.index = yearly.index.year
    best, worst = yearly.idxmax(), yearly.idxmin()
    tbl = pd.DataFrame({
        "Lợi suất năm": [pct(v) for v in yearly.values],
    }, index=yearly.index)
    txt = (
        f"Năm tốt nhất của {a.ticker} là **{best}** với **{pct(yearly.max())}**; "
        f"năm tệ nhất là **{worst}** với **{pct(yearly.min())}**.\n\n"
        f"Trong {len(yearly)} năm có dữ liệu, **{int((yearly > 0).sum())} năm tăng** và "
        f"**{int((yearly <= 0).sum())} năm giảm** — tỷ lệ năm có lãi là "
        f"**{pct((yearly > 0).mean(), 0)}**."
    )
    return Answer(txt, tbl, followups=["px_monthly", "rk_crisis"])


def px_invest_100(a: Analysis) -> Answer:
    growth = 1 + M.total_return(a.returns)
    bench_growth = 1 + M.total_return(a.benchmark_returns)
    years = (a.prices.index[-1] - a.prices.index[0]).days / 365.25
    saving = (1 + a.rf_annual) ** years
    tbl = pd.DataFrame({"Giá trị hiện tại": {
        f"Mua và nắm giữ {a.ticker}": vnd(100e6 * growth),
        "Đầu tư theo VNINDEX": vnd(100e6 * bench_growth),
        f"Gửi tiết kiệm {pct(a.rf_annual, 1)}/năm": vnd(100e6 * saving),
    }})
    txt = (
        f"100 triệu đồng đầu tư vào {a.ticker} từ phiên đầu tiên "
        f"({a.prices.index[0]:%d/%m/%Y}) và giữ tới nay sẽ thành **{vnd(100e6 * growth)}** "
        f"— gấp **{growth:,.1f} lần** sau {years:.1f} năm.\n\n"
        "Con số này chưa tính cổ tức tiền mặt nhận thêm và giả định không mua bán trong kỳ; "
        "giá đã được điều chỉnh theo các đợt chia tách, thưởng cổ phiếu."
    )
    return Answer(txt, tbl, followups=["px_cagr", "mg_dca"])


def px_monthly(a: Analysis) -> Answer:
    table = M.monthly_return_table(a.returns)
    monthly_avg = table.drop(columns=["Cả năm"], errors="ignore").mean()
    best_m, worst_m = monthly_avg.idxmax(), monthly_avg.idxmin()
    txt = (
        f"Tính trung bình toàn bộ lịch sử, tháng tốt nhất của {a.ticker} là "
        f"**{best_m}** ({pct(monthly_avg.max())}/tháng) và yếu nhất là **{worst_m}** "
        f"({pct(monthly_avg.min())}/tháng).\n\n"
        "Lưu ý: quy luật mùa vụ tính trên số ít quan sát mỗi tháng nên độ tin cậy thấp, "
        "không nên dùng làm căn cứ giao dịch duy nhất."
    )
    show = (table * 100).round(1)
    return Answer(txt, show, followups=["px_best_worst"])


# Nhóm: rủi ro
def rk_vol(a: Analysis) -> Answer:
    daily = float(a.returns.std(ddof=1))
    recent = float(a.returns.tail(252).std(ddof=1) * np.sqrt(TRADING_DAYS))
    rows = {
        "Độ lệch chuẩn theo phiên": pct(daily),
        "Độ biến động năm (toàn kỳ)": pct(a.sigma_annual),
        "Độ biến động năm (12 tháng gần nhất)": pct(recent),
        "Độ biến động VNINDEX (toàn kỳ)": pct(M.annual_volatility(a.benchmark_returns)),
        "Phiên biến động mạnh nhất": pct(float(a.returns.abs().max())),
    }
    txt = (
        f"Độ biến động (độ lệch chuẩn lợi suất) của {a.ticker} là **{pct(a.sigma_annual)}/năm**, "
        f"cao hơn mức **{pct(M.annual_volatility(a.benchmark_returns))}** của VNINDEX.\n\n"
        f"Diễn giải trực quan: nếu lợi suất phân phối chuẩn, khoảng 2/3 số năm lợi suất sẽ "
        f"nằm trong khoảng **{pct(a.mu_annual - a.sigma_annual)}** đến "
        f"**{pct(a.mu_annual + a.sigma_annual)}**.\n\n"
        f"12 tháng gần đây mức biến động là **{pct(recent)}** — "
        + ("cao hơn" if recent > a.sigma_annual else "thấp hơn") + " trung bình lịch sử."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["rk_var", "rk_dd"])


def rk_var(a: Analysis) -> Answer:
    rows = {}
    for lv in (0.90, 0.95, 0.99):
        rows[f"VaR {lv:.0%} lịch sử"] = pct(M.var_historical(a.returns, lv))
        rows[f"CVaR {lv:.0%} lịch sử"] = pct(M.cvar_historical(a.returns, lv))
    rows["VaR 95% tham số (chuẩn)"] = pct(M.var_parametric(a.returns, 0.95))
    v95 = M.var_historical(a.returns, 0.95)
    c95 = M.cvar_historical(a.returns, 0.95)
    txt = (
        f"**VaR 95% một ngày = {pct(v95)}**: với 95% số phiên, khoản lỗ không vượt quá "
        f"{pct(abs(v95))} giá trị khoản đầu tư. Nói cách khác, trung bình cứ 20 phiên thì "
        f"có 1 phiên lỗ nặng hơn mức này.\n\n"
        f"**CVaR 95% = {pct(c95)}**: khi rơi vào nhóm 5% phiên xấu nhất đó, mức lỗ trung bình "
        f"là {pct(abs(c95))}. CVaR luôn xấu hơn VaR và cho biết mức độ nghiêm trọng của phần đuôi.\n\n"
        f"Ví dụ với danh mục 100 triệu đồng: ngưỡng lỗ ngày là khoảng "
        f"**{vnd(abs(v95) * 100e6)}**, còn nếu rơi vào vùng xấu thì trung bình mất "
        f"**{vnd(abs(c95) * 100e6)}**."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["th_var", "mc_var"])


def rk_dd(a: Analysis) -> Answer:
    dd = M.drawdown_series(a.returns)
    trough = dd.idxmin()
    peak = a.prices.loc[:trough].idxmax()
    after = dd.loc[trough:]
    recovered = after[after >= -1e-9]
    rec_date = recovered.index[0] if len(recovered) else None
    rows = {
        "Sụt giảm sâu nhất": pct(dd.min()),
        "Đỉnh trước đó": f"{peak:%d/%m/%Y} — {a.prices.loc[peak]:,.2f}",
        "Đáy": f"{trough:%d/%m/%Y} — {a.prices.loc[trough]:,.2f}",
        "Thời gian rơi": f"{(trough - peak).days} ngày",
        "Ngày hồi phục lại đỉnh": f"{rec_date:%d/%m/%Y}" if rec_date is not None else "Chưa hồi phục",
        "Thời gian hồi phục": f"{(rec_date - trough).days} ngày" if rec_date is not None else "—",
        "Sụt giảm hiện tại": pct(dd.iloc[-1]),
    }
    txt = (
        f"Mức sụt giảm sâu nhất của {a.ticker} là **{pct(dd.min())}**, xảy ra từ đỉnh "
        f"{peak:%d/%m/%Y} tới đáy {trough:%d/%m/%Y}"
        + (f", và phải tới {rec_date:%d/%m/%Y} giá mới trở lại đỉnh cũ — mất "
           f"**{(rec_date - trough).days / 365.25:.1f} năm** để hoà vốn."
           if rec_date is not None else " và tới nay vẫn chưa lấy lại đỉnh cũ.")
        + f"\n\nHiện tại giá đang thấp hơn đỉnh lịch sử **{pct(abs(dd.iloc[-1]))}**."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["rk_crisis", "th_dd"])


def rk_sharpe(a: Analysis) -> Answer:
    s = M.sharpe_ratio(a.returns, a.rf_annual)
    rows = {
        "Sharpe": num(s),
        "Sortino": num(M.sortino_ratio(a.returns, a.rf_annual)),
        "Calmar": num(M.calmar_ratio(a.returns)),
        "Treynor": num(a.capm.treynor),
        "Sharpe của VNINDEX": num(M.sharpe_ratio(a.benchmark_returns, a.rf_annual)),
    }
    judge = ("dưới mức trung bình — lợi suất chưa tương xứng rủi ro" if s < 0.5
             else "ở mức chấp nhận được" if s < 1
             else "tốt" if s < 2 else "rất tốt")
    txt = (
        f"Tỷ số Sharpe của {a.ticker} là **{num(s)}** — {judge}.\n\n"
        f"Sharpe đo mỗi đơn vị rủi ro đổi được bao nhiêu lợi suất vượt lãi suất phi rủi ro: "
        f"({pct(a.mu_annual)} − {pct(a.rf_annual)}) / {pct(a.sigma_annual)} ≈ {num(s)}.\n\n"
        f"Thang tham chiếu thường dùng: dưới 1 là bình thường, 1–2 là tốt, trên 2 là rất tốt. "
        f"Sortino **{num(M.sortino_ratio(a.returns, a.rf_annual))}** cao hơn Sharpe cho thấy "
        f"phần biến động của {a.ticker} nghiêng về chiều tăng nhiều hơn chiều giảm."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["th_sharpe", "px_vs_index"])


def rk_split(a: Analysis) -> Answer:
    sys_share = a.capm.systematic_share
    rows = {
        "Tổng độ biến động (năm)": pct(a.sigma_annual),
        "Rủi ro hệ thống (theo thị trường)": pct(a.sigma_annual * np.sqrt(sys_share)),
        "Rủi ro phi hệ thống (riêng doanh nghiệp)": pct(a.capm.resid_std_annual),
        "Tỷ trọng rủi ro hệ thống": pct(sys_share, 1),
        "Tỷ trọng rủi ro phi hệ thống": pct(1 - sys_share, 1),
    }
    txt = (
        f"Tổng rủi ro của {a.ticker} tách thành hai phần:\n\n"
        f"- **Rủi ro hệ thống {pct(sys_share, 0)}** — do biến động chung của thị trường, "
        f"không thể loại bỏ bằng đa dạng hoá, và là phần được thị trường trả công qua beta.\n"
        f"- **Rủi ro phi hệ thống {pct(1 - sys_share, 0)}** ({pct(a.capm.resid_std_annual)}/năm) "
        f"— rủi ro riêng của doanh nghiệp, có thể giảm mạnh nếu nắm nhiều cổ phiếu khác ngành.\n\n"
        "Đây chính là lý do lý thuyết danh mục khuyên không dồn toàn bộ vốn vào một mã: "
        "phần rủi ro riêng không được đền bù bằng lợi suất kỳ vọng cao hơn."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["capm_beta", "mg_weight"])


def rk_crisis(a: Analysis) -> Answer:
    events = [
        ("Khủng hoảng tài chính 2008", "2007-10-01", "2009-02-28"),
        ("Hồi phục 2009", "2009-03-01", "2009-12-31"),
        ("Covid-19 lao dốc", "2020-01-20", "2020-03-31"),
        ("Hồi phục sau Covid", "2020-04-01", "2021-12-31"),
        ("Siết tín dụng 2022", "2022-04-01", "2022-11-15"),
        ("12 tháng gần nhất", str(a.prices.index[-1] - pd.Timedelta(days=365))[:10],
         str(a.prices.index[-1])[:10]),
    ]
    rows = {}
    for label, s, e in events:
        rs, rb = _period_return(a, s, e), None
        b = a.benchmark.loc[s:e]
        if len(b) > 1:
            rb = float(b.iloc[-1] / b.iloc[0] - 1)
        rows[label] = {a.ticker: pct(rs), "VNINDEX": pct(rb) if rb is not None else "—",
                       "Giai đoạn": f"{s} → {e}"}
    tbl = pd.DataFrame(rows).T
    c2008 = _period_return(a, "2007-10-01", "2009-02-28")
    ccovid = _period_return(a, "2020-01-20", "2020-03-31")
    txt = (
        f"Trong khủng hoảng 2008, {a.ticker} mất **{pct(abs(c2008))}** giá trị từ đỉnh tháng "
        f"10/2007 tới đáy tháng 2/2009. Cú sốc Covid tháng 3/2020 nhẹ hơn nhiều: "
        f"**{pct(ccovid)}** chỉ trong hơn 2 tháng.\n\n"
        "Những giai đoạn này cho thấy rủi ro thực tế của cổ phiếu vượt xa những gì độ lệch "
        "chuẩn thông thường mô tả — đó là lý do cần xem thêm CVaR và mức sụt giảm tối đa."
    )
    return Answer(txt, tbl, followups=["rk_dd", "mc_var"])


# Nhóm: CAPM
def capm_beta(a: Analysis) -> Answer:
    c = a.capm
    rows = {
        "Beta (β)": num(c.beta),
        "Sai số chuẩn của β": num(float(c.model.bse["ex_market"])),
        "t-stat của β": num(c.t_beta),
        "Khoảng tin cậy 95%": f"[{c.beta - 1.96 * float(c.model.bse['ex_market']):.2f}; "
                              f"{c.beta + 1.96 * float(c.model.bse['ex_market']):.2f}]",
        "Số quan sát": f"{c.n_obs:,}",
    }
    txt = (
        f"**Beta của {a.ticker} = {num(c.beta)}**, ước lượng trên {c.n_obs:,} phiên với "
        f"t-stat = {num(c.t_beta)} (rất có ý nghĩa thống kê).\n\n"
        f"Ý nghĩa: khi VNINDEX biến động 1%, {a.ticker} biến động trung bình "
        f"**{num(c.beta)}%** cùng chiều. "
        + ("Beta dưới 1 nghĩa là cổ phiếu ít nhạy hơn thị trường, mang tính phòng thủ."
           if c.beta < 1 else
           "Beta trên 1 nghĩa là cổ phiếu khuếch đại biến động thị trường, mang tính tấn công.")
        + "\n\nBeta là thước đo rủi ro hệ thống — phần rủi ro không thể loại bỏ bằng đa dạng hoá."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}),
                  followups=["capm_rolling", "th_beta", "rk_split"])


def capm_alpha(a: Analysis) -> Answer:
    c = a.capm
    sig = c.p_alpha < 0.05
    rows = {
        "Alpha theo ngày": f"{c.alpha_daily * 100:.4f}%",
        "Alpha năm hoá (Jensen)": pct(c.alpha_annual),
        "t-stat của α": num(c.t_alpha),
        "p-value của α": num(c.p_alpha, 4),
        "Kết luận ở mức 5%": "Có ý nghĩa thống kê" if sig else "Chưa có ý nghĩa thống kê",
    }
    txt = (
        f"**Alpha Jensen = {pct(c.alpha_annual)}/năm** với p-value = {num(c.p_alpha, 3)}.\n\n"
        f"Alpha là phần lợi suất vượt trên mức mà CAPM đòi hỏi cho rủi ro hệ thống đã gánh. "
        f"Alpha dương {pct(c.alpha_annual)} nghĩa là suốt giai đoạn nghiên cứu, {a.ticker} "
        f"mang lại nhiều hơn kỳ vọng khoảng {pct(c.alpha_annual)} mỗi năm.\n\n"
        + ("Tuy nhiên p-value > 0,05 nên về mặt thống kê **chưa thể khẳng định** alpha khác 0 — "
           "kết quả này có thể do may mắn của mẫu quan sát. Đây là điều thường gặp: alpha rất "
           "khó đạt ý nghĩa thống kê trên dữ liệu ngày."
           if not sig else
           "p-value < 0,05 nên alpha khác 0 có ý nghĩa thống kê ở mức tin cậy 95%.")
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["capm_expected", "capm_fair"])


def capm_expected(a: Analysis) -> Answer:
    c = a.capm
    rows = {
        "Lãi suất phi rủi ro (Rf)": pct(c.rf_annual),
        "Lợi suất thị trường (Rm)": pct(c.market_return_annual),
        "Phần bù rủi ro thị trường (Rm − Rf)": pct(c.market_return_annual - c.rf_annual),
        "Beta": num(c.beta),
        "Lợi suất kỳ vọng theo CAPM": pct(c.expected_return),
        "Lợi suất thực tế đạt được": pct(c.realized_return),
        "Chênh lệch": pct(c.realized_return - c.expected_return),
    }
    txt = (
        "Công thức CAPM: **E(Ri) = Rf + β × (Rm − Rf)**\n\n"
        f"Thay số: {pct(c.rf_annual)} + {num(c.beta)} × "
        f"({pct(c.market_return_annual)} − {pct(c.rf_annual)}) = **{pct(c.expected_return)}/năm**.\n\n"
        f"Đây là mức lợi suất tối thiểu mà nhà đầu tư nên đòi hỏi khi nắm giữ {a.ticker}, "
        f"tương ứng với rủi ro hệ thống của cổ phiếu. Thực tế cổ phiếu đạt "
        f"**{pct(c.realized_return)}/năm**, tức "
        + ("vượt" if c.realized_return > c.expected_return else "thấp hơn")
        + f" kỳ vọng **{pct(abs(c.realized_return - c.expected_return))}**."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["capm_fair", "apt_expected"])


def capm_fair(a: Analysis) -> Answer:
    c = a.capm
    over = c.realized_return > c.expected_return
    txt = (
        f"Theo khung CAPM, {a.ticker} nằm **{'phía trên' if over else 'phía dưới'} đường thị "
        f"trường chứng khoán (SML)**: lợi suất thực tế {pct(c.realized_return)} so với mức "
        f"kỳ vọng {pct(c.expected_return)} cho beta {num(c.beta)}.\n\n"
        + ("Cổ phiếu nằm trên SML tức là sinh lời cao hơn mức đền bù rủi ro thông thường — "
           "theo lý thuyết là **đang bị định giá thấp** so với rủi ro, hoặc doanh nghiệp có "
           "lợi thế mà mô hình một nhân tố chưa nắm bắt được."
           if over else
           "Cổ phiếu nằm dưới SML tức là chưa bù đắp đủ rủi ro hệ thống — theo lý thuyết là "
           "**đang bị định giá cao**.")
        + "\n\nCần lưu ý: kết luận này dựa trên dữ liệu quá khứ và giả định beta ổn định, "
          "không phải khuyến nghị mua bán. CAPM chỉ giải thích được "
        f"{pct(c.r_squared, 0)} biến động của cổ phiếu."
    )
    return Answer(txt, followups=["capm_r2", "co_valuation", "apt_vs_capm"])


def capm_r2(a: Analysis) -> Answer:
    c = a.capm
    rows = {
        "R² của CAPM": pct(c.r_squared, 1),
        "R² hiệu chỉnh của APT": pct(a.apt.adj_r_squared, 1),
        "Rủi ro phi hệ thống (năm)": pct(c.resid_std_annual),
        "Số quan sát": f"{c.n_obs:,} phiên",
    }
    txt = (
        f"**R² = {pct(c.r_squared, 1)}** — thị trường giải thích được khoảng "
        f"{pct(c.r_squared, 0)} biến động lợi suất của {a.ticker}; "
        f"{pct(1 - c.r_squared, 0)} còn lại đến từ yếu tố riêng của doanh nghiệp "
        "(kết quả kinh doanh, tin tức, dòng tiền của cổ phiếu này).\n\n"
        "Mức R² như vậy là bình thường với một cổ phiếu đơn lẻ. Nếu xét cả danh mục nhiều mã, "
        "R² so với thị trường thường cao hơn nhiều vì phần riêng lẻ triệt tiêu lẫn nhau."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["rk_split", "apt_vs_capm"])


def capm_rolling(a: Analysis) -> Answer:
    from src.analytics.capm import rolling_beta

    rb = rolling_beta(a.returns, a.benchmark_returns, 126)
    rows = {
        "Beta toàn kỳ": num(a.capm.beta),
        "Beta trượt hiện tại": num(float(rb.iloc[-1])),
        "Beta trượt cao nhất": num(float(rb.max())),
        "Beta trượt thấp nhất": num(float(rb.min())),
        "Độ lệch chuẩn của beta trượt": num(float(rb.std())),
    }
    txt = (
        f"Beta của {a.ticker} không cố định: ước lượng trượt 126 phiên dao động từ "
        f"**{num(rb.min())}** đến **{num(rb.max())}**, hiện ở mức **{num(rb.iloc[-1])}** "
        f"so với beta toàn kỳ **{num(a.capm.beta)}**.\n\n"
        "Beta thay đổi theo thời gian là hạn chế quan trọng của CAPM tĩnh: dùng một con số "
        "beta duy nhất cho suốt 20 năm sẽ bỏ qua việc cấu trúc rủi ro doanh nghiệp và trạng "
        "thái thị trường đã đổi khác."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["capm_beta", "capm_r2"])


# Nhóm: APT
def apt_factors(a: Analysis) -> Answer:
    r = a.apt
    tbl = r.to_frame().round(4)
    sig = [FACTOR_LABELS.get(f, f) for f in r.params.drop("const").index
           if r.pvalues[f] < 0.05]
    txt = (
        f"Mô hình APT ước lượng {a.ticker} theo {len(r.params) - 1} nhân tố rủi ro vĩ mô "
        f"trên {r.n_obs:,} phiên.\n\n"
        + (f"Các nhân tố **có ý nghĩa thống kê ở mức 5%**: {', '.join(sig)}.\n\n"
           if sig else "Không nhân tố nào đạt ý nghĩa thống kê ở mức 5%.\n\n")
        + "Hệ số b cho biết khi nhân tố biến động 1%, lợi suất cổ phiếu thay đổi bao nhiêu %, "
          "trong điều kiện các nhân tố khác không đổi."
    )
    return Answer(txt, tbl, followups=["apt_vs_capm", "apt_usd", "apt_expected"])


def apt_vs_capm(a: Analysis) -> Answer:
    tbl = pd.DataFrame({
        "CAPM (1 nhân tố)": {"R²": pct(a.capm.r_squared, 1), "Số nhân tố": "1",
                             "Lợi suất kỳ vọng": pct(a.capm.expected_return)},
        f"APT ({len(a.apt.params) - 1} nhân tố)": {
            "R²": pct(a.apt.r_squared, 1), "Số nhân tố": str(len(a.apt.params) - 1),
            "Lợi suất kỳ vọng": pct(a.apt.expected_return)},
    })
    gain = a.apt.r_squared - a.capm.r_squared
    txt = (
        f"R² của APT là **{pct(a.apt.r_squared, 1)}**, chỉ nhỉnh hơn CAPM "
        f"(**{pct(a.capm.r_squared, 1)}**) khoảng **{pct(gain, 1)}**.\n\n"
        "Kết quả này nói lên điều quan trọng: với cổ phiếu Việt Nam, **nhân tố thị trường "
        "đóng vai trò áp đảo**, còn các biến vĩ mô toàn cầu (giá dầu, vàng, tỷ giá) chỉ bổ "
        "sung rất ít khả năng giải thích ở tần suất ngày. Thêm nhân tố làm mô hình phức tạp "
        "hơn nhưng không cải thiện đáng kể — đúng tinh thần cần cân nhắc giữa độ phù hợp và "
        "tính tiết kiệm tham số.\n\n"
        "Ưu điểm của APT là không cần giả định danh mục thị trường hiệu quả và cho phép nhiều "
        "nguồn rủi ro; nhược điểm là lý thuyết không chỉ rõ nhân tố nào phải đưa vào."
    )
    return Answer(txt, tbl, followups=["th_apt", "apt_factors"])


def apt_expected(a: Analysis) -> Answer:
    r = a.apt
    contrib = pd.DataFrame({
        "Phần bù nhân tố (năm)": [pct(v) for v in r.factor_premia],
        "Hệ số nhạy b": [num(v) for v in r.params.drop("const")],
        "Đóng góp vào lợi suất": [pct(v) for v in r.contributions],
    }, index=[FACTOR_LABELS.get(i, i) for i in r.contributions.index])
    txt = (
        f"Lợi suất kỳ vọng theo APT = lãi suất phi rủi ro **{pct(a.rf_annual)}** cộng tổng "
        f"đóng góp của các nhân tố = **{pct(r.expected_return)}/năm**.\n\n"
        f"So sánh: CAPM cho **{pct(a.capm.expected_return)}/năm**, lợi suất thực tế trong kỳ "
        f"là **{pct(a.capm.realized_return)}/năm**.\n\n"
        "Bảng dưới tách rõ mỗi nhân tố đóng góp bao nhiêu vào con số kỳ vọng đó."
    )
    return Answer(txt, contrib, followups=["capm_expected", "apt_factors"])


def apt_usd(a: Analysis) -> Answer:
    r = a.apt
    if "USDVND" not in r.params.index:
        return Answer("Mô hình hiện không có nhân tố tỷ giá.")
    b, p = float(r.params["USDVND"]), float(r.pvalues["USDVND"])
    txt = (
        f"Hệ số nhạy của {a.ticker} với tỷ giá USD/VND là **b = {num(b)}** "
        f"(p-value = {num(p, 3)}).\n\n"
        f"Nghĩa là khi VND mất giá 1% so với USD, lợi suất {a.ticker} thay đổi trung bình "
        f"**{num(b)}%** — "
        + ("cùng chiều, phù hợp với doanh nghiệp có doanh thu xuất khẩu hoặc thu ngoại tệ."
           if b > 0 else
           "ngược chiều, phù hợp với doanh nghiệp có chi phí nhập khẩu hoặc nợ ngoại tệ.")
        + "\n\n"
        + ("Tuy nhiên p-value > 0,05 nên tác động này **chưa có ý nghĩa thống kê** — "
           "không nên diễn giải quá mức."
           if p >= 0.05 else
           "p-value < 0,05 nên tác động này có ý nghĩa thống kê.")
    )
    return Answer(txt, followups=["apt_factors", "apt_vs_capm"])


# Nhóm: Monte Carlo
def mc_1y(a: Analysis) -> Answer:
    sim = a.simulate(252, 10000, "gbm")
    st = sim.stats()
    rows = {
        "Giá hiện tại": f"{a.last_price:,.2f}",
        "Trung vị sau 1 năm": f"{st['Trung vị']:,.2f}",
        "Kỳ vọng sau 1 năm": f"{st['Giá kỳ vọng']:,.2f}",
        "Khoảng 90% tin cậy": f"{st['Phân vị 5%']:,.2f} — {st['Phân vị 95%']:,.2f}",
        "Xác suất tăng giá": pct(1 - st["Xác suất lỗ"], 1),
    }
    txt = (
        f"Mô phỏng 10.000 kịch bản bằng chuyển động Brown hình học, tham số ước lượng từ "
        f"{len(a.log_returns):,} phiên lịch sử:\n\n"
        f"- Giá trung vị sau 1 năm: **{st['Trung vị']:,.1f} nghìn đồng** "
        f"(hiện tại {a.last_price:,.1f})\n"
        f"- 90% kịch bản rơi vào khoảng **{st['Phân vị 5%']:,.1f} – {st['Phân vị 95%']:,.1f}**\n"
        f"- Xác suất giá cao hơn hiện tại: **{pct(1 - st['Xác suất lỗ'], 0)}**\n\n"
        "Khoảng dự báo rất rộng — đó chính là thông điệp của mô phỏng: với độ biến động "
        f"{pct(a.sigma_annual)}/năm, không thể dự báo điểm cho giá cổ phiếu, chỉ có thể mô tả "
        "phân phối xác suất của các kết cục."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["mc_prob_loss", "mc_var", "mc_methods"])


def mc_prob_loss(a: Analysis) -> Answer:
    rows = {}
    for label, h in [("3 tháng", 63), ("6 tháng", 126), ("1 năm", 252),
                     ("2 năm", 504), ("3 năm", 756), ("5 năm", 1260)]:
        sim = a.simulate(h, 5000, "gbm")
        st = sim.stats()
        rows[label] = {"Xác suất lỗ": pct(st["Xác suất lỗ"], 1),
                       "Lợi suất kỳ vọng": pct(st["Lợi suất kỳ vọng"], 1),
                       "Trung vị giá": f"{st['Trung vị']:,.1f}"}
    tbl = pd.DataFrame(rows).T
    p1 = a.simulate(252, 5000, "gbm").stats()["Xác suất lỗ"]
    p5 = a.simulate(1260, 5000, "gbm").stats()["Xác suất lỗ"]
    txt = (
        f"Xác suất thua lỗ khi nắm giữ {a.ticker} **1 năm là {pct(p1, 1)}**, nhưng nếu giữ "
        f"**5 năm chỉ còn {pct(p5, 1)}**.\n\n"
        "Quy luật này xuất phát từ việc lợi suất kỳ vọng tăng tuyến tính theo thời gian trong "
        "khi độ lệch chuẩn chỉ tăng theo căn bậc hai — thời gian nắm giữ càng dài, phần xu "
        "hướng càng lấn át phần nhiễu. Đây là lập luận định lượng cho đầu tư dài hạn."
    )
    return Answer(txt, tbl, followups=["mg_horizon", "mc_1y"])


def mc_var(a: Analysis) -> Answer:
    sim = a.simulate(252, 10000, "gbm")
    st = sim.stats()
    var, cvar = st["VaR 95% (toàn kỳ)"], st["CVaR 95% (toàn kỳ)"]
    rows = {
        "VaR 95% (1 năm)": pct(var),
        "CVaR 95% (1 năm)": pct(cvar),
        "Kịch bản xấu nhất trong 10.000 lần": pct(st["Lỗ tệ nhất"]),
        "Mất mát với vốn 100 triệu (VaR)": vnd(abs(var) * 100e6),
        "Mất mát với vốn 100 triệu (CVaR)": vnd(abs(cvar) * 100e6),
    }
    txt = (
        f"Theo 10.000 kịch bản mô phỏng cho kỳ hạn 1 năm:\n\n"
        f"- **VaR 95% = {pct(var)}** — có 5% khả năng lỗ nặng hơn mức này.\n"
        f"- **CVaR 95% = {pct(cvar)}** — nếu rơi vào nhóm 5% xấu đó, mức lỗ trung bình.\n"
        f"- Kịch bản tệ nhất trong toàn bộ mô phỏng: **{pct(st['Lỗ tệ nhất'])}**.\n\n"
        f"Với 100 triệu đồng, ngưỡng lỗ 5% xấu nhất tương ứng khoảng "
        f"**{vnd(abs(var) * 100e6)}**. Nhà đầu tư nên tự hỏi: mình có chịu được mức lỗ này "
        "mà không bán tháo hay không — đó mới là thước đo khẩu vị rủi ro thực tế."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["rk_var", "mg_weight"])


def mc_methods(a: Analysis) -> Answer:
    from src.analytics.montecarlo import compare_methods

    tbl = compare_methods(a.last_price, a.log_returns, 252, 5000)
    show = tbl.copy()
    for i in show.index:
        for c in show.columns:
            v = tbl.loc[i, c]
            show.loc[i, c] = pct(v) if any(k in i for k in ("Lợi suất", "Xác suất", "VaR", "CVaR", "Lỗ", "Lãi")) else f"{v:,.2f}"
    txt = (
        "Ba cách sinh kịch bản khác nhau ở giả định về phân phối lợi suất:\n\n"
        "- **GBM** giả định lợi suất log phân phối chuẩn — đơn giản, chuẩn mực lý thuyết, "
        "nhưng đánh giá thấp xác suất biến cố cực đoan.\n"
        "- **Student-t** giữ nguyên cấu trúc nhưng dùng phân phối đuôi dày, phản ánh đúng hơn "
        "thực tế thị trường có những phiên sốc lớn.\n"
        "- **Bootstrap lịch sử** lấy mẫu trực tiếp từ dữ liệu quá khứ, không giả định phân "
        "phối; lấy theo khối 5 phiên để giữ hiệu ứng cụm biến động.\n\n"
        f"Điểm đáng chú ý: độ nhọn thực tế của {a.ticker} là "
        f"**{float(a.returns.kurtosis()):.2f}** (phân phối chuẩn bằng 0), nên GBM cho VaR "
        "lạc quan hơn hai phương pháp còn lại."
    )
    return Answer(txt, show, followups=["mc_var", "th_mc"])


def mc_double(a: Analysis) -> Answer:
    rows = {}
    for label, h in [("1 năm", 252), ("2 năm", 504), ("3 năm", 756), ("5 năm", 1260)]:
        sim = a.simulate(h, 5000, "gbm")
        rows[label] = {
            "Xác suất tăng gấp đôi": pct(sim.prob_above(a.last_price * 2), 1),
            "Xác suất tăng ≥ 50%": pct(sim.prob_above(a.last_price * 1.5), 1),
            "Xác suất mất ≥ 50%": pct(1 - sim.prob_above(a.last_price * 0.5), 1),
        }
    tbl = pd.DataFrame(rows).T
    p3 = a.simulate(756, 5000, "gbm").prob_above(a.last_price * 2)
    txt = (
        f"Xác suất giá {a.ticker} tăng gấp đôi trong **3 năm là {pct(p3, 1)}** theo mô phỏng "
        f"GBM với tham số lịch sử ({pct(a.mu_annual)}/năm, biến động {pct(a.sigma_annual)}).\n\n"
        "Bảng dưới cho thấy xác suất đạt các ngưỡng giá khác nhau theo thời gian nắm giữ. "
        "Cần nhớ mô phỏng giả định lợi suất kỳ vọng tương lai bằng quá khứ — giả định mạnh "
        "và thường không đúng với từng doanh nghiệp cụ thể."
    )
    return Answer(txt, tbl, followups=["mc_1y", "mg_horizon"])


# Nhóm: kỹ thuật
def tc_trend(a: Analysis) -> Answer:
    sig = TA.signal_summary(a.tech)
    last = a.tech.iloc[-1]
    rows = {k: v for k, v in sig.items()}
    rows["Giá đóng cửa"] = f"{last['close']:,.2f}"
    for ma in ("MA20", "MA50", "MA200"):
        if ma in last and pd.notna(last[ma]):
            rows[ma] = f"{last[ma]:,.2f}"
    txt = (
        f"Trạng thái kỹ thuật của {a.ticker} phiên {a.last_date:%d/%m/%Y}:\n\n"
        + "\n".join(f"- **{k}**: {v}" for k, v in sig.items())
        + "\n\nCác chỉ báo kỹ thuật mô tả hành vi giá gần đây, không phải công cụ dự báo. "
          "Nên đọc cùng với phân tích rủi ro và định giá cơ bản."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["tc_rsi", "tc_macd", "tc_levels"])


def tc_rsi(a: Analysis) -> Answer:
    r = float(a.tech["RSI14"].iloc[-1])
    hist = a.tech["RSI14"].dropna()
    zone = "quá mua" if r >= 70 else "quá bán" if r <= 30 else "trung tính"
    rows = {
        "RSI(14) hiện tại": num(r, 1),
        "Trung bình lịch sử": num(float(hist.mean()), 1),
        "Số phiên quá mua (>70)": f"{int((hist > 70).sum())} ({pct((hist > 70).mean(), 1)})",
        "Số phiên quá bán (<30)": f"{int((hist < 30).sum())} ({pct((hist < 30).mean(), 1)})",
    }
    txt = (
        f"**RSI(14) = {num(r, 1)}** — vùng **{zone}**.\n\n"
        "RSI so sánh sức tăng và sức giảm trong 14 phiên gần nhất, dao động 0–100. "
        "Trên 70 thường được coi là quá mua (giá tăng nhanh, dễ điều chỉnh), dưới 30 là "
        "quá bán. Với cổ phiếu có xu hướng tăng dài hạn, RSI có thể duy trì mức cao lâu mà "
        "giá vẫn tiếp tục lên, nên không nên dùng riêng lẻ để ra quyết định."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["tc_trend", "tc_macd"])


def tc_macd(a: Analysis) -> Answer:
    last = a.tech.iloc[-1]
    diff = float(last["macd"] - last["signal"])
    cross = a.tech["hist"].tail(60)
    signs = np.sign(cross)
    last_cross_idx = None
    for i in range(len(signs) - 1, 0, -1):
        if signs.iloc[i] != signs.iloc[i - 1]:
            last_cross_idx = cross.index[i]
            break
    rows = {
        "Đường MACD": num(float(last["macd"]), 3),
        "Đường tín hiệu": num(float(last["signal"]), 3),
        "Histogram": num(float(last["hist"]), 3),
        "Trạng thái": "Động lượng dương" if diff > 0 else "Động lượng âm",
        "Lần giao cắt gần nhất": f"{last_cross_idx:%d/%m/%Y}" if last_cross_idx is not None else "Trên 60 phiên trước",
    }
    txt = (
        f"MACD hiện **{'nằm trên' if diff > 0 else 'nằm dưới'} đường tín hiệu** "
        f"(chênh lệch {num(diff, 3)}), tức động lượng ngắn hạn đang "
        f"**{'tích cực' if diff > 0 else 'tiêu cực'}**.\n\n"
        "MACD là hiệu của hai đường trung bình động hàm mũ 12 và 26 phiên; đường tín hiệu là "
        "EMA 9 phiên của chính MACD. Giao cắt lên thường được xem là tín hiệu mua, giao cắt "
        "xuống là tín hiệu bán — nhưng độ trễ khá lớn và hay nhiễu trong thị trường đi ngang."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["tc_trend", "tc_rsi"])


def tc_levels(a: Analysis) -> Answer:
    recent = a.prices.tail(126)
    last = a.last_price
    hi, lo = float(recent.max()), float(recent.min())
    ma20 = float(a.tech["MA20"].iloc[-1])
    ma50 = float(a.tech["MA50"].iloc[-1])
    ma200 = float(a.tech["MA200"].iloc[-1])
    levels = {
        "Đỉnh 6 tháng": hi, "Đáy 6 tháng": lo,
        "MA20": ma20, "MA50": ma50, "MA200": ma200,
        "Đỉnh lịch sử": float(a.prices.max()),
    }
    above = {k: v for k, v in levels.items() if v > last}
    below = {k: v for k, v in levels.items() if v <= last}
    tbl = pd.DataFrame({
        "Mức giá": [f"{v:,.2f}" for v in levels.values()],
        "Vị trí": ["Kháng cự (trên giá)" if v > last else "Hỗ trợ (dưới giá)" for v in levels.values()],
        "Cách giá hiện tại": [pct(v / last - 1) for v in levels.values()],
    }, index=list(levels.keys()))
    nearest_res = min(above.items(), key=lambda kv: kv[1]) if above else None
    nearest_sup = max(below.items(), key=lambda kv: kv[1]) if below else None
    txt = f"Giá hiện tại **{last:,.2f} nghìn đồng**.\n\n"
    if nearest_sup:
        txt += (f"- Hỗ trợ gần nhất: **{nearest_sup[0]} = {nearest_sup[1]:,.2f}** "
                f"({pct(nearest_sup[1] / last - 1)})\n")
    if nearest_res:
        txt += (f"- Kháng cự gần nhất: **{nearest_res[0]} = {nearest_res[1]:,.2f}** "
                f"({pct(nearest_res[1] / last - 1)})\n")
    txt += ("\nCác mức này lấy từ đỉnh/đáy 6 tháng và các đường trung bình động — nơi thường "
            "tập trung lệnh mua bán. Đây là tham chiếu thống kê, không phải ngưỡng chắc chắn.")
    return Answer(txt, tbl, followups=["tc_trend", "mg_shares"])


# Nhóm: quản lý đầu tư
def mg_weight(a: Analysis) -> Answer:
    rows = {}
    for A_ in (2.0, 3.0, 4.0, 6.0):
        o = a.allocation(A_)
        label = {2.0: "Ưa rủi ro (A=2)", 3.0: "Trung bình (A=3)",
                 4.0: "Thận trọng (A=4)", 6.0: "Rất thận trọng (A=6)"}[A_]
        rows[label] = {
            "Tỷ trọng cổ phiếu": pct(o.y_optimal, 0),
            "Tỷ trọng tiền gửi": pct(1 - o.y_optimal, 0),
            "Lợi suất kỳ vọng": pct(o.expected_return),
            "Độ biến động": pct(o.volatility),
        }
    tbl = pd.DataFrame(rows).T
    o3 = a.allocation(3.0)
    txt = (
        f"Theo mô hình hữu dụng trung bình – phương sai, tỷ trọng tối ưu vào {a.ticker} là "
        f"**y\\* = (E[R] − Rf) / (A × σ²)**.\n\n"
        f"Với nhà đầu tư có mức ngại rủi ro trung bình (A = 3): "
        f"**{pct(o3.y_optimal, 0)} vốn vào cổ phiếu, {pct(1 - o3.y_optimal, 0)} gửi tiết kiệm** — "
        f"danh mục này kỳ vọng {pct(o3.expected_return)}/năm với biến động {pct(o3.volatility)}.\n\n"
        "Lưu ý quan trọng: đây là bài toán một tài sản rủi ro duy nhất. Nếu phân bổ vào danh "
        "mục nhiều cổ phiếu, phần rủi ro riêng lẻ được triệt tiêu nên tỷ trọng cổ phiếu hợp "
        "lý sẽ cao hơn con số trên."
    )
    return Answer(txt, tbl, followups=["mg_kelly", "mg_mix", "mg_shares"])


def mg_shares(a: Analysis) -> Answer:
    capital = 100e6
    price_vnd = a.last_price * 1000
    atr = float(a.tech["ATR14"].iloc[-1]) * 1000 if "ATR14" in a.tech else np.nan
    stop = 2 * atr
    ps = AL.position_size_by_risk(capital, price_vnd, stop, 0.02)
    o3 = a.allocation(3.0)
    by_weight = int(capital * o3.y_optimal / price_vnd / 100) * 100
    rows = {
        "Vốn giả định": vnd(capital),
        "Giá hiện tại": f"{price_vnd:,.0f} đồng/cổ phiếu",
        "ATR(14)": f"{atr:,.0f} đồng",
        "Cắt lỗ đề xuất (2×ATR)": f"{price_vnd - stop:,.0f} đồng ({pct(-stop / price_vnd)})",
        "Số cổ phiếu theo quy tắc rủi ro 2%": f"{ps['shares']:,} cổ phiếu",
        "Giá trị vị thế": vnd(ps["value"]),
        "Số cổ phiếu theo tỷ trọng tối ưu": f"{by_weight:,} cổ phiếu",
        "Giá trị theo tỷ trọng tối ưu": vnd(by_weight * price_vnd),
    }
    txt = (
        f"Với **100 triệu đồng** và giá {price_vnd:,.0f} đồng/cổ phiếu, có hai cách xác định "
        "quy mô vị thế:\n\n"
        f"**1. Theo mức rủi ro chấp nhận (2% vốn mỗi lệnh)** — đặt cắt lỗ tại "
        f"{price_vnd - stop:,.0f} đồng (cách giá {pct(stop / price_vnd)}, bằng 2×ATR). "
        f"Nếu chạm cắt lỗ thì mất đúng 2 triệu, tức mua **{ps['shares']:,} cổ phiếu** "
        f"(≈ {vnd(ps['value'])}).\n\n"
        f"**2. Theo tỷ trọng tối ưu từ lý thuyết danh mục** ({pct(o3.y_optimal, 0)} vốn) — "
        f"mua **{by_weight:,} cổ phiếu** (≈ {vnd(by_weight * price_vnd)}), phần còn lại gửi "
        "tiết kiệm.\n\n"
        "Cách 1 kiểm soát khoản lỗ tối đa mỗi lệnh, cách 2 tối ưu quan hệ lợi nhuận – rủi ro "
        "của toàn danh mục. Thực tế nên lấy giá trị nhỏ hơn giữa hai cách."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["mg_weight", "tc_levels"])


def mg_kelly(a: Analysis) -> Answer:
    f = AL.kelly_fraction(a.mu_arithmetic - a.rf_annual, a.sigma_annual)
    rows = {
        "Lợi suất kỳ vọng (trung bình số học)": pct(a.mu_arithmetic),
        "Lợi suất vượt trội kỳ vọng": pct(a.mu_arithmetic - a.rf_annual),
        "Phương sai (năm)": num(a.sigma_annual ** 2, 4),
        "Tỷ trọng Kelly đầy đủ": pct(f, 0),
        "Nửa Kelly (thực dụng)": pct(f / 2, 0),
        "Tỷ trọng tối ưu theo A=3": pct(a.allocation(3.0).y_optimal, 0),
    }
    txt = (
        f"Tiêu chí Kelly cho tài sản liên tục: **f\\* = (μ − Rf) / σ²** = "
        f"({pct(a.mu_arithmetic)} − {pct(a.rf_annual)}) / {num(a.sigma_annual ** 2, 3)} = "
        f"**{pct(f, 0)}**.\n\n"
        "Kelly tối đa hoá tốc độ tăng trưởng dài hạn của vốn, nhưng nổi tiếng là **rất hung "
        "hăng**: sai số nhỏ trong ước lượng μ khiến tỷ trọng lệch rất xa, và đường vốn biến "
        f"động dữ dội. Thực hành phổ biến là dùng **nửa Kelly ≈ {pct(f / 2, 0)}**, "
        "hy sinh chút tăng trưởng để đổi lấy mức dao động dễ chịu hơn.\n\n"
        f"Con số này cũng cần so với tỷ trọng {pct(a.allocation(3.0).y_optimal, 0)} từ mô hình "
        "hữu dụng trung bình – phương sai; Kelly tương đương trường hợp mức ngại rủi ro A = 1."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["mg_weight", "mg_mix"])


def mg_dca(a: Analysis) -> Answer:
    df = AL.dca_vs_lumpsum(a.prices, 10e6)
    dca_final = float(df["DCA hàng tháng"].iloc[-1])
    ls_final = float(df["Mua một lần"].iloc[-1])
    invested = float(df["Vốn đã giải ngân (DCA)"].iloc[-1])
    rows = {
        "Tổng vốn giải ngân": vnd(invested),
        "Giá trị cuối kỳ — DCA": vnd(dca_final),
        "Giá trị cuối kỳ — mua một lần": vnd(ls_final),
        "Lợi nhuận DCA": pct(dca_final / invested - 1),
        "Lợi nhuận mua một lần": pct(ls_final / invested - 1),
    }
    better = "mua một lần" if ls_final > dca_final else "DCA"
    txt = (
        f"Thử nghiệm trên toàn bộ lịch sử {a.ticker} với 10 triệu đồng mỗi tháng "
        f"(tổng {vnd(invested)}), so với việc dồn toàn bộ số tiền đó mua ngay phiên đầu:\n\n"
        f"- DCA hàng tháng: **{vnd(dca_final)}**\n"
        f"- Mua một lần: **{vnd(ls_final)}**\n\n"
        f"Trong mẫu này **{better}** cho kết quả cao hơn. Kết quả đó không bất ngờ: với tài "
        "sản có xu hướng tăng dài hạn, dồn vốn sớm thường thắng về giá trị cuối kỳ, nhưng "
        "phải chịu rủi ro thời điểm rất lớn — nếu mua trúng đỉnh 2007 thì phải chờ nhiều năm.\n\n"
        "DCA đổi một phần lợi nhuận kỳ vọng để lấy sự ổn định tâm lý và giảm rủi ro chọn sai "
        "thời điểm. Với nhà đầu tư có thu nhập đều hàng tháng, DCA thường là lựa chọn thực tế hơn."
    )
    return Answer(txt, pd.DataFrame({"Giá trị": rows}), followups=["mg_horizon", "px_invest_100"])


def mg_horizon(a: Analysis) -> Answer:
    rows = {}
    for label, d in [("1 tháng", 21), ("3 tháng", 63), ("6 tháng", 126), ("1 năm", 252),
                     ("2 năm", 504), ("3 năm", 756), ("5 năm", 1260)]:
        if len(a.prices) <= d:
            continue
        fwd = a.prices.shift(-d) / a.prices - 1
        fwd = fwd.dropna()
        rows[label] = {
            "Xác suất có lãi": pct(float((fwd > 0).mean()), 1),
            "Lợi suất trung bình": pct(float(fwd.mean())),
            "Trường hợp xấu nhất": pct(float(fwd.min())),
            "Trường hợp tốt nhất": pct(float(fwd.max())),
        }
    tbl = pd.DataFrame(rows).T
    one_y = (a.prices.shift(-252) / a.prices - 1).dropna()
    five_y = (a.prices.shift(-1260) / a.prices - 1).dropna()
    txt = (
        f"Thống kê trên toàn bộ lịch sử thực tế của {a.ticker} (không phải mô phỏng): "
        f"nếu mua ngẫu nhiên một phiên rồi giữ **1 năm**, xác suất có lãi là "
        f"**{pct(float((one_y > 0).mean()), 1)}**; giữ **5 năm** thì tăng lên "
        f"**{pct(float((five_y > 0).mean()), 1)}**.\n\n"
        f"Trường hợp xấu nhất khi giữ 1 năm là **{pct(float(one_y.min()))}** — rơi vào giai "
        "đoạn khủng hoảng 2008. Bảng dưới cho thấy thời gian nắm giữ càng dài, phân phối kết "
        "quả càng dịch về phía có lãi."
    )
    return Answer(txt, tbl, followups=["mc_prob_loss", "mg_dca"])


def mg_mix(a: Analysis) -> Answer:
    navs, table = AL.compare_allocations(a.returns, (0.0, 0.25, 0.5, 0.75, 1.0), a.rf_annual)
    show = table.copy()
    for i in show.index:
        show.loc[i, "Lợi suất năm"] = pct(table.loc[i, "Lợi suất năm"])
        show.loc[i, "Độ biến động"] = pct(table.loc[i, "Độ biến động"])
        show.loc[i, "Sharpe"] = num(table.loc[i, "Sharpe"])
        show.loc[i, "Sụt giảm tối đa"] = pct(table.loc[i, "Sụt giảm tối đa"])
        show.loc[i, "NAV cuối kỳ"] = f"{table.loc[i, 'NAV cuối kỳ']:,.2f} lần"
    txt = (
        "Danh mục pha trộn giữa cổ phiếu và tiền gửi, tái cân bằng hằng tháng và có tính phí "
        "giao dịch 0,15% mỗi chiều:\n\n"
        "Điểm đáng chú ý là **Sharpe gần như không đổi** giữa các mức phân bổ — đúng như lý "
        "thuyết: mọi danh mục nằm trên cùng một đường phân bổ vốn đều có cùng tỷ số "
        "lợi nhuận/rủi ro. Thay đổi tỷ trọng chỉ dịch chuyển vị trí trên đường đó, tức chọn "
        "mức rủi ro phù hợp với bản thân, chứ không tạo ra hiệu quả tốt hơn.\n\n"
        "Khác biệt lớn nằm ở **mức sụt giảm tối đa**: danh mục 100% cổ phiếu từng mất "
        f"{pct(table.loc['100% cổ phiếu', 'Sụt giảm tối đa'])}, trong khi danh mục 50% chỉ "
        f"mất {pct(table.loc['50% cổ phiếu', 'Sụt giảm tối đa'])} — khác biệt quyết định việc "
        "nhà đầu tư có trụ được qua khủng hoảng hay không."
    )
    return Answer(txt, show, followups=["mg_weight", "rk_dd"])


# Nhóm: lý thuyết
def th_capm(a: Analysis) -> Answer:
    txt = (
        "**CAPM (Capital Asset Pricing Model)** — mô hình định giá tài sản vốn do Sharpe, "
        "Lintner và Mossin phát triển những năm 1960, dựa trên lý thuyết danh mục của "
        "Markowitz.\n\n"
        "**Công thức:** E(Ri) = Rf + βi × [E(Rm) − Rf]\n\n"
        "- **Rf** — lãi suất phi rủi ro (trái phiếu chính phủ)\n"
        "- **E(Rm) − Rf** — phần bù rủi ro thị trường\n"
        "- **βi** — độ nhạy của tài sản i với biến động thị trường\n\n"
        "**Ý tưởng cốt lõi:** thị trường chỉ trả công cho phần rủi ro *không thể tránh được* "
        "bằng đa dạng hoá (rủi ro hệ thống, đo bằng beta). Rủi ro riêng của từng doanh nghiệp "
        "không được đền bù, vì nhà đầu tư có thể loại bỏ nó miễn phí bằng cách nắm nhiều mã.\n\n"
        "**Giả định chính:** nhà đầu tư duy lý và ngại rủi ro, thị trường không có chi phí "
        "giao dịch và thuế, mọi người có cùng kỳ vọng, vay và cho vay ở cùng lãi suất Rf.\n\n"
        "**Hạn chế:** thực nghiệm cho thấy beta không giải thích đủ chênh lệch lợi suất giữa "
        "các cổ phiếu; điều này dẫn tới các mô hình đa nhân tố như Fama–French và APT.\n\n"
        f"Áp dụng cho {a.ticker}: β = {num(a.capm.beta)}, lợi suất kỳ vọng "
        f"{pct(a.capm.expected_return)}/năm."
    )
    return Answer(txt, followups=["capm_beta", "th_apt", "th_beta"])


def th_beta(a: Analysis) -> Answer:
    txt = (
        "**Beta (β)** đo mức độ biến động của một cổ phiếu so với toàn thị trường.\n\n"
        "**Công thức:** β = Cov(Ri, Rm) / Var(Rm) — hệ số góc khi hồi quy lợi suất cổ phiếu "
        "theo lợi suất thị trường.\n\n"
        "**Cách đọc:**\n"
        "- β = 1: biến động cùng nhịp thị trường\n"
        "- β > 1: khuếch đại biến động (cổ phiếu tấn công) — lãi nhiều hơn khi thị trường "
        "lên, lỗ nặng hơn khi thị trường xuống\n"
        "- 0 < β < 1: ít nhạy hơn thị trường (cổ phiếu phòng thủ)\n"
        "- β < 0: ngược chiều thị trường (rất hiếm, ví dụ vàng trong một số giai đoạn)\n\n"
        f"**{a.ticker} có β = {num(a.capm.beta)}**, nghĩa là khi VNINDEX thay đổi 1% thì cổ "
        f"phiếu thay đổi trung bình {num(a.capm.beta)}% cùng chiều.\n\n"
        "Lưu ý: beta chỉ đo *quan hệ tuyến tính với thị trường*, không đo tổng rủi ro. Một cổ "
        "phiếu có beta thấp vẫn có thể rất rủi ro nếu phần biến động riêng lớn."
    )
    return Answer(txt, followups=["capm_beta", "capm_rolling", "rk_split"])


def th_apt(a: Analysis) -> Answer:
    txt = (
        "**APT (Arbitrage Pricing Theory)** do Stephen Ross đề xuất năm 1976, là mô hình định "
        "giá đa nhân tố thay thế cho CAPM.\n\n"
        "**Công thức:** E(Ri) = Rf + b₁λ₁ + b₂λ₂ + … + bₖλₖ\n\n"
        "trong đó bⱼ là độ nhạy của tài sản với nhân tố j, còn λⱼ là phần bù rủi ro của nhân "
        "tố đó.\n\n"
        "**Khác biệt so với CAPM:**\n\n"
        "| | CAPM | APT |\n"
        "|---|---|---|\n"
        "| Số nhân tố | 1 (thị trường) | Nhiều |\n"
        "| Giả định | Danh mục thị trường hiệu quả, nhà đầu tư đồng nhất | Chỉ cần không tồn "
        "tại cơ hội kinh doanh chênh lệch |\n"
        "| Nhân tố | Xác định rõ | Lý thuyết không chỉ rõ, phải chọn thực nghiệm |\n\n"
        "**Nguyên lý:** nếu hai tài sản có cùng độ nhạy với mọi nhân tố rủi ro nhưng lợi suất "
        "kỳ vọng khác nhau, nhà đầu tư sẽ mua rẻ bán đắt cho tới khi chênh lệch biến mất.\n\n"
        "**Nhân tố thường dùng** (theo Chen–Roll–Ross): tăng trưởng sản xuất công nghiệp, lạm "
        "phát bất ngờ, chênh lệch lãi suất, phần bù rủi ro tín dụng. Trong đề tài này dùng "
        "nhóm nhân tố quan sát được theo ngày: thị trường, tỷ giá, giá dầu, giá vàng và "
        "chứng khoán Mỹ.\n\n"
        f"Kết quả với {a.ticker}: R² của APT là {pct(a.apt.adj_r_squared, 1)} so với "
        f"{pct(a.capm.r_squared, 1)} của CAPM."
    )
    return Answer(txt, followups=["apt_factors", "apt_vs_capm", "th_capm"])


def th_sharpe(a: Analysis) -> Answer:
    s = M.sharpe_ratio(a.returns, a.rf_annual)
    txt = (
        "**Tỷ số Sharpe** (William Sharpe, 1966) đo lợi suất vượt trội trên mỗi đơn vị rủi ro "
        "tổng thể.\n\n"
        "**Công thức:** Sharpe = (Rp − Rf) / σp\n\n"
        "**Thang tham chiếu thông dụng:**\n"
        "- < 0: kém hơn gửi tiết kiệm\n"
        "- 0 – 1: bình thường\n"
        "- 1 – 2: tốt\n"
        "- 2 – 3: rất tốt\n"
        "- > 3: xuất sắc (hiếm, cần kiểm tra kỹ xem có sai sót phương pháp không)\n\n"
        f"**{a.ticker} có Sharpe = {num(s)}** tính trên toàn bộ lịch sử.\n\n"
        "**Hạn chế:** Sharpe phạt cả biến động tăng lẫn giảm, trong khi nhà đầu tư chỉ ngại "
        "biến động giảm — đó là lý do có **tỷ số Sortino** chỉ dùng độ lệch chuẩn của phần "
        "âm. Sharpe cũng giả định lợi suất phân phối chuẩn, nên đánh giá sai với chiến lược "
        "có đuôi dày."
    )
    return Answer(txt, followups=["rk_sharpe", "px_vs_index"])


def th_var(a: Analysis) -> Answer:
    v = M.var_historical(a.returns, 0.95)
    c = M.cvar_historical(a.returns, 0.95)
    txt = (
        "**VaR (Value at Risk — giá trị chịu rủi ro)** trả lời câu hỏi: *với độ tin cậy X%, "
        "khoản lỗ tối đa trong một khoảng thời gian là bao nhiêu?*\n\n"
        f"Ví dụ VaR 95% một ngày của {a.ticker} = **{pct(v)}**: 95% số phiên khoản lỗ không "
        f"vượt quá {pct(abs(v))}.\n\n"
        "**Ba cách tính:**\n"
        "1. *Lịch sử* — lấy phân vị trực tiếp từ dữ liệu quá khứ, không giả định phân phối.\n"
        "2. *Tham số* — giả định phân phối chuẩn: VaR = μ + z·σ.\n"
        "3. *Monte Carlo* — mô phỏng hàng nghìn kịch bản rồi lấy phân vị.\n\n"
        "**Điểm yếu lớn nhất của VaR:** nó không nói gì về mức độ tệ khi đã vượt ngưỡng. Hai "
        "danh mục có cùng VaR nhưng một cái lỗ 6%, cái kia lỗ 60% trong tình huống xấu.\n\n"
        f"**CVaR (Conditional VaR / Expected Shortfall)** khắc phục điều đó: là **lỗ trung "
        f"bình trong nhóm kịch bản xấu nhất**. Với {a.ticker}, CVaR 95% = **{pct(c)}**. "
        "CVaR là thước đo rủi ro nhất quán về mặt toán học và được Basel III khuyến nghị "
        "thay thế VaR."
    )
    return Answer(txt, followups=["rk_var", "mc_var"])


def th_mc(a: Analysis) -> Answer:
    txt = (
        "**Mô phỏng Monte Carlo** là kỹ thuật dùng số ngẫu nhiên để mô tả phân phối của những "
        "kết cục có thể xảy ra, thay vì đưa ra một con số dự báo duy nhất. Tên gọi lấy từ sòng "
        "bạc Monte Carlo, do von Neumann và Ulam đặt khi làm dự án Manhattan.\n\n"
        "**Các bước áp dụng cho giá cổ phiếu:**\n\n"
        "1. Ước lượng tham số từ lịch sử: lợi suất kỳ vọng μ và độ biến động σ.\n"
        "2. Chọn mô hình sinh dữ liệu, phổ biến nhất là chuyển động Brown hình học:\n"
        "   **S(t+1) = S(t) × exp[(μ − σ²/2)Δt + σ√Δt × ε]**, với ε là biến ngẫu nhiên chuẩn.\n"
        "3. Sinh hàng nghìn quỹ đạo giá độc lập.\n"
        "4. Thống kê phân phối kết quả: trung vị, khoảng tin cậy, xác suất lỗ, VaR, CVaR.\n\n"
        "**Vì sao hữu ích:** nhiều bài toán tài chính không có lời giải giải tích (quyền chọn "
        "phức tạp, danh mục nhiều tài sản, rút vốn theo lộ trình). Monte Carlo giải được bằng "
        "cách mô phỏng trực tiếp.\n\n"
        "**Cần thận trọng:** kết quả chỉ tốt bằng giả định đầu vào. Dùng μ và σ quá khứ tức là "
        "ngầm cho rằng tương lai giống quá khứ — giả định này thường sai với từng cổ phiếu "
        "riêng lẻ."
    )
    return Answer(txt, followups=["mc_1y", "mc_methods", "mc_var"])


def th_dd(a: Analysis) -> Answer:
    dd = M.drawdown_series(a.returns)
    txt = (
        "**Maximum Drawdown (sụt giảm tối đa)** là mức lỗ lớn nhất tính từ một đỉnh tới đáy "
        "kế tiếp của đường giá trị tài khoản.\n\n"
        "**Công thức:** MDD = min[ NAV(t) / max(NAV(0..t)) − 1 ]\n\n"
        "**Vì sao quan trọng hơn độ lệch chuẩn với nhà đầu tư thực tế:**\n"
        "- Nó là con số nhà đầu tư thực sự cảm nhận và là nguyên nhân chính khiến người ta "
        "bán tháo ở đáy.\n"
        "- Nó thể hiện được chuỗi biến động liên tiếp cùng chiều, điều mà độ lệch chuẩn bỏ qua.\n"
        "- Cần lãi lớn hơn nhiều để hoà vốn: lỗ 50% phải lãi 100% mới về mức cũ; lỗ 80% cần "
        "lãi 400%.\n\n"
        f"**{a.ticker} có MDD = {pct(dd.min())}** trong lịch sử — muốn phục hồi cần mức tăng "
        f"**{pct(1 / (1 + dd.min()) - 1)}**.\n\n"
        "**Tỷ số Calmar** = lợi suất năm / |MDD| là cách kết hợp hai chỉ tiêu này thành một "
        f"thước đo hiệu quả: {a.ticker} đạt {num(M.calmar_ratio(a.returns))}."
    )
    return Answer(txt, followups=["rk_dd", "rk_crisis"])


# Bảng điều phối
HANDLERS = {
    "co_profile": co_profile, "co_listing": co_listing, "co_marketcap": co_marketcap,
    "co_shareholders": co_shareholders, "co_foreign": co_foreign,
    "co_valuation": co_valuation, "co_growth": co_growth,
    "px_now": px_now, "px_change": px_change, "px_cagr": px_cagr,
    "px_vs_index": px_vs_index, "px_best_worst": px_best_worst,
    "px_invest_100": px_invest_100, "px_monthly": px_monthly,
    "rk_vol": rk_vol, "rk_var": rk_var, "rk_dd": rk_dd, "rk_sharpe": rk_sharpe,
    "rk_split": rk_split, "rk_crisis": rk_crisis,
    "capm_beta": capm_beta, "capm_alpha": capm_alpha, "capm_expected": capm_expected,
    "capm_fair": capm_fair, "capm_r2": capm_r2, "capm_rolling": capm_rolling,
    "apt_factors": apt_factors, "apt_vs_capm": apt_vs_capm,
    "apt_expected": apt_expected, "apt_usd": apt_usd,
    "mc_1y": mc_1y, "mc_prob_loss": mc_prob_loss, "mc_var": mc_var,
    "mc_methods": mc_methods, "mc_double": mc_double,
    "tc_trend": tc_trend, "tc_rsi": tc_rsi, "tc_macd": tc_macd, "tc_levels": tc_levels,
    "mg_weight": mg_weight, "mg_shares": mg_shares, "mg_kelly": mg_kelly,
    "mg_dca": mg_dca, "mg_horizon": mg_horizon, "mg_mix": mg_mix,
    "th_capm": th_capm, "th_beta": th_beta, "th_apt": th_apt, "th_sharpe": th_sharpe,
    "th_var": th_var, "th_mc": th_mc, "th_dd": th_dd,
}


def answer(qid: str, a: Analysis) -> Answer:
    fn = HANDLERS.get(qid)
    if fn is None:
        return Answer("Câu hỏi này chưa được hỗ trợ.")
    try:
        return fn(a)
    except Exception as exc:                     # không để lỗi làm sập giao diện
        return Answer(
            f"Không tính được câu trả lời do thiếu dữ liệu.\n\n`{type(exc).__name__}: {exc}`"
        )


# Khớp câu hỏi tự do
def _normalize(s: str) -> str:
    """Bỏ dấu tiếng Việt, đưa về chữ thường để so khớp."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("đ", "d")
    return re.sub(r"[^a-z0-9\s]", " ", s)


_STOP = {"la", "gi", "cua", "co", "the", "nao", "bao", "nhieu", "va", "cho", "toi",
         "voi", "trong", "thi", "khong", "duoc", "mot", "nhu", "o", "ra", "sao", "hay"}


# Ngưỡng điểm tối thiểu để coi là khớp. Đặt đủ cao để chatbot thà nói "chưa hiểu"
# còn hơn trả lời tự tin nhưng lạc đề — sai lệch nguy hiểm hơn im lặng.
MATCH_THRESHOLD = 0.34


def other_ticker_mentioned(text: str, ticker: str) -> str | None:
    """Phát hiện người dùng hỏi về mã chứng khoán khác mã đang phân tích."""
    words = re.findall(r"\b[A-Za-z]{3}\b", text)
    for w in words:
        up = w.upper()
        if up != ticker.upper() and up == w and w.isalpha() and w.isupper():
            return up
    return None


def match_questions(text: str, ticker: str, top_k: int = 5) -> list[tuple[dict, float]]:
    """Tìm các câu hỏi gợi ý gần nhất với câu người dùng tự gõ."""
    q_norm = _normalize(text)
    tokens = [t for t in q_norm.split() if t and t not in _STOP]
    if not tokens:
        return []
    scored = []
    for item in QUESTIONS:
        hay = _normalize(item["kw"] + " " + render(item, ticker))
        hay_tokens = set(hay.split())
        hits = sum(1 for t in tokens if t in hay_tokens)
        if not hits:
            continue
        phrase_bonus = 0.1 if any(len(t) > 4 and t in hay for t in tokens) else 0.0
        score = hits / len(tokens) + phrase_bonus
        if score >= MATCH_THRESHOLD:
            scored.append((item, score))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored[:top_k]
