"""Thư viện biểu đồ Plotly dùng chung cho dashboard, chatbot và báo cáo."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from src.analytics.eda import qq_data
from src.config import COLOR_ACCENT, COLOR_DOWN, COLOR_MUTED, COLOR_PRIMARY, COLOR_UP

# Tiêu đề canh trái cho mọi biểu đồ (đặt trong template để áp dụng đồng nhất)
_TEMPLATE = pio.templates["plotly_white"]
_TEMPLATE.layout.title.x = 0
_TEMPLATE.layout.title.xanchor = "left"
_TEMPLATE.layout.title.font.size = 15

LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=60, r=30, t=60, b=70),
    hovermode="x unified",
    # Chú giải đặt phía dưới để không đè lên tiêu đề biểu đồ
    legend=dict(orientation="h", yanchor="top", y=-0.14, x=0),
    font=dict(family="Segoe UI, Arial", size=13),
)


def _fig(title: str = "", **kw) -> go.Figure:
    f = go.Figure()
    f.update_layout(title=title, **LAYOUT, **kw)
    return f


# Giá và kỹ thuật
def candlestick(tech: pd.DataFrame, ticker: str, show_bb: bool = True) -> go.Figure:
    f = _fig(f"Diễn biến giá {ticker}")
    f.add_trace(go.Candlestick(
        x=tech.index, open=tech["open"], high=tech["high"], low=tech["low"],
        close=tech["close"], name=ticker,
        increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN,
    ))
    for col, color, width in (("MA20", COLOR_PRIMARY, 1.3), ("MA50", COLOR_ACCENT, 1.3),
                              ("MA200", COLOR_MUTED, 1.6)):
        if col in tech:
            f.add_trace(go.Scatter(x=tech.index, y=tech[col], name=col,
                                   line=dict(color=color, width=width)))
    if show_bb and "bb_upper" in tech:
        f.add_trace(go.Scatter(x=tech.index, y=tech["bb_upper"], name="Bollinger trên",
                               line=dict(color="rgba(100,116,139,0.35)", width=1)))
        f.add_trace(go.Scatter(x=tech.index, y=tech["bb_lower"], name="Bollinger dưới",
                               line=dict(color="rgba(100,116,139,0.35)", width=1),
                               fill="tonexty", fillcolor="rgba(100,116,139,0.08)"))
    f.update_layout(xaxis_rangeslider_visible=False, yaxis_title="Giá (nghìn đồng)")
    return f


def volume_chart(tech: pd.DataFrame) -> go.Figure:
    colors = np.where(tech["close"] >= tech["open"], COLOR_UP, COLOR_DOWN)
    f = _fig("Khối lượng giao dịch (thang logarit)")
    f.add_trace(go.Bar(x=tech.index, y=tech["volume"], marker_color=colors, name="Khối lượng"))
    # Khối lượng tăng hàng trăm lần qua hai thập kỷ nên trục tuyến tính sẽ ép
    # toàn bộ giai đoạn đầu thành một đường sát 0.
    f.update_yaxes(type="log")
    if "VOL_MA20" in tech:
        f.add_trace(go.Scatter(x=tech.index, y=tech["VOL_MA20"], name="TB 20 phiên",
                               line=dict(color=COLOR_PRIMARY, width=1.5)))
    f.update_layout(yaxis_title="Cổ phiếu")
    return f


def rsi_macd(tech: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    f1 = _fig("RSI (14)")
    f1.add_trace(go.Scatter(x=tech.index, y=tech["RSI14"], name="RSI",
                            line=dict(color=COLOR_PRIMARY)))
    f1.add_hline(y=70, line_dash="dash", line_color=COLOR_DOWN, annotation_text="Quá mua 70")
    f1.add_hline(y=30, line_dash="dash", line_color=COLOR_UP, annotation_text="Quá bán 30")
    f1.update_layout(yaxis_range=[0, 100])

    f2 = _fig("MACD (12, 26, 9)")
    f2.add_trace(go.Bar(x=tech.index, y=tech["hist"], name="Histogram",
                        marker_color=np.where(tech["hist"] >= 0, COLOR_UP, COLOR_DOWN)))
    f2.add_trace(go.Scatter(x=tech.index, y=tech["macd"], name="MACD",
                            line=dict(color=COLOR_PRIMARY)))
    f2.add_trace(go.Scatter(x=tech.index, y=tech["signal"], name="Đường tín hiệu",
                            line=dict(color=COLOR_ACCENT)))
    return f1, f2


# Hiệu quả & rủi ro
def growth_comparison(stock_ret: pd.Series, bench_ret: pd.Series,
                      ticker: str, bench_name: str = "VNINDEX") -> go.Figure:
    f = _fig("Tăng trưởng 1 đồng vốn")
    f.add_trace(go.Scatter(x=stock_ret.index, y=(1 + stock_ret).cumprod(),
                           name=ticker, line=dict(color=COLOR_PRIMARY, width=2)))
    f.add_trace(go.Scatter(x=bench_ret.index, y=(1 + bench_ret).cumprod(),
                           name=bench_name, line=dict(color=COLOR_MUTED, width=2)))
    f.update_layout(yaxis_title="Lần vốn ban đầu")
    return f


def drawdown_chart(dd: pd.Series, ticker: str) -> go.Figure:
    f = _fig(f"Sụt giảm từ đỉnh — {ticker}")
    f.add_trace(go.Scatter(x=dd.index, y=dd * 100, name="Drawdown",
                           fill="tozeroy", line=dict(color=COLOR_DOWN, width=1)))
    f.update_layout(yaxis_title="%")
    return f


def return_distribution(returns: pd.Series, var: float, cvar: float,
                        level: float = 0.95) -> go.Figure:
    """Phân phối lợi suất kèm hai ngưỡng rủi ro ở mức tin cậy ``level``."""
    f = _fig("Phân phối lợi suất theo phiên")
    f.add_trace(go.Histogram(x=returns * 100, nbinsx=90, name="Tần suất",
                             marker_color=COLOR_PRIMARY, opacity=0.75))
    f.add_vline(x=var * 100, line_dash="dash", line_color=COLOR_ACCENT,
                annotation_text=f"VaR {level:.0%} = {var * 100:.2f}%",
                annotation_position="top left")
    f.add_vline(x=cvar * 100, line_dash="dash", line_color=COLOR_DOWN,
                annotation_text=f"CVaR {level:.0%} = {cvar * 100:.2f}%",
                annotation_position="bottom left")
    f.update_layout(xaxis_title="Lợi suất (%)", yaxis_title="Số phiên", hovermode="x")
    return f


def monthly_heatmap(table: pd.DataFrame) -> go.Figure:
    data = table.drop(columns=["Cả năm"], errors="ignore") * 100
    # Tháng chưa có dữ liệu để trống thay vì in chữ NaN lên ô
    labels = np.where(np.isnan(data.values), "", np.round(data.values, 1).astype(object))
    f = _fig("Lợi suất theo tháng (%)")
    f.add_trace(go.Heatmap(
        z=data.values, x=list(data.columns), y=data.index.astype(str),
        colorscale=[[0, COLOR_DOWN], [0.5, "#ffffff"], [1, COLOR_UP]], zmid=0,
        text=labels, texttemplate="%{text}", hoverongaps=False,
        colorbar=dict(title="%"),
    ))
    f.update_layout(hovermode="closest")
    return f


# CAPM / APT
def capm_scatter(capm_res, ticker: str) -> go.Figure:
    d = capm_res.data
    f = _fig(f"Hồi quy CAPM: {ticker} theo VNINDEX")
    f.add_trace(go.Scatter(x=d["ex_market"] * 100, y=d["ex_stock"] * 100, mode="markers",
                           name="Quan sát ngày",
                           marker=dict(size=4, color=COLOR_PRIMARY, opacity=0.35)))
    xs = np.linspace(d["ex_market"].min(), d["ex_market"].max(), 50)
    ys = capm_res.alpha_daily + capm_res.beta * xs
    f.add_trace(go.Scatter(x=xs * 100, y=ys * 100, name=f"β = {capm_res.beta:.2f}",
                           line=dict(color=COLOR_DOWN, width=2.5)))
    f.update_layout(xaxis_title="Lợi suất vượt trội thị trường (%)",
                    yaxis_title=f"Lợi suất vượt trội {ticker} (%)", hovermode="closest")
    return f


def rolling_beta_chart(beta: pd.Series, full_beta: float, window: int = 126) -> go.Figure:
    """Beta trượt; tiêu đề nêu đúng cửa sổ ước lượng đang dùng."""
    f = _fig(f"Beta trượt {window} phiên (~{window / 21:.0f} tháng)")
    f.add_trace(go.Scatter(x=beta.index, y=beta, name="Beta trượt",
                           line=dict(color=COLOR_PRIMARY)))
    f.add_hline(y=full_beta, line_dash="dash", line_color=COLOR_DOWN,
                annotation_text=f"Beta toàn kỳ = {full_beta:.2f}")
    f.add_hline(y=1.0, line_dash="dot", line_color=COLOR_MUTED, annotation_text="Thị trường = 1")
    return f


def sml_chart(capm_res, ticker: str) -> go.Figure:
    from src.analytics.capm import security_market_line

    sml = security_market_line(capm_res)
    f = _fig("Đường thị trường chứng khoán (SML)")
    f.add_trace(go.Scatter(x=sml["beta"], y=sml["expected_return"] * 100, name="SML",
                           line=dict(color=COLOR_MUTED, width=2)))
    f.add_trace(go.Scatter(x=[capm_res.beta], y=[capm_res.expected_return * 100],
                           mode="markers+text", name="Kỳ vọng CAPM", text=["Kỳ vọng"],
                           textposition="bottom center",
                           marker=dict(size=12, color=COLOR_PRIMARY)))
    f.add_trace(go.Scatter(x=[capm_res.beta], y=[capm_res.realized_return * 100],
                           mode="markers+text", name="Thực tế", text=[ticker],
                           textposition="top center",
                           marker=dict(size=14, color=COLOR_UP, symbol="star")))
    f.update_layout(xaxis_title="Beta", yaxis_title="Lợi suất năm (%)", hovermode="closest")
    return f


def factor_contribution(apt_res) -> go.Figure:
    from src.analytics.apt import FACTOR_LABELS

    c = apt_res.contributions * 100
    labels = [FACTOR_LABELS.get(i, i) for i in c.index]
    f = _fig("Đóng góp của từng nhân tố vào lợi suất kỳ vọng (%/năm)")
    f.add_trace(go.Bar(x=c.values, y=labels, orientation="h",
                       marker_color=np.where(c.values >= 0, COLOR_UP, COLOR_DOWN),
                       text=[f"{v:+.2f}%" for v in c.values], textposition="outside"))
    f.update_layout(xaxis_title="%/năm", hovermode="closest")
    return f


def factor_betas(apt_res) -> go.Figure:
    from src.analytics.apt import FACTOR_LABELS

    b = apt_res.params.drop("const")
    p = apt_res.pvalues.drop("const")
    labels = [FACTOR_LABELS.get(i, i) for i in b.index]
    colors = [COLOR_PRIMARY if pv < 0.05 else COLOR_MUTED for pv in p]
    f = _fig("Hệ số nhạy với từng nhân tố rủi ro (đậm = có ý nghĩa 5%)")
    f.add_trace(go.Bar(x=labels, y=b.values, marker_color=colors,
                       text=[f"{v:.3f}" for v in b.values], textposition="outside"))
    f.update_layout(yaxis_title="Hệ số b", hovermode="closest")
    return f


# Monte Carlo
def mc_fan(sim, ticker: str, sample: int = 60) -> go.Figure:
    pct = sim.percentiles
    x = np.arange(len(pct))
    f = _fig(f"Mô phỏng Monte Carlo giá {ticker} — {sim.method}")
    for i in range(min(sample, sim.paths.shape[1])):
        f.add_trace(go.Scatter(x=x, y=sim.paths[:, i], mode="lines", showlegend=False,
                               line=dict(width=0.5, color="rgba(37,99,235,0.12)"),
                               hoverinfo="skip"))
    f.add_trace(go.Scatter(x=x, y=pct["p95"], name="Phân vị 95%",
                           line=dict(color=COLOR_UP, width=1.5, dash="dash")))
    f.add_trace(go.Scatter(x=x, y=pct["p50"], name="Trung vị",
                           line=dict(color=COLOR_PRIMARY, width=2.5)))
    f.add_trace(go.Scatter(x=x, y=pct["p5"], name="Phân vị 5%",
                           line=dict(color=COLOR_DOWN, width=1.5, dash="dash")))
    f.add_hline(y=sim.s0, line_dash="dot", line_color=COLOR_MUTED,
                annotation_text=f"Giá hiện tại {sim.s0:,.1f}")
    f.update_layout(xaxis_title="Số phiên tới", yaxis_title="Giá (nghìn đồng)")
    return f


def mc_histogram(sim, ticker: str) -> go.Figure:
    stats = sim.stats()
    var = stats["VaR 95% (toàn kỳ)"]
    f = _fig(f"Phân phối giá {ticker} sau {sim.horizon} phiên")
    f.add_trace(go.Histogram(x=sim.terminal, nbinsx=80, marker_color=COLOR_PRIMARY,
                             opacity=0.8, name="Kịch bản"))
    f.add_vline(x=sim.s0, line_dash="dot", line_color=COLOR_MUTED,
                annotation_text="Giá hiện tại")
    f.add_vline(x=sim.s0 * (1 + var), line_dash="dash", line_color=COLOR_DOWN,
                annotation_text=f"VaR 95%: {var * 100:.1f}%")
    f.update_layout(xaxis_title="Giá cuối kỳ (nghìn đồng)", yaxis_title="Số kịch bản",
                    hovermode="x")
    return f


# Phân bổ vốn
def cal_chart(cal: pd.DataFrame, opt, mu: float, sigma: float, rf: float,
              ticker: str) -> go.Figure:
    f = _fig("Đường phân bổ vốn (CAL): cổ phiếu + tiền gửi")
    f.add_trace(go.Scatter(x=cal["volatility"] * 100, y=cal["expected_return"] * 100,
                           name="CAL", line=dict(color=COLOR_PRIMARY, width=2)))
    f.add_trace(go.Scatter(x=[0], y=[rf * 100], mode="markers+text", text=["Tiền gửi"],
                           textposition="top right", name="Tài sản phi rủi ro",
                           marker=dict(size=11, color=COLOR_MUTED)))
    f.add_trace(go.Scatter(x=[sigma * 100], y=[mu * 100], mode="markers+text",
                           text=[f"100% {ticker}"], textposition="top center",
                           name=ticker, marker=dict(size=13, color=COLOR_UP)))
    f.add_trace(go.Scatter(x=[opt.volatility * 100], y=[opt.expected_return * 100],
                           mode="markers+text", text=[f"Tối ưu y*={opt.y_optimal:.0%}"],
                           textposition="bottom right", name="Danh mục tối ưu",
                           marker=dict(size=15, color=COLOR_ACCENT, symbol="star")))
    f.update_layout(xaxis_title="Độ biến động năm (%)", yaxis_title="Lợi suất kỳ vọng (%/năm)",
                    hovermode="closest")
    return f


def nav_comparison(navs: pd.DataFrame, title: str = "So sánh các mức phân bổ") -> go.Figure:
    f = _fig(title)
    palette = [COLOR_MUTED, "#93c5fd", COLOR_PRIMARY, "#1d4ed8", COLOR_UP, COLOR_ACCENT]
    for i, col in enumerate(navs.columns):
        f.add_trace(go.Scatter(x=navs.index, y=navs[col], name=str(col),
                               line=dict(color=palette[i % len(palette)], width=2)))
    f.update_layout(yaxis_title="NAV (lần vốn ban đầu)")
    return f


# Phân tích khám phá dữ liệu
def qq_plot(returns: pd.Series, ticker: str, df_t: float = 4.0) -> go.Figure:
    """Biểu đồ Q-Q so phân phối lợi suất thực tế với phân phối chuẩn và Student-t."""
    qn = qq_data(returns, "norm")
    qt = qq_data(returns, "t", df_t)
    lim = float(np.nanmax(np.abs(qn["thuc_nghiem"]))) * 1.05

    f = _fig(f"Biểu đồ Q-Q của lợi suất {ticker}")
    f.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
                           name="Khớp hoàn hảo",
                           line=dict(color=COLOR_MUTED, width=1.5, dash="dash")))
    f.add_trace(go.Scatter(x=qn["ly_thuyet"], y=qn["thuc_nghiem"], mode="markers",
                           name="So với phân phối chuẩn",
                           marker=dict(size=4, color=COLOR_PRIMARY, opacity=0.6)))
    f.add_trace(go.Scatter(x=qt["ly_thuyet"], y=qt["thuc_nghiem"], mode="markers",
                           name=f"So với Student-t (df={df_t:g})",
                           marker=dict(size=4, color=COLOR_ACCENT, opacity=0.6)))
    f.update_layout(xaxis_title="Phân vị lý thuyết (đã chuẩn hoá)",
                    yaxis_title="Phân vị thực nghiệm (đã chuẩn hoá)", hovermode="closest")
    return f


def acf_chart(returns: pd.Series, ticker: str, nlags: int = 20) -> go.Figure:
    """Hệ số tự tương quan của lợi suất và của bình phương lợi suất."""
    from statsmodels.tsa.stattools import acf

    r = pd.Series(returns).dropna()
    lags = np.arange(1, nlags + 1)
    a1 = acf(r, nlags=nlags, fft=True)[1:]
    a2 = acf(r ** 2, nlags=nlags, fft=True)[1:]
    ci = 1.96 / np.sqrt(len(r))

    f = _fig(f"Tự tương quan lợi suất {ticker}")
    f.add_trace(go.Bar(x=lags, y=a1, name="Lợi suất", marker_color=COLOR_PRIMARY))
    f.add_trace(go.Bar(x=lags, y=a2, name="Bình phương lợi suất", marker_color=COLOR_ACCENT))
    for sign in (1, -1):
        f.add_hline(y=sign * ci, line_dash="dash", line_color=COLOR_MUTED)
    f.update_layout(xaxis_title="Độ trễ (phiên)", yaxis_title="Hệ số tự tương quan",
                    barmode="group")
    return f


def correlation_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    """Ma trận tương quan dạng bản đồ nhiệt."""
    f = _fig(title)
    f.add_trace(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index), zmid=0, zmin=-1, zmax=1,
        colorscale=[[0, COLOR_DOWN], [0.5, "#ffffff"], [1, COLOR_PRIMARY]],
        text=corr.round(2).values, texttemplate="%{text}",
    ))
    f.update_layout(hovermode="closest")
    return f
