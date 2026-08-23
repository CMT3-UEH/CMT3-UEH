# Thư viện biểu đồ Plotly dùng chung cho ứng dụng.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.formatting import cnt, num
from src.config import (
    COLOR_ACCENT,
    COLOR_DOWN,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_UP,
)

BANG_MAU = [COLOR_PRIMARY, COLOR_ACCENT, COLOR_UP, COLOR_DOWN, COLOR_MUTED,
            "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16", "#f97316"]


def _khung(fig: go.Figure, tieu_de: str = "", cao: int = 420) -> go.Figure:
    fig.update_layout(
        # ",." = dấu thập phân phẩy, hàng nghìn chấm — chuẩn Việt Nam.
        separators=",.",
        title=tieu_de or None,
        height=cao,
        margin=dict(l=10, r=10, t=45 if tieu_de else 15, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        template="plotly_white",
    )
    return fig


def duong_tai_san(nav: dict[str, pd.Series], tieu_de: str = "",
                  log: bool = True) -> go.Figure:
    # Đường tăng trưởng vốn của nhiều chiến lược đặt chồng nhau.
    fig = go.Figure()
    for i, (ten, s) in enumerate(nav.items()):
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=ten, mode="lines",
            line=dict(width=2.2 if i == 0 else 1.6, color=BANG_MAU[i % len(BANG_MAU)]),
        ))
    fig.update_yaxes(type="log" if log else "linear", title="Tăng trưởng 1 đồng vốn")
    return _khung(fig, tieu_de, 460)


def drawdown(dd: dict[str, pd.Series], tieu_de: str = "") -> go.Figure:
    # Biểu đồ sụt giảm so với đỉnh — nhìn rủi ro thật rõ hơn nhìn lợi suất.
    fig = go.Figure()
    for i, (ten, s) in enumerate(dd.items()):
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values * 100, name=ten, mode="lines",
            fill="tozeroy" if i == 0 else None,
            line=dict(width=1.5, color=BANG_MAU[i % len(BANG_MAU)]),
        ))
    fig.update_yaxes(title="Sụt giảm (%)")
    return _khung(fig, tieu_de, 320)


def gia_va_vi_the(px: pd.DataFrame, vi_the: pd.Series, tieu_de: str = "") -> go.Figure:
    # Giá cổ phiếu kèm dải nền đánh dấu những đoạn đang nắm giữ.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=px.index, y=px["close"], name="Giá FPT",
                             line=dict(color=COLOR_PRIMARY, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=vi_the.index, y=vi_the.values, name="Vị thế",
                             line=dict(color=COLOR_ACCENT, width=1.4),
                             fill="tozeroy"), row=2, col=1)
    fig.update_yaxes(title="Nghìn đồng", row=1, col=1)
    fig.update_yaxes(title="Tỷ trọng", range=[-0.05, 1.05], row=2, col=1)
    return _khung(fig, tieu_de, 520)


def cong_nhip(cong: pd.DataFrame, gia: pd.Series | None = None,
              tieu_de: str = "") -> go.Figure:
    # Trọng số cổng chọn tầm dự báo theo thời gian, vẽ chồng lên giá.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.4, 0.6], vertical_spacing=0.05)
    if gia is not None:
        fig.add_trace(go.Scatter(x=gia.index, y=gia.values, name="Giá FPT",
                                 line=dict(color=COLOR_MUTED, width=1.3)), row=1, col=1)

    for i, cot in enumerate(cong.columns):
        fig.add_trace(go.Scatter(
            x=cong.index, y=cong[cot], name=f"tầm dự báo {cot}", mode="lines",
            stackgroup="nhip", line=dict(width=0.6, color=BANG_MAU[i % len(BANG_MAU)]),
        ), row=2, col=1)

    fig.update_yaxes(title="Nghìn đồng", row=1, col=1)
    fig.update_yaxes(title="Trọng số cổng", range=[0, 1], row=2, col=1)
    return _khung(fig, tieu_de, 520)


def phan_phoi_nam_giu(do_dai: pd.Series, tieu_de: str = "") -> go.Figure:
    # Phân phối độ dài các lần nắm giữ, tính bằng số phiên.
    fig = go.Figure(go.Histogram(x=do_dai.values, nbinsx=40,
                                 marker_color=COLOR_PRIMARY))
    fig.update_xaxes(title="Số phiên nắm giữ mỗi lần vào lệnh")
    fig.update_yaxes(title="Số lần")
    return _khung(fig, tieu_de, 340)


def cot_so_sanh(bang: pd.DataFrame, cot: str, tieu_de: str = "",
                nguong: float | None = None) -> go.Figure:
    # So sánh một chỉ tiêu giữa các chiến lược, sắp xếp giảm dần.
    s = bang[cot].sort_values()
    mau = [COLOR_UP if v >= 0 else COLOR_DOWN for v in s.values]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker_color=mau,
                           text=[num(v, 2, vi=True) for v in s.values],
                           textposition="outside"))
    if nguong is not None:
        fig.add_vline(x=nguong, line=dict(color=COLOR_ACCENT, dash="dash", width=2),
                      annotation_text=f"chỉ tiêu {nguong}")
    fig.update_xaxes(title=cot)
    return _khung(fig, tieu_de, max(340, 26 * len(s) + 120))


def thac_nuoc_ablation(bang: pd.DataFrame, cot: str = "Sharpe",
                       tieu_de: str = "") -> go.Figure:
    # Biểu đồ thác nước cho mức tăng cộng dồn qua các cấu hình ablation.
    gia_tri = bang[cot]
    delta = gia_tri.diff()
    delta.iloc[0] = gia_tri.iloc[0]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * (len(gia_tri) - 1),
        x=list(gia_tri.index),
        y=list(delta.values),
        text=[("+" if v >= 0 else "") + num(v, 3, vi=True) for v in delta.values],
        textposition="outside",
        increasing=dict(marker=dict(color=COLOR_UP)),
        decreasing=dict(marker=dict(color=COLOR_DOWN)),
        totals=dict(marker=dict(color=COLOR_PRIMARY)),
    ))
    fig.update_yaxes(title=cot)
    return _khung(fig, tieu_de, 440)


def khoang_tin_cay(bang: pd.DataFrame, tieu_de: str = "") -> go.Figure:
    # Sharpe kèm khoảng tin cậy — biểu đồ chống lại việc đọc bảng xếp hạng máy móc.
    s = bang.sort_values("Sharpe")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s["Sharpe"], y=s.index, mode="markers",
        marker=dict(size=10, color=COLOR_PRIMARY),
        error_x=dict(
            type="data", symmetric=False,
            array=(s["KTC 95% trên"] - s["Sharpe"]).clip(lower=0),
            arrayminus=(s["Sharpe"] - s["KTC 95% dưới"]).clip(lower=0),
            color=COLOR_MUTED, thickness=1.4,
        ),
        name="Sharpe",
    ))
    fig.add_vline(x=0, line=dict(color=COLOR_MUTED, dash="dot"))
    fig.update_xaxes(title="Sharpe (khoảng tin cậy 95% bằng bootstrap theo khối)")
    return _khung(fig, tieu_de, max(340, 26 * len(s) + 120))


def chia_du_lieu(index: pd.DatetimeIndex, split, gia: pd.Series) -> go.Figure:
    # Trực quan hoá lần chia train/valid/test kèm vùng thanh lọc và cách ly.
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gia.index, y=gia.values, name="Giá FPT",
                             line=dict(color=COLOR_MUTED, width=1.2)))
    vung = [
        ("Huấn luyện", split.train, "rgba(37,99,235,0.12)"),
        ("Kiểm định", split.valid, "rgba(245,158,11,0.16)"),
        ("Kiểm tra", split.test, "rgba(22,163,74,0.14)"),
    ]
    for ten, idx, mau in vung:
        if idx is None or len(idx) == 0:
            continue
        fig.add_vrect(x0=idx.min(), x1=idx.max(), fillcolor=mau, line_width=0,
                      annotation_text=f"{ten} ({cnt(len(idx), vi=True)} phiên)",
                      annotation_position="top left")

    bo = split.purged.union(split.embargoed)
    if len(bo):
        fig.add_trace(go.Scatter(
            x=bo, y=gia.reindex(bo).values, mode="markers", name="Đã loại bỏ",
            marker=dict(color=COLOR_DOWN, size=5, symbol="x"),
        ))
    fig.update_yaxes(title="Nghìn đồng", type="log")
    return _khung(fig, "", 420)


def sharpe_truot(r: dict[str, pd.Series], cua_so: int = 252,
                 tieu_de: str = "") -> go.Figure:
    # Sharpe trượt — cho thấy hiệu quả ổn định hay chỉ tập trung ở vài giai đoạn.
    from src.evaluation.metrics import rolling_metric

    fig = go.Figure()
    for i, (ten, s) in enumerate(r.items()):
        fig.add_trace(go.Scatter(
            x=s.index, y=rolling_metric(s, cua_so, "sharpe"), name=ten,
            line=dict(width=1.6, color=BANG_MAU[i % len(BANG_MAU)]),
        ))
    fig.add_hline(y=0, line=dict(color=COLOR_MUTED, dash="dot"))
    fig.update_yaxes(title=f"Sharpe trượt {cua_so} phiên")
    return _khung(fig, tieu_de, 360)


def ban_do_nhiet_thang(bang: pd.DataFrame, tieu_de: str = "") -> go.Figure:
    # Bản đồ nhiệt lợi suất theo tháng và năm.
    z = bang.drop(columns=[c for c in bang.columns if c == "Cả năm"], errors="ignore")
    fig = go.Figure(go.Heatmap(
        z=z.values * 100, x=list(z.columns), y=[str(i) for i in z.index],
        colorscale="RdYlGn", zmid=0,
        text=np.round(z.values * 100, 1), texttemplate="%{text}",
        colorbar=dict(title="%"),
    ))
    return _khung(fig, tieu_de, max(320, 26 * len(z) + 120))
