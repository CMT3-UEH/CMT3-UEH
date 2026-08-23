# Kiến trúc học sâu — baseline nhóm B và các khối dùng lại cho mô hình đề xuất.

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import HORIZONS, LOOKBACK

# Kích thước cố định cho mọi baseline học sâu, chọn theo quy mô dữ liệu chứ không
# theo kết quả dò tìm. Khoảng 2.400 quan sát huấn luyện chỉ đỡ nổi mô hình cỡ này.
AN = 64            # số nơ-ron ẩn
LOP = 2            # số lớp
DAU = 4            # số đầu chú ý của Transformer
DROPOUT = 0.2


@dataclass
class WindowData:
    # Tập cửa sổ trượt đã chuẩn bị sẵn cho PyTorch.

    X: torch.Tensor            # (N, L, d)
    y: torch.Tensor            # (N,) lợi suất tương lai ở tầm dự báo chính
    index: pd.DatetimeIndex    # thời điểm ra quyết định của từng cửa sổ
    Y: torch.Tensor = None     # (N, H) lợi suất ở mọi tầm dự báo, cho đầu ra đa tầm dự báo
    horizons: tuple = ()

    def __len__(self) -> int:
        return len(self.index)

    def subset(self, mask: np.ndarray) -> "WindowData":
        return WindowData(
            self.X[mask], self.y[mask], self.index[mask],
            None if self.Y is None else self.Y[mask], self.horizons,
        )

    def select(self, idx: pd.DatetimeIndex) -> "WindowData":
        return self.subset(self.index.isin(idx))


def build_windows(
    ds: pd.DataFrame,
    features: list[str],
    scaler,
    horizon: int = 1,
    lookback: int = LOOKBACK,
    horizons: tuple = HORIZONS,
) -> WindowData:
    # Dựng tensor cửa sổ trượt từ bảng dữ liệu.
    X = scaler.transform(ds[features]).to_numpy(dtype=np.float32)
    y = ds[f"y_{horizon}"].to_numpy(dtype=np.float32)

    # Nhãn của mọi tầm dự báo, phục vụ đầu ra đa tầm dự báo của mô hình đề xuất.
    co = [h for h in horizons if f"y_{h}" in ds.columns]
    Y = ds[[f"y_{h}" for h in co]].to_numpy(dtype=np.float32) if co else None

    # Cửa sổ trượt không sao chép dữ liệu, chỉ tạo khung nhìn theo bước nhảy.
    n, d = X.shape
    if n <= lookback:
        raise ValueError(f"Chuỗi quá ngắn ({n}) so với cửa sổ nhìn lại ({lookback})")

    khung = np.lib.stride_tricks.sliding_window_view(X, lookback, axis=0)
    khung = np.ascontiguousarray(khung.transpose(0, 2, 1))     # (N, L, d)
    vi_tri = np.arange(lookback - 1, n)

    hop_le = np.isfinite(khung).all(axis=(1, 2))
    return WindowData(
        X=torch.from_numpy(khung[hop_le]),
        y=torch.from_numpy(y[vi_tri][hop_le]),
        index=ds.index[vi_tri][hop_le],
        Y=None if Y is None else torch.from_numpy(Y[vi_tri][hop_le]),
        horizons=tuple(co),
    )


class ChuanHoaDauVao(nn.Module):
    # Lớp chuẩn hoá theo đặc trưng, đặt ngay đầu mọi mô hình.

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class MLP(nn.Module):
    # B2 — chỉ nhìn phiên gần nhất, không có cấu trúc thời gian.

    def __init__(self, d: int, an: int = AN):
        super().__init__()
        self.chuan = ChuanHoaDauVao(d)
        self.mang = nn.Sequential(
            nn.Linear(d, an), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(an, an // 2), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(an // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mang(self.chuan(x)[:, -1, :]).squeeze(-1)


class CNN1D(nn.Module):
    # B3 — tích chập một chiều, bắt mẫu hình cục bộ trong cửa sổ giá.

    def __init__(self, d: int, an: int = AN):
        super().__init__()
        self.chuan = ChuanHoaDauVao(d)
        self.conv = nn.Sequential(
            nn.Conv1d(d, an, kernel_size=5, padding=2), nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Conv1d(an, an, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.ra = nn.Linear(an, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.chuan(x).transpose(1, 2)         # (N, d, L)
        return self.ra(self.conv(h).squeeze(-1)).squeeze(-1)


class ChuoiHoi(nn.Module):
    # B4 — RNN, LSTM hoặc GRU, khác nhau đúng ở loại ô nhớ.

    def __init__(self, d: int, loai: str = "gru", an: int = AN, lop: int = LOP):
        super().__init__()
        self.chuan = ChuanHoaDauVao(d)
        lop_rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[loai]
        self.rnn = lop_rnn(d, an, num_layers=lop, batch_first=True,
                           dropout=DROPOUT if lop > 1 else 0.0)
        self.ra = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(an, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.chuan(x))
        return self.ra(h[:, -1, :]).squeeze(-1)


class MaHoaViTri(nn.Module):
    # Mã hoá vị trí dạng hình sin, tiêu chuẩn của Transformer.

    def __init__(self, d: int, do_dai_toi_da: int = 512):
        super().__init__()
        vi_tri = torch.arange(do_dai_toi_da).unsqueeze(1).float()
        chia = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10_000.0) / d))
        pe = torch.zeros(do_dai_toi_da, d)
        pe[:, 0::2] = torch.sin(vi_tri * chia)
        pe[:, 1::2] = torch.cos(vi_tri * chia[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class Transformer(nn.Module):
    # B5 — bộ mã hoá Transformer, tự chú ý theo trục thời gian.

    def __init__(self, d: int, an: int = AN, lop: int = LOP, dau: int = DAU):
        super().__init__()
        self.chuan = ChuanHoaDauVao(d)
        self.chieu = nn.Linear(d, an)
        self.vi_tri = MaHoaViTri(an)
        lop_ma_hoa = nn.TransformerEncoderLayer(
            d_model=an, nhead=dau, dim_feedforward=an * 2,
            dropout=DROPOUT, batch_first=True, activation="gelu",
        )
        self.ma_hoa = nn.TransformerEncoder(lop_ma_hoa, num_layers=lop)
        self.ra = nn.Sequential(nn.LayerNorm(an), nn.Linear(an, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.vi_tri(self.chieu(self.chuan(x)))
        return self.ra(self.ma_hoa(h)[:, -1, :]).squeeze(-1)


class TCN(nn.Module):
    # Mạng tích chập giãn nở — khối (A) của mô hình đề xuất.

    def __init__(self, d: int, an: int = AN, so_lop: int = 4, hat_nhan: int = 3):
        super().__init__()
        lop = []
        vao = d
        for i in range(so_lop):
            gian = 2 ** i
            dem = (hat_nhan - 1) * gian
            lop += [
                nn.Conv1d(vao, an, hat_nhan, padding=dem, dilation=gian),
                CatPhanThua(dem),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            ]
            vao = an
        self.mang = nn.Sequential(*lop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Nhận (N, L, d), trả về (N, L, an).
        return self.mang(x.transpose(1, 2)).transpose(1, 2)


class CatPhanThua(nn.Module):
    # Cắt phần đệm thừa bên phải để phép tích chập giữ tính nhân quả.

    def __init__(self, so_cat: int):
        super().__init__()
        self.so_cat = so_cat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.so_cat] if self.so_cat > 0 else x


# Danh mục baseline nhóm B, kích thước đã cố định trước.
BASELINE_HOC_SAU = {
    "B2 · MLP": lambda d: MLP(d),
    "B3 · CNN 1 chiều": lambda d: CNN1D(d),
    "B4a · RNN": lambda d: ChuoiHoi(d, "rnn"),
    "B4b · LSTM": lambda d: ChuoiHoi(d, "lstm"),
    "B4c · GRU": lambda d: ChuoiHoi(d, "gru"),
    "B5 · Transformer": lambda d: Transformer(d),
}


def dem_tham_so(mo_hinh: nn.Module) -> int:
    return sum(p.numel() for p in mo_hinh.parameters() if p.requires_grad)
