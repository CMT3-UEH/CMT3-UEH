# Tải dữ liệu về data/raw và ghép thành bảng giá thống nhất ở data/processed.

from src.console import setup as _setup_console

_setup_console()

import pandas as pd

from src.config import BENCHMARK, PROCESSED_DIR, TICKER
from src.data.loaders import fetch_macro, fetch_ohlcv

PANEL_PATH = PROCESSED_DIR / f"panel_{TICKER}.csv"


def build_panel(
    stock: pd.DataFrame,
    benchmark: pd.DataFrame,
    macro: pd.DataFrame | None,
) -> pd.DataFrame:
    # Ghép giá cổ phiếu, chỉ số và nhân tố vĩ mô theo lịch giao dịch Việt Nam.
    s = stock.set_index("time").add_prefix("stock_")
    b = benchmark.set_index("time")[["close"]].rename(columns={"close": "benchmark_close"})
    panel = s.join(b, how="left")

    if macro is not None and not macro.empty:
        m = macro.copy()
        m["time"] = pd.to_datetime(m["time"])
        m = m.set_index("time")
        panel = panel.join(m, how="left")
        # Nhân tố vĩ mô nghỉ lệch ngày với thị trường Việt Nam nên được điền tiến;
        # đây là thông tin quá khứ nên không gây rò rỉ.
        panel[list(m.columns)] = panel[list(m.columns)].ffill()

    panel = panel.dropna(subset=["stock_close"])

    thieu_bm = int(panel["benchmark_close"].isna().sum())
    if thieu_bm:
        print(f"  Loại {thieu_bm} phiên thiếu dữ liệu {BENCHMARK} "
              f"(không điền tiến để tránh lợi suất thị trường bằng 0 giả tạo)")
        panel = panel.dropna(subset=["benchmark_close"])

    panel.index.name = "time"
    return panel


def load_panel() -> pd.DataFrame:
    # Đọc bảng giá đã dựng sẵn.
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Chưa có {PANEL_PATH}. Chạy `python fetch_data.py` trước."
        )
    return pd.read_csv(PANEL_PATH, index_col="time", parse_dates=["time"])


def main() -> None:
    print(f"Tải dữ liệu cho {TICKER} và {BENCHMARK} ...")
    stock = fetch_ohlcv(TICKER, refresh=True)
    benchmark = fetch_ohlcv(BENCHMARK, refresh=True)
    macro = fetch_macro(refresh=True)
    for ten, bang in (("cổ phiếu", stock), ("chỉ số", benchmark), ("vĩ mô", macro)):
        print(f"  {ten:<10} {bang.shape}")

    panel = build_panel(stock, benchmark, macro)
    panel.to_csv(PANEL_PATH, encoding="utf-8-sig")

    print(f"\nBảng giá tổng hợp: {panel.shape} -> {PANEL_PATH}")
    print(f"Khoảng thời gian: {panel.index.min():%d/%m/%Y} – {panel.index.max():%d/%m/%Y}")
    thieu = panel.isna().sum()
    if thieu.any():
        print("\nSố ô thiếu theo cột:")
        print(thieu[thieu > 0])


if __name__ == "__main__":
    main()
