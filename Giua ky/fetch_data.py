#Tải toàn bộ dữ liệu về thư mục data/raw và tiền xử lý ra data/processed.

import sys
from pathlib import Path


import pandas as pd  # noqa: E402

from src.config import BENCHMARK, PROCESSED_DIR, TICKER  # noqa: E402
from src.data.loaders import fetch_all  # noqa: E402


def build_panel(stock: pd.DataFrame, benchmark: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    #Ghép giá cổ phiếu, chỉ số và nhân tố vĩ mô về một bảng theo ngày giao dịch VN.
    s = stock.set_index("time").add_prefix("stock_")
    b = benchmark.set_index("time")[["close"]].rename(columns={"close": "benchmark_close"})
    panel = s.join(b, how="left")

    macro_cols: list[str] = []
    if macro is not None and not macro.empty:
        m = macro.copy()
        m["time"] = pd.to_datetime(m["time"])
        m = m.set_index("time")
        macro_cols = list(m.columns)
        panel = panel.join(m, how="left")
        panel[macro_cols] = panel[macro_cols].ffill()

    panel = panel.dropna(subset=["stock_close"])

    missing_bm = int(panel["benchmark_close"].isna().sum())
    if missing_bm:
        print(f"  Loại {missing_bm} phiên thiếu dữ liệu VNINDEX "
              f"(không điền tiến để tránh lợi suất thị trường bằng 0 giả tạo)")
        panel = panel.dropna(subset=["benchmark_close"])
    return panel


def main() -> None:
    print(f"Tải dữ liệu cho {TICKER} và {BENCHMARK} ...")
    data = fetch_all(refresh=True)
    panel = build_panel(data["stock"], data["benchmark"], data["macro"])
    out = PROCESSED_DIR / f"panel_{TICKER}.csv"
    panel.to_csv(out, encoding="utf-8-sig")
    print(f"\nBang du lieu tong hop: {panel.shape} -> {out}")
    print(f"Khoang thoi gian: {panel.index.min():%d/%m/%Y} - {panel.index.max():%d/%m/%Y}")
    missing = panel.isna().sum()
    if missing.any():
        print("\nSo o thieu theo cot:")
        print(missing[missing > 0])


if __name__ == "__main__":
    main()
