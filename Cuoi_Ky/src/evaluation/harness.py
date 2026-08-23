# Khung đánh giá dùng chung — mọi chiến lược trong dự án đi qua đúng hàm này.

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from src.config import REPORT_DIR
from src.evaluation.metrics import (
    annual_return,
    annual_volatility,
    calmar_ratio,
    hit_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

TEST_LOG = REPORT_DIR / "test_touches.log"
PARTS = ("train", "valid", "test")


def log_test_touch(ly_do: str, so_cau_hinh: int = 1) -> None:
    # Ghi nhận một lần chạy trên tập kiểm tra.
    with TEST_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{so_cau_hinh}\t{ly_do}\n")


def count_test_touches() -> tuple[int, int]:
    # Trả về (số lần chạm, tổng số cấu hình đã đánh giá trên tập kiểm tra).
    if not TEST_LOG.exists():
        return 0, 0
    dong = [d for d in TEST_LOG.read_text(encoding="utf-8").splitlines() if d.strip()]
    tong = sum(int(d.split("\t")[1]) for d in dong)
    return len(dong), tong


@dataclass
class StrategyResult:
    # Kết quả của một chiến lược trên từng phần dữ liệu.

    name: str
    signal: pd.Series
    runs: dict[str, BacktestResult] = field(default_factory=dict)
    note: str = ""

    def returns(self, part: str) -> pd.Series:
        # Bỏ phiên đầu của mỗi đoạn: vị thế khởi tạo bằng 0 nên phiên đó chưa có ý nghĩa.
        return self.runs[part].returns.iloc[1:]

    def metrics(self, part: str) -> dict:
        r = self.returns(part)
        kq = self.runs[part]
        nhip = kq.tempo()
        return {
            "Chiến lược": self.name,
            "CAGR": annual_return(r),
            "Độ biến động": annual_volatility(r),
            "Sharpe": sharpe_ratio(r),
            "Sortino": sortino_ratio(r),
            "Calmar": calmar_ratio(r),
            "Sụt giảm tối đa": max_drawdown(r),
            "Tỷ lệ phiên tăng": hit_ratio(r),
            "Lệnh mỗi năm": nhip["Số lần khớp lệnh mỗi năm"],
            "Phiên giữa hai lệnh": nhip["Số phiên giữa hai lệnh"],
            "Quy mô lệnh trung bình": nhip["Quy mô lệnh trung bình"],
            "Tỷ lệ thời gian có vị thế": nhip["Tỷ lệ thời gian có vị thế"],
            "Tổng chi phí giao dịch": nhip["Tổng chi phí giao dịch"],
        }


def evaluate_strategy(
    name: str,
    signal: pd.Series,
    ds: pd.DataFrame,
    d,
    parts: tuple[str, ...] = ("train", "valid"),
    config: BacktestConfig | None = None,
    note: str = "",
) -> StrategyResult:
    # Chạy một chiến lược trên các phần dữ liệu được chỉ định.
    if "test" in parts:
        log_test_touch(f"evaluate_strategy: {name}")

    kq = StrategyResult(name=name, signal=signal, note=note)
    for part in parts:
        idx = d.index(part)
        kq.runs[part] = run_backtest(ds.loc[idx], signal.reindex(idx), config)
    return kq


def comparison_table(
    ket_qua: list[StrategyResult], part: str = "valid"
) -> pd.DataFrame:
    # Bảng so sánh nhiều chiến lược trên cùng một phần dữ liệu.
    bang = pd.DataFrame([k.metrics(part) for k in ket_qua if part in k.runs])
    return bang.set_index("Chiến lược").sort_values("Sharpe", ascending=False)


def format_table(bang: pd.DataFrame) -> pd.DataFrame:
    # Định dạng bảng so sánh cho dễ đọc trên màn hình và trong báo cáo.
    pct = ["CAGR", "Độ biến động", "Sụt giảm tối đa", "Tỷ lệ phiên tăng",
           "Tỷ lệ thời gian có vị thế", "Tổng chi phí giao dịch", "Quy mô lệnh trung bình"]
    so = ["Sharpe", "Sortino", "Calmar", "Lệnh mỗi năm", "Phiên giữa hai lệnh"]
    ra = bang.copy()
    for c in pct:
        if c in ra:
            ra[c] = ra[c].map(lambda x: "—" if pd.isna(x) else f"{x:.2%}")
    for c in so:
        if c in ra:
            ra[c] = ra[c].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    return ra
