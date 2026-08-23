# Lớp tổng hợp dùng chung cho ứng dụng Streamlit, báo cáo và slide.

from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from fetch_data import load_panel
from src.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from src.config import REPORT_DIR, SHARPE_TARGET, TICKER
from src.dataset import Dataset, make_dataset
from src.evaluation.metrics import summary_table
from src.features.builder import build_dataset, feature_columns, feature_groups

TEP_TIN_HIEU = {
    "baseline": "tin_hieu_baseline.parquet",
    "học sâu": "tin_hieu_hoc_sau.parquet",
    "đề xuất": "tin_hieu_atfn.parquet",
}

BANG_BAO_CAO = {
    "đặc trưng": "01_dac_trung.csv",
    "chia dữ liệu": "02_chia_du_lieu.csv",
    "mốc mua và giữ": "03_moc_mua_va_giu.csv",
    "chi phí theo nhịp": "04_chi_phi_theo_nhip.csv",
    "baseline valid": "05_baseline_valid.csv",
    "độ tin cậy valid": "06_do_tin_cay_valid.csv",
    "ứng viên tiến dần": "07_ung_vien_tien_dan.csv",
    "học sâu valid": "08_hoc_sau_valid.csv",
    "ablation valid": "09_ablation_valid.csv",
    "kết quả test": "10_ket_qua_test.csv",
    "độ tin cậy test": "11_do_tin_cay_test.csv",
    "alpha beta": "12_alpha_beta.csv",
    "theo năm": "13_theo_nam.csv",
    "theo chế độ": "13_theo_che_do.csv",
    "độ vững": "14_do_ben_vung.csv",
    "ablation nhịp": "15_ablation_nhip.csv",
}


@dataclass
class KhoDuLieu:
    # Toàn bộ dữ liệu và kết quả của dự án, nạp một lần rồi dùng chung.

    ds: pd.DataFrame
    d: Dataset
    tin_hieu: pd.DataFrame
    bang: dict[str, pd.DataFrame] = field(default_factory=dict)
    nhip: pd.DataFrame | None = None

    @property
    def features(self) -> list[str]:
        return feature_columns(self.ds)

    @property
    def nhom_dac_trung(self) -> dict[str, list[str]]:
        return feature_groups(self.features)

    def gia(self, part: str = "all") -> pd.DataFrame:
        return self.ds.loc[self.d.index(part)]

    def co_bang(self, ten: str) -> bool:
        return ten in self.bang and not self.bang[ten].empty

    def chay(self, ten_tin_hieu: str, part: str = "test",
             config: BacktestConfig | None = None) -> BacktestResult:
        idx = self.d.index(part)
        return run_backtest(self.ds.loc[idx], self.tin_hieu[ten_tin_hieu].reindex(idx),
                            config)

    def tom_tat_tai_san(self, part: str = "all") -> pd.DataFrame:
        r = self.gia(part)["close"].pct_change().dropna()
        return summary_table(r, label=TICKER)

    def dat_muc_tieu(self, ten_tin_hieu: str) -> tuple[float, bool]:
        from src.evaluation.metrics import sharpe_ratio

        s = sharpe_ratio(self.chay(ten_tin_hieu, "test").returns.iloc[1:])
        return s, s >= SHARPE_TARGET


def _nap_bang() -> dict[str, pd.DataFrame]:
    bang = {}
    for ten, tep in BANG_BAO_CAO.items():
        duong = REPORT_DIR / tep
        if duong.exists():
            bang[ten] = pd.read_csv(duong, index_col=0)
    return bang


def _nap_tin_hieu(index: pd.DatetimeIndex) -> pd.DataFrame:
    phan = [
        pd.read_parquet(REPORT_DIR / tep)
        for tep in TEP_TIN_HIEU.values()
        if (REPORT_DIR / tep).exists()
    ]
    if not phan:
        return pd.DataFrame(index=index)
    return pd.concat(phan, axis=1).reindex(index).fillna(0.0)


@lru_cache(maxsize=1)
def nap_kho() -> KhoDuLieu:
    # Nạp toàn bộ kho dữ liệu.
    ds = build_dataset(load_panel())
    d = make_dataset(ds, horizon=1)
    nhip_path = REPORT_DIR / "nhip_atfn.parquet"
    return KhoDuLieu(
        ds=ds,
        d=d,
        tin_hieu=_nap_tin_hieu(ds.index),
        bang=_nap_bang(),
        nhip=pd.read_parquet(nhip_path) if nhip_path.exists() else None,
    )


def trang_thai_giai_doan() -> pd.DataFrame:
    # Giai đoạn nào đã chạy xong, dựa trên tệp kết quả có mặt trong reports/.
    can = {
        "1–2 · Dữ liệu và bộ máy backtest": ["01_dac_trung.csv", "03_moc_mua_va_giu.csv"],
        "3–4a · Baseline cổ điển và máy học": ["05_baseline_valid.csv"],
        "4b · Baseline học sâu": ["08_hoc_sau_valid.csv"],
        "5 · Mô hình đề xuất và ablation": ["09_ablation_valid.csv"],
        "6 · Đánh giá trên tập kiểm tra": ["10_ket_qua_test.csv"],
    }
    hang = {}
    for ten, tep in can.items():
        du = all((REPORT_DIR / t).exists() for t in tep)
        hang[ten] = {"Trạng thái": "đã chạy" if du else "chưa chạy"}
    return pd.DataFrame(hang).T
