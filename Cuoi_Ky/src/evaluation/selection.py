# Chọn mô hình bằng kiểm định cuốn chiếu thay vì bằng một cửa sổ kiểm định duy nhất.

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.config import EMBARGO
from src.dataset import Standardizer
from src.evaluation.metrics import sharpe_ratio
from src.features.labeling import label_end_time
from src.splits import Split, purge_embargo


def walk_forward_folds(
    index: pd.DatetimeIndex,
    horizon: int = 1,
    train_span: int = 1_260,
    test_span: int = 126,
    embargo: int = EMBARGO,
    mode: str = "expanding",
) -> list[Split]:
    # Chuỗi cửa sổ cuốn chiếu: huấn luyện trên quá khứ, đánh giá trên đoạn kế tiếp.
    idx = pd.DatetimeIndex(index).sort_values()
    t1 = label_end_time(idx, horizon)
    folds, dau = [], 0

    while dau + train_span + test_span <= len(idx):
        lo = 0 if mode == "expanding" else dau
        train = idx[lo:dau + train_span]
        test = idx[dau + train_span:dau + train_span + test_span]
        sach, loc, cach = purge_embargo(train, test, t1, embargo)

        # Cắt thêm vùng đệm ngay trước ranh giới, cùng lý do như ở src/dataset.py.
        if embargo > 0 and len(sach) > embargo:
            dem = sach[-embargo:]
            sach = sach.difference(dem)
            cach = cach.union(dem)

        folds.append(Split(name=f"wf_{test[0]:%Y-%m}", train=sach, test=test,
                           purged=loc, embargoed=cach,
                           meta={"horizon": horizon, "t1": t1}))
        dau += test_span

    return folds


def sharpe_an_toan(r: pd.Series) -> float:
    # Sharpe của một đoạn, trả về 0 thay vì rỗng khi chiến lược đứng ngoài cả đoạn.
    x = r.dropna()
    if len(x) < 2:
        return 0.0
    s = sharpe_ratio(x)
    return 0.0 if not np.isfinite(s) else float(s)


@dataclass
class DiemTienDan:
    # Điểm của một cấu hình sau khi chạy qua toàn bộ cửa sổ cuốn chiếu.

    ten: str
    tham_so: dict
    sharpe_tung_cua_so: pd.Series
    loi_suat_gop: pd.Series

    @property
    def sharpe_gop(self) -> float:
        # Sharpe tính trên toàn bộ lợi suất ngoài mẫu của mọi cửa sổ ghép lại.
        return sharpe_an_toan(self.loi_suat_gop)

    @property
    def trung_binh(self) -> float:
        return float(self.sharpe_tung_cua_so.mean())

    @property
    def trung_vi(self) -> float:
        return float(self.sharpe_tung_cua_so.median())

    @property
    def do_lech(self) -> float:
        return float(self.sharpe_tung_cua_so.std(ddof=1))

    @property
    def ty_le_thang(self) -> float:
        # Tỷ lệ cửa sổ cho Sharpe dương — thước đo độ ổn định, chỉ để báo cáo.
        return float((self.sharpe_tung_cua_so > 0).mean())

    @property
    def diem_chon(self) -> float:
        # Tiêu chí chọn duy nhất: Sharpe ngoài mẫu gộp qua mọi cửa sổ.
        if len(self.sharpe_tung_cua_so) < 2:
            return float("-inf")
        return self.sharpe_gop

    def tom_tat(self) -> dict:
        return {
            "Sharpe gộp ngoài mẫu": self.sharpe_gop,
            "Sharpe trung bình cửa sổ": self.trung_binh,
            "Độ lệch giữa cửa sổ": self.do_lech,
            "Tỷ lệ cửa sổ dương": self.ty_le_thang,
            "Số cửa sổ": len(self.sharpe_tung_cua_so),
        }


def cham_diem_tin_hieu(
    ten: str,
    tham_so: dict,
    signal: pd.Series,
    ds: pd.DataFrame,
    folds: list[Split],
    config: BacktestConfig | None = None,
) -> DiemTienDan:
    # Chấm điểm một chiến lược dựa trên quy tắc: cùng tín hiệu, đánh giá trên mọi cửa sổ.
    diem, gop = {}, []
    for f in folds:
        r = run_backtest(ds.loc[f.test], signal.reindex(f.test), config).returns.iloc[1:]
        diem[f.name] = sharpe_an_toan(r)
        gop.append(r)
    return DiemTienDan(ten, tham_so, pd.Series(diem, name=ten),
                       pd.concat(gop) if gop else pd.Series(dtype=float))


def cham_diem_mo_hinh(
    ten: str,
    tham_so: dict,
    tao_uoc_luong,
    ds: pd.DataFrame,
    features: list[str],
    folds: list[Split],
    horizon: int = 1,
    phan_loai: bool = False,
    nguong: float = 0.0,
    config: BacktestConfig | None = None,
) -> DiemTienDan:
    # Chấm điểm một mô hình có tham số: khớp lại từ đầu ở **từng** cửa sổ.
    cot_nhan = f"y_{horizon}"
    diem, gop = {}, []

    for f in folds:
        scaler = Standardizer().fit(ds.loc[f.train, features])

        X_tr = scaler.transform(ds.loc[f.train, features])
        y_tr = ds.loc[f.train, cot_nhan]
        du = X_tr.notna().all(axis=1) & y_tr.notna()
        if du.sum() < 200:
            continue

        muc_tieu = (y_tr[du] > 0).astype(int) if phan_loai else y_tr[du]
        uoc_luong = tao_uoc_luong(**tham_so)
        uoc_luong.fit(X_tr[du].to_numpy(), muc_tieu.to_numpy())

        X_te = scaler.transform(ds.loc[f.test, features])
        hop_le = X_te.notna().all(axis=1)
        s = pd.Series(np.nan, index=f.test)
        if hop_le.any():
            if phan_loai:
                s[hop_le] = uoc_luong.predict_proba(X_te[hop_le].to_numpy())[:, 1] - 0.5
            else:
                s[hop_le] = uoc_luong.predict(X_te[hop_le].to_numpy())

        tin_hieu = (s > nguong).astype(float).fillna(0.0)
        r = run_backtest(ds.loc[f.test], tin_hieu, config).returns.iloc[1:]
        diem[f.name] = sharpe_an_toan(r)
        gop.append(r)

    return DiemTienDan(ten, tham_so, pd.Series(diem, name=ten),
                       pd.concat(gop) if gop else pd.Series(dtype=float))


def chon_tot_nhat(ung_vien: list[DiemTienDan]) -> DiemTienDan:
    # Chọn cấu hình có điểm ổn định cao nhất qua các cửa sổ cuốn chiếu.
    hop_le = [u for u in ung_vien if np.isfinite(u.diem_chon)]
    if not hop_le:
        raise ValueError("Không cấu hình nào chạy đủ số cửa sổ tối thiểu.")
    return max(hop_le, key=lambda u: u.diem_chon)


def bang_ung_vien(ung_vien: list[DiemTienDan]) -> pd.DataFrame:
    # Bảng đầy đủ mọi cấu hình đã dò — đưa vào phụ lục báo cáo.
    return (
        pd.DataFrame({f"{u.ten} {u.tham_so}": u.tom_tat() for u in ung_vien})
        .T.sort_values("Sharpe gộp ngoài mẫu", ascending=False)
    )
