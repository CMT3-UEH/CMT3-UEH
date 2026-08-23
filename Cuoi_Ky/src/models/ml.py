# Baseline có tham số ước lượng: hồi quy tuyến tính, phân loại, cây và LightGBM.

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FittedModel:
    # Mô hình đã ước lượng, kèm chuỗi điểm dự báo trên toàn chỉ mục.

    name: str
    scores: pd.Series
    model: object = None
    importance: pd.Series = None


def _xy(d, part: str):
    idx = d.usable(part)
    X = d.X("all").loc[idx]
    y = d.y("all").loc[idx]
    return X, y, idx


def fit_predict(uoc_luong, d, name: str, phan_loai: bool = False) -> FittedModel:
    # Ước lượng trên tập huấn luyện rồi dự báo trên toàn chuỗi.
    X_tr, y_tr, _ = _xy(d, "train")
    muc_tieu = (y_tr > 0).astype(int) if phan_loai else y_tr
    uoc_luong.fit(X_tr.to_numpy(), muc_tieu.to_numpy())

    X_all = d.X("all")
    hop_le = X_all.notna().all(axis=1)
    diem = pd.Series(np.nan, index=X_all.index, name=name)

    if phan_loai:
        p = uoc_luong.predict_proba(X_all[hop_le].to_numpy())[:, 1]
        diem[hop_le] = p - 0.5
    else:
        diem[hop_le] = uoc_luong.predict(X_all[hop_le].to_numpy())

    return FittedModel(name=name, scores=diem, model=uoc_luong,
                       importance=_do_quan_trong(uoc_luong, list(X_all.columns)))


def _do_quan_trong(uoc_luong, ten_cot: list[str]) -> pd.Series | None:
    # Trích độ quan trọng đặc trưng nếu mô hình có cung cấp.
    for thuoc_tinh in ("feature_importances_", "coef_"):
        gia_tri = getattr(uoc_luong, thuoc_tinh, None)
        if gia_tri is None:
            continue
        v = np.asarray(gia_tri).ravel()
        if len(v) == len(ten_cot):
            return pd.Series(np.abs(v), index=ten_cot).sort_values(ascending=False)
    return None


# Hồi quy tuyến tính và phân loại
def hoi_quy_tuyen_tinh(alpha: float = 1.0, kieu: str = "ridge"):
    # A10 — OLS, Ridge hoặc Lasso trên toàn bộ bộ đặc trưng.
    from sklearn.linear_model import Lasso, LinearRegression, Ridge

    if kieu == "ols":
        return LinearRegression()
    if kieu == "ridge":
        return Ridge(alpha=alpha, random_state=None)
    if kieu == "lasso":
        return Lasso(alpha=alpha, max_iter=20_000)
    raise ValueError(f"Kiểu hồi quy không hợp lệ: {kieu}")


def phan_loai_logistic(C: float = 0.01):
    # A11 — hồi quy logistic phân loại dấu của lợi suất.
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=C, max_iter=5_000, solver="lbfgs")


def apt_da_nhan_to(d, ds: pd.DataFrame, name: str = "A9 · Hồi quy đa nhân tố APT") -> FittedModel:
    # A9 — hồi quy lợi suất theo nhân tố thị trường và nhân tố vĩ mô.
    from sklearn.linear_model import LinearRegression

    nhan_to = [c for c in ds.columns if c.startswith("mac_") and c.endswith("_r1")]
    nhan_to += [c for c in ("rel_excess_1", "rel_beta_60") if c in ds.columns]
    if not nhan_to:
        raise KeyError("Không tìm thấy cột nhân tố vĩ mô trong bảng dữ liệu")

    idx_tr = d.usable("train")
    X_tr = ds.loc[idx_tr, nhan_to]
    y_tr = d.y("all").loc[idx_tr]

    mo_hinh = LinearRegression().fit(X_tr.to_numpy(), y_tr.to_numpy())

    X_all = ds[nhan_to]
    hop_le = X_all.notna().all(axis=1)
    diem = pd.Series(np.nan, index=ds.index, name=name)
    diem[hop_le] = mo_hinh.predict(X_all[hop_le].to_numpy())

    return FittedModel(name=name, scores=diem, model=mo_hinh,
                       importance=pd.Series(np.abs(mo_hinh.coef_), index=nhan_to)
                       .sort_values(ascending=False))


# Cây và tập hợp cây
def rung_ngau_nhien(n_cay: int = 400, sau: int = 4, seed: int = 0):
    # B1a — rừng ngẫu nhiên, giới hạn độ sâu để không học thuộc nhiễu.
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=n_cay,
        max_depth=sau,
        min_samples_leaf=50,
        max_features=0.3,
        n_jobs=-1,
        random_state=seed,
    )


def lightgbm(
    n_cay: int = 300, sau: int = 3, hoc: float = 0.02, seed: int = 0
):
    # B1b — LightGBM, đối thủ mạnh nhất trên dữ liệu bảng.
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=n_cay,
        max_depth=sau,
        num_leaves=2 ** sau,
        learning_rate=hoc,
        min_child_samples=60,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=10.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


# Lưới cấu hình dò trên tập kiểm định
LUOI_MAY_HOC = {
    "A10 · Ridge": [{"kieu": "ridge", "alpha": a} for a in (10.0, 100.0, 1000.0)],
    "A10 · Lasso": [{"kieu": "lasso", "alpha": a} for a in (1e-5, 1e-4, 1e-3)],
    "A10 · OLS": [{"kieu": "ols"}],
    "A11 · Logistic": [{"C": c} for c in (0.003, 0.01, 0.1)],
    "B1a · Rừng ngẫu nhiên": [{"sau": s} for s in (3, 4, 6)],
    "B1b · LightGBM": [{"sau": s, "hoc": h} for s in (2, 3) for h in (0.01, 0.03)],
}


def dem_cau_hinh() -> int:
    return sum(len(v) for v in LUOI_MAY_HOC.values())
