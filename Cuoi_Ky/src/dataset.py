# Lớp dữ liệu dùng chung cho mọi mô hình trong dự án.

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import EMBARGO, TRAIN_END, VALID_END
from src.features.builder import feature_columns
from src.features.labeling import label_end_time
from src.splits import Split, assert_no_leakage, purge_embargo

PARTS = ("train", "valid", "test")


def date_split(
    index: pd.DatetimeIndex,
    t1: pd.Series,
    train_end: str = TRAIN_END,
    valid_end: str = VALID_END,
    embargo: int = EMBARGO,
    name: str = "holdout",
) -> Split:
    # Chia ba tập theo mốc thời gian, có thanh lọc và cách ly ở hai ranh giới.
    idx = pd.DatetimeIndex(index).sort_values()
    train = idx[idx <= pd.Timestamp(train_end)]
    valid = idx[(idx > pd.Timestamp(train_end)) & (idx <= pd.Timestamp(valid_end))]
    test = idx[idx > pd.Timestamp(valid_end)]

    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(
            f"Mốc chia không hợp lệ: train={len(train)}, valid={len(valid)}, test={len(test)}"
        )

    train, purge1, emb1 = purge_embargo(train, valid, t1, embargo)
    train, purge2, emb2 = purge_embargo(train, test, t1, embargo)
    valid, purge3, emb3 = purge_embargo(valid, test, t1, embargo)

    # Thanh lọc chỉ xử lý nhãn chồng lấn, không xử lý tự tương quan của lợi suất.
    # Vì vậy cắt thêm vùng đệm ngay trước mỗi ranh giới.
    dem = pd.DatetimeIndex([])
    if embargo > 0:
        for i, khoi in enumerate((train, valid)):
            if len(khoi) > embargo:
                dem = dem.union(khoi[-embargo:])
        train = train.difference(dem)
        valid = valid.difference(dem)

    return Split(
        name=name,
        train=train,
        valid=valid,
        test=test,
        purged=purge1.union(purge2).union(purge3),
        embargoed=emb1.union(emb2).union(emb3).union(dem),
        meta={"embargo": embargo, "t1": t1,
              "moc": (train_end, valid_end)},
    )


@dataclass
class Standardizer:
    # Chuẩn hoá đặc trưng bằng trung vị và khoảng tứ phân vị của tập huấn luyện.

    center: pd.Series = field(default=None)
    scale: pd.Series = field(default=None)
    lo: pd.Series = field(default=None)
    hi: pd.Series = field(default=None)
    clip_quantile: float = 0.01

    def fit(self, X: pd.DataFrame) -> "Standardizer":
        self.lo = X.quantile(self.clip_quantile)
        self.hi = X.quantile(1.0 - self.clip_quantile)
        cat = X.clip(self.lo, self.hi, axis=1)
        self.center = cat.median()
        khoang = cat.quantile(0.75) - cat.quantile(0.25)
        self.scale = khoang.replace(0.0, np.nan).fillna(cat.std(ddof=1)).replace(0.0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.center is None:
            raise RuntimeError("Phải gọi fit trên tập huấn luyện trước.")
        cat = X.clip(self.lo, self.hi, axis=1)
        return ((cat - self.center) / self.scale).astype(float)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


@dataclass
class Dataset:
    # Bảng dữ liệu đã chia, kèm mọi thứ mô hình cần và không kèm gì thừa.

    ds: pd.DataFrame
    features: list[str]
    split: Split
    horizon: int
    scaler: Standardizer

    def index(self, part: str) -> pd.DatetimeIndex:
        if part == "train":
            return self.split.train
        if part == "valid":
            return self.split.valid
        if part == "test":
            return self.split.test
        if part == "all":
            return self.ds.index
        raise ValueError(f"Phần dữ liệu không hợp lệ: {part}")

    def X(self, part: str = "all", scaled: bool = True) -> pd.DataFrame:
        raw = self.ds.loc[self.index(part), self.features]
        return self.scaler.transform(raw) if scaled else raw

    def y(self, part: str = "all") -> pd.Series:
        return self.ds.loc[self.index(part), f"y_{self.horizon}"]

    def prices(self, part: str = "all") -> pd.DataFrame:
        cot = ["open", "high", "low", "close", "volume", "benchmark_close", "tradable"]
        return self.ds.loc[self.index(part), [c for c in cot if c in self.ds.columns]]

    def usable(self, part: str) -> pd.DatetimeIndex:
        # Chỉ mục của những phiên có đủ cả đặc trưng lẫn nhãn để huấn luyện.
        idx = self.index(part)
        con = self.ds.loc[idx, self.features + [f"y_{self.horizon}"]].notna().all(axis=1)
        return idx[con.to_numpy()]

    def report(self) -> pd.DataFrame:
        # Bảng mô tả lần chia — đưa thẳng vào báo cáo và lên dashboard.
        hang = {}
        tong = sum(self.split.sizes.values())
        for ten in PARTS:
            idx = self.index(ten)
            hang[ten] = {
                "Số phiên": len(idx),
                "Tỷ lệ": len(idx) / max(tong, 1),
                "Từ": f"{idx.min():%d/%m/%Y}" if len(idx) else "—",
                "Đến": f"{idx.max():%d/%m/%Y}" if len(idx) else "—",
            }
        hang["đã thanh lọc"] = {"Số phiên": len(self.split.purged), "Tỷ lệ": np.nan,
                                "Từ": "—", "Đến": "—"}
        hang["đã cách ly"] = {"Số phiên": len(self.split.embargoed), "Tỷ lệ": np.nan,
                              "Từ": "—", "Đến": "—"}
        return pd.DataFrame(hang).T


def make_dataset(
    ds: pd.DataFrame,
    horizon: int = 1,
    embargo: int = EMBARGO,
    train_end: str = TRAIN_END,
    valid_end: str = VALID_END,
) -> Dataset:
    # Dựng đối tượng `Dataset` và kiểm tra ngay rằng lần chia không rò rỉ.
    cot = feature_columns(ds)

    # Dùng đúng t1 của nhãn đang học: nhãn tầm dự báo 20 phiên cần vùng thanh lọc
    # rộng hơn nhiều so với nhãn một phiên.
    t1 = label_end_time(ds.index, horizon)
    split = date_split(ds.index, t1, train_end, valid_end, embargo,
                       name=f"holdout_h{horizon}")

    # Nhãn ở đây mở vị thế tại t+1, nên tính nhân quả phải kiểm với tầm dự báo h+1.
    assert_no_leakage(split, horizon=horizon + 1)

    scaler = Standardizer().fit(ds.loc[split.train, cot])
    return Dataset(ds=ds, features=cot, split=split, horizon=horizon, scaler=scaler)
