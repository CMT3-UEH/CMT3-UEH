# Tách dữ liệu chuỗi thời gian chống rò rỉ thông tin.

from dataclasses import dataclass, field
from typing import Iterator, Literal

import numpy as np
import pandas as pd

Mode = Literal["rolling", "expanding"]


class LeakageError(AssertionError):
    # Ném ra khi phát hiện tập huấn luyện chứa thông tin của tập kiểm tra.
    pass


@dataclass(frozen=True)
class Split:
    # Một lần chia dữ liệu.

    name: str
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    valid: pd.DatetimeIndex | None = None
    purged: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    embargoed: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    meta: dict = field(default_factory=dict)

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "valid": len(self.valid) if self.valid is not None else 0,
            "test": len(self.test),
            "purged": len(self.purged),
            "embargoed": len(self.embargoed),
        }


# Nhãn và vùng loại trừ
def label_end_times(index: pd.DatetimeIndex, horizon: int) -> pd.Series:
    # Thời điểm nhãn của mỗi quan sát kết thúc.
    idx = pd.DatetimeIndex(index)
    ends = idx[np.minimum(np.arange(len(idx)) + horizon, len(idx) - 1)]
    return pd.Series(ends, index=idx, name="t1")


def purge_embargo(
    train: pd.DatetimeIndex,
    test: pd.DatetimeIndex,
    t1: pd.Series,
    embargo: int = 5,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    # Trả về (tập huấn luyện đã làm sạch, phần bị thanh lọc, phần bị cách ly).
    if len(test) == 0 or len(train) == 0:
        return train, pd.DatetimeIndex([]), pd.DatetimeIndex([])

    test_start, test_end = test[0], test[-1]

    # Thanh lọc: nhãn của quan sát huấn luyện lấn sang khoảng thời gian kiểm tra
    overlap = train[(t1.reindex(train).values >= test_start) & (train <= test_end)]

    # Cách ly: một số phiên ngay sau tập kiểm tra
    all_after = t1.index[t1.index > test_end]
    banned = all_after[:embargo] if embargo > 0 else pd.DatetimeIndex([])
    embargoed = train.intersection(banned)

    clean = train.difference(overlap).difference(embargoed)
    return clean, overlap, embargoed


def walk_forward(
    index: pd.DatetimeIndex,
    train_span: int = 1260,
    test_span: int = 252,
    step: int | None = None,
    mode: Mode = "rolling",
    horizon: int = 1,
    embargo: int = 5,
) -> Iterator[Split]:
    # Kiểm định cuốn chiếu: huấn luyện trên quá khứ, kiểm tra trên đoạn kế tiếp.
    idx = pd.DatetimeIndex(index).sort_values()
    t1 = label_end_times(idx, horizon)
    step = step or test_span
    start = 0
    fold = 0

    while start + train_span + test_span <= len(idx):
        tr_lo = 0 if mode == "expanding" else start
        train = idx[tr_lo:start + train_span]
        test = idx[start + train_span:start + train_span + test_span]
        clean, purged, embargoed = purge_embargo(train, test, t1, embargo)
        yield Split(name=f"wf_{fold}_{test[0]:%Y%m}", train=clean, test=test,
                    purged=purged, embargoed=embargoed,
                    meta={"mode": mode, "horizon": horizon, "embargo": embargo, "t1": t1})
        start += step
        fold += 1


# Kiểm định và báo cáo
def assert_no_leakage(split: Split, horizon: int = 1) -> None:
    # Kiểm tra một lần chia có bảo đảm tính nhân quả hay không.
    if len(split.train) == 0 or len(split.test) == 0:
        raise LeakageError(f"{split.name}: tập huấn luyện hoặc kiểm tra rỗng.")

    if len(split.train.intersection(split.test)) > 0:
        raise LeakageError(f"{split.name}: tập huấn luyện giao với tập kiểm tra.")

    if split.valid is not None and len(split.valid) > 0:
        if len(split.train.intersection(split.valid)) > 0:
            raise LeakageError(f"{split.name}: tập huấn luyện giao với tập kiểm định.")
        if len(split.valid.intersection(split.test)) > 0:
            raise LeakageError(f"{split.name}: tập kiểm định giao với tập kiểm tra.")

    test_start = split.test[0]
    # Nhãn phải được tính trên lịch giao dịch GỐC. Nếu tính lại trên chỉ mục đã
    # bị thanh lọc thì các phiên bị bỏ sẽ làm lệch vị trí và báo động giả.
    t1 = split.meta.get("t1")
    if t1 is None:
        t1 = label_end_times(split.train.union(split.test).sort_values(), horizon)
    leak = [t for t in split.train if t < test_start and t1.get(t, t) >= test_start]
    if leak:
        raise LeakageError(
            f"{split.name}: {len(leak)} quan sát huấn luyện có nhãn kéo sang tập kiểm tra "
            f"(ví dụ {leak[0]:%d/%m/%Y}). Cần tăng khoảng thanh lọc."
        )
