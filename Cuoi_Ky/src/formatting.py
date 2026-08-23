# Định dạng số dùng chung cho toàn dự án.

import math

_SEP = "\x1f"          # ký tự phân tách tạm khi hoán đổi dấu chấm và dấu phẩy
_MINUS = "−"      # dấu trừ đúng kiểu chữ, dùng cho văn bản tiếng Việt
NA = "—"               # ký hiệu cho giá trị không xác định


def _is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float):
        return math.isnan(x) or math.isinf(x)
    return False


def to_vietnamese(s: str) -> str:
    # Đổi cách viết số sang chuẩn Việt Nam: `1,234.56` → `1.234,56`.
    return s.replace(",", _SEP).replace(".", ",").replace(_SEP, ".").replace("-", _MINUS)


def num(x, digits: int = 2, vi: bool = False) -> str:
    # Số thực có dấu phân cách hàng nghìn.
    if _is_missing(x):
        return NA
    s = f"{x:,.{digits}f}"
    return to_vietnamese(s) if vi else s


def cnt(x, vi: bool = False) -> str:
    # Số nguyên có dấu phân cách hàng nghìn.
    if _is_missing(x):
        return NA
    s = f"{int(round(x)):,d}"
    return to_vietnamese(s) if vi else s


def pct(x, digits: int = 2, vi: bool = False) -> str:
    # Tỷ lệ hiển thị theo phần trăm.
    if _is_missing(x):
        return NA
    s = f"{x * 100:,.{digits}f}"
    return (to_vietnamese(s) if vi else s) + "%"


def money_vnd(x, vi: bool = True) -> str:
    # Số tiền quy về cách đọc quen thuộc: nghìn tỷ / tỷ / triệu đồng.
    if _is_missing(x):
        return NA
    a = abs(x)
    if a >= 1e12:
        return f"{num(x / 1e12, 2, vi)} nghìn tỷ đồng"
    if a >= 1e9:
        return f"{num(x / 1e9, 2, vi)} tỷ đồng"
    if a >= 1e6:
        return f"{num(x / 1e6, 1, vi)} triệu đồng"
    return f"{num(x, 0, vi)} đồng"
