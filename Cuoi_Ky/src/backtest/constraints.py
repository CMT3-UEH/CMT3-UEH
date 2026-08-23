# Ràng buộc của thị trường Việt Nam mà backtest bắt buộc phải mô phỏng.

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    CAPITAL_VND,
    MAX_ADV_PARTICIPATION,
    PRICE_UNIT_VND,
    SETTLEMENT_DAYS,
)


@dataclass
class Constraints:
    # Tập ràng buộc áp lên mỗi lần đổi vị thế.

    settlement_days: int = SETTLEMENT_DAYS
    max_adv: float | None = MAX_ADV_PARTICIPATION
    capital: float = CAPITAL_VND
    min_weight: float = 0.0
    max_weight: float = 1.0
    enforce_settlement: bool = True

    # Lệnh nhỏ hơn ngưỡng này thì không đặt. Trên thực tế cổ phiếu giao dịch theo lô chẵn
    # nên không thể chỉnh vị thế từng phần nghìn; về mặt kỹ thuật, ngưỡng này còn
    # chặn hiện tượng vị thế trôi theo giá sinh ra một lệnh vá tí hon mỗi phiên.
    min_trade: float = 0.005

    def clip_weight(self, w: float) -> float:
        return float(np.clip(w, self.min_weight, self.max_weight))

    def liquidity_cap(self, adv_vnd: float) -> float:
        # Mức thay đổi vị thế tối đa cho phép bởi ràng buộc thanh khoản.
        if self.max_adv is None or not np.isfinite(adv_vnd) or adv_vnd <= 0:
            return np.inf
        return self.max_adv * adv_vnd / self.capital


def average_daily_value(
    close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    # Giá trị giao dịch bình quân ngày của `window` phiên gần nhất, quy ra đồng.
    gia_tri = close * volume * PRICE_UNIT_VND
    return gia_tri.rolling(window).mean().shift(1)


@dataclass
class SettlementBook:
    # Theo dõi phần cổ phiếu và tiền chưa về theo chu kỳ T+2,5.

    settlement_days: int = SETTLEMENT_DAYS
    _cho: list[tuple[int, float]] = None      # (phiên tiền/cổ phiếu về, lượng có dấu)

    def __post_init__(self) -> None:
        self._cho = []

    def _thanh_ly(self, phien: int) -> None:
        self._cho = [(ngay, x) for ngay, x in self._cho if ngay > phien]

    def locked(self, phien: int) -> tuple[float, float]:
        # Trả về (cổ phiếu chưa về, tiền chưa về) tại phiên đang xét.
        self._thanh_ly(phien)
        co_phieu = sum(x for _, x in self._cho if x > 0)
        tien = sum(-x for _, x in self._cho if x < 0)
        return co_phieu, tien

    def allowed(self, phien: int, w_hien_tai: float, max_weight: float) -> tuple[float, float]:
        # Khoảng thay đổi vị thế được phép: (mức bán tối đa, mức mua tối đa).
        cp_khoa, tien_khoa = self.locked(phien)
        ban_toi_da = max(0.0, w_hien_tai - cp_khoa)
        mua_toi_da = max(0.0, (max_weight - w_hien_tai) - tien_khoa)
        return ban_toi_da, mua_toi_da

    def record(self, phien: int, delta_w: float) -> None:
        # Ghi nhận một lệnh vừa khớp; nó sẽ về tài khoản sau `settlement_days` phiên.
        if delta_w != 0.0:
            self._cho.append((phien + self.settlement_days, delta_w))
