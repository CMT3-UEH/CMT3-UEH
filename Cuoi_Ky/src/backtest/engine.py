# Bộ máy backtest — mọi mô hình trong dự án đều đi qua đúng hàm này.

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import RISK_FREE_ANNUAL, TRADING_DAYS

from .constraints import Constraints, SettlementBook, average_daily_value
from .costs import BASE_COST, CostModel


@dataclass
class BacktestConfig:
    # Toàn bộ giả định của một lần chạy backtest.

    cost: CostModel = BASE_COST
    constraints: Constraints = field(default_factory=Constraints)
    rf_annual: float = RISK_FREE_ANNUAL
    exec_lag: int = 1              # số phiên giữa lúc có tín hiệu và lúc khớp lệnh
    cash_earns_rf: bool = True     # phần vốn đứng ngoài có hưởng lãi hay không
    respect_tradable: bool = True  # bỏ qua phiên trần/sàn không khớp lệnh


@dataclass
class BacktestResult:
    # Kết quả một lần chạy, đủ để dựng mọi bảng và biểu đồ ở phần đánh giá.

    returns: pd.Series             # lợi suất ròng theo phiên
    gross_returns: pd.Series       # lợi suất trước chi phí
    weights: pd.Series             # vị thế thực tế nắm giữ sau khi khớp lệnh
    target: pd.Series              # vị thế mục tiêu trước khi bị ràng buộc cắt
    traded: pd.Series              # phần vị thế thực sự phải khớp lệnh, có dấu
    costs: pd.Series               # chi phí theo phiên, tỷ lệ trên vốn
    blocked: pd.Series             # phiên muốn giao dịch nhưng bị ràng buộc chặn lại
    config: BacktestConfig

    @property
    def nav(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()

    @property
    def turnover(self) -> pd.Series:
        # Độ lớn giao dịch thật mỗi phiên — không tính phần vị thế tự trôi theo giá.
        return self.traded.abs()

    @property
    def n_years(self) -> float:
        return len(self.returns) / TRADING_DAYS

    def trade_days(self) -> pd.Series:
        # Cờ đánh dấu những phiên thực sự có khớp lệnh.
        return self.turnover > 0.0

    def holding_periods(self) -> pd.Series:
        # Phân phối độ dài các lần nắm giữ, tính bằng số phiên.
        trong = self.weights > 1e-6
        doan = (trong != trong.shift(1)).cumsum()
        bang = trong.groupby(doan).agg(["sum", "size"])
        return bang.loc[bang["sum"] > 0, "size"].reset_index(drop=True)

    def tempo(self) -> dict[str, float]:
        # Các đại lượng mô tả nhịp giao dịch mà chiến lược đã chọn.
        so_lan = int(self.trade_days().sum())
        nam = max(self.n_years, 1e-9)
        do_dai = self.holding_periods()
        trong_thi_truong = self.weights > 1e-6

        return {
            "Số lần khớp lệnh": so_lan,
            "Số lần khớp lệnh mỗi năm": so_lan / nam,
            # Nhịp thật: chiến lược có thể ở trong thị trường 100 phiên liền mà vẫn
            # khớp lệnh mỗi ba phiên, vì nó chỉnh kích cỡ chứ không ra vào hẳn.
            "Số phiên giữa hai lệnh": float(len(self.weights) / so_lan) if so_lan else float("nan"),
            "Vòng quay vị thế mỗi năm": float(self.turnover.sum() / nam),
            "Quy mô lệnh trung bình": float(self.turnover[self.turnover > 0].mean())
                                       if so_lan else float("nan"),
            "Số lần vào vị thế": int(len(do_dai)),
            "Thời gian nắm giữ trung bình (phiên)": float(do_dai.mean()) if len(do_dai) else float("nan"),
            "Thời gian nắm giữ trung vị (phiên)": float(do_dai.median()) if len(do_dai) else float("nan"),
            "Tỷ lệ thời gian có vị thế": float(trong_thi_truong.mean()),
            "Tổng chi phí giao dịch": float(self.costs.sum()),
            "Số phiên bị ràng buộc chặn lại": int(self.blocked.sum()),
        }


def daily_rf(rf_annual: float) -> float:
    return (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0


def run_backtest(
    ds: pd.DataFrame,
    signal: pd.Series,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    # Chạy một chiến lược trên bảng dữ liệu đã dựng.
    cfg = config or BacktestConfig()
    rb = cfg.constraints

    idx = ds.index
    o = ds["open"].to_numpy(dtype=float)
    c = ds["close"].to_numpy(dtype=float)

    # Tín hiệu của phiên t chỉ được áp dụng cho phiên t + exec_lag.
    tgt = signal.reindex(idx).shift(cfg.exec_lag).to_numpy(dtype=float)

    if cfg.respect_tradable and "tradable" in ds.columns:
        co_the_khop = ds["tradable"].fillna(False).to_numpy(dtype=bool)
    else:
        co_the_khop = np.ones(len(idx), dtype=bool)

    adv = average_daily_value(ds["close"], ds["volume"]).to_numpy(dtype=float)
    rf_ngay = daily_rf(cfg.rf_annual) if cfg.cash_earns_rf else 0.0

    n = len(idx)
    w_sau = np.zeros(n)
    w_muc_tieu = np.full(n, np.nan)
    da_giao_dich = np.zeros(n)
    r_rong = np.zeros(n)
    r_gop = np.zeros(n)
    phi = np.zeros(n)
    bi_chan = np.zeros(n, dtype=bool)

    w_truoc = 0.0
    so_thanh_toan = SettlementBook(rb.settlement_days)

    for t in range(1, n):
        g = o[t] / c[t - 1]
        k = c[t] / o[t]

        # Tăng trưởng vốn qua đêm và tỷ trọng cổ phiếu sau khi bị giá làm trôi.
        von_qua_dem = w_truoc * g + (1.0 - w_truoc)
        w_troi = (w_truoc * g / von_qua_dem) if von_qua_dem > 0 else w_truoc

        muon = tgt[t]
        w_muc_tieu[t] = muon
        w_moi = w_troi
        khop = 0.0

        if np.isfinite(muon):
            w_mong_muon = rb.clip_weight(muon)
            chenh = w_mong_muon - w_troi

            if abs(chenh) >= rb.min_trade:
                if not co_the_khop[t]:
                    # Phiên khoá trần hoặc khoá sàn: không có lệnh nào khớp được.
                    bi_chan[t] = True
                else:
                    gioi_han = rb.liquidity_cap(adv[t])

                    if rb.enforce_settlement:
                        ban_toi_da, mua_toi_da = so_thanh_toan.allowed(
                            t, w_troi, rb.max_weight
                        )
                        gioi_han = min(
                            gioi_han, mua_toi_da if chenh > 0 else ban_toi_da
                        )

                    if abs(chenh) > gioi_han:
                        chenh = np.sign(chenh) * gioi_han
                        bi_chan[t] = True

                    if abs(chenh) >= rb.min_trade:
                        khop = chenh
                        w_moi = w_troi + chenh
                        so_thanh_toan.record(t, khop)
                    else:
                        # Phần được phép giao dịch nhỏ hơn cả lô giao dịch tối thiểu.
                        bi_chan[t] = True

        von_trong_phien = w_moi * k + (1.0 - w_moi)
        tang_truong = von_qua_dem * von_trong_phien

        # Lãi tiền mặt tính trên phần vốn đứng ngoài sau khi đã khớp lệnh; lệnh khớp
        # ngay đầu phiên nên vị thế mới chiếm gần trọn thời gian của ngày giao dịch.
        r_tien = rf_ngay * (1.0 - w_moi)
        chi_phi = cfg.cost.cost_of(khop)

        r_gop[t] = tang_truong - 1.0 + r_tien
        phi[t] = chi_phi
        r_rong[t] = r_gop[t] - chi_phi
        w_sau[t] = w_moi
        da_giao_dich[t] = khop
        w_truoc = w_moi

    return BacktestResult(
        returns=pd.Series(r_rong, index=idx, name="ret"),
        gross_returns=pd.Series(r_gop, index=idx, name="gross"),
        weights=pd.Series(w_sau, index=idx, name="w"),
        target=pd.Series(w_muc_tieu, index=idx, name="w_target"),
        traded=pd.Series(da_giao_dich, index=idx, name="traded"),
        costs=pd.Series(phi, index=idx, name="cost"),
        blocked=pd.Series(bi_chan, index=idx, name="blocked"),
        config=cfg,
    )


def buy_and_hold(ds: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    # Mốc tham chiếu quan trọng nhất: mua ngay phiên đầu và giữ tới hết kỳ.
    return run_backtest(ds, pd.Series(1.0, index=ds.index), config)
