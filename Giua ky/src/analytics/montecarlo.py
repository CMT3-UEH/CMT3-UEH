"""Mô phỏng Monte Carlo cho giá cổ phiếu và giá trị danh mục."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS


@dataclass
class SimulationResult:
    paths: np.ndarray                  # ma trận (n_days+1, n_sims)
    terminal: np.ndarray               # giá trị cuối kỳ của từng kịch bản
    s0: float
    horizon: int
    method: str
    percentiles: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def stats(self, level: float = 0.95) -> dict[str, float]:
        ret = self.terminal / self.s0 - 1.0
        var = float(np.percentile(ret, (1 - level) * 100))
        tail = ret[ret <= var]
        return {
            "Giá kỳ vọng": float(self.terminal.mean()),
            "Trung vị": float(np.median(self.terminal)),
            "Phân vị 5%": float(np.percentile(self.terminal, 5)),
            "Phân vị 95%": float(np.percentile(self.terminal, 95)),
            "Lợi suất kỳ vọng": float(ret.mean()),
            "Xác suất lỗ": float((ret < 0).mean()),
            f"VaR {level:.0%} (toàn kỳ)": var,
            f"CVaR {level:.0%} (toàn kỳ)": float(tail.mean()) if len(tail) else np.nan,
            "Lỗ tệ nhất": float(ret.min()),
            "Lãi tốt nhất": float(ret.max()),
        }

    def prob_above(self, price: float) -> float:
        return float((self.terminal >= price).mean())


def _percentile_frame(paths: np.ndarray) -> pd.DataFrame:
    qs = [5, 25, 50, 75, 95]
    data = {f"p{q}": np.percentile(paths, q, axis=1) for q in qs}
    data["mean"] = paths.mean(axis=1)
    return pd.DataFrame(data)


def simulate_gbm(
    s0: float,
    mu_daily: float,
    sigma_daily: float,
    horizon: int = TRADING_DAYS,
    n_sims: int = 10_000,
    seed: int = 42,
) -> SimulationResult:
    """S_t = S_0 * exp((mu - sigma^2/2) t + sigma * W_t)."""
    rng = np.random.default_rng(seed)
    drift = mu_daily - 0.5 * sigma_daily ** 2
    shocks = rng.normal(0.0, sigma_daily, size=(horizon, n_sims))
    log_paths = np.vstack([np.zeros((1, n_sims)), np.cumsum(drift + shocks, axis=0)])
    paths = s0 * np.exp(log_paths)
    return SimulationResult(
        paths, paths[-1], s0, horizon, "GBM (phân phối chuẩn)", _percentile_frame(paths)
    )


def simulate_student_t(
    s0: float,
    mu_daily: float,
    sigma_daily: float,
    df: float = 4.0,
    horizon: int = TRADING_DAYS,
    n_sims: int = 10_000,
    seed: int = 42,
) -> SimulationResult:
    """Như GBM nhưng cú sốc theo phân phối Student-t (đuôi dày hơn)."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_t(df, size=(horizon, n_sims))
    raw = raw / np.sqrt(df / (df - 2))          # chuẩn hoá về phương sai 1
    shocks = raw * sigma_daily
    drift = mu_daily - 0.5 * sigma_daily ** 2
    log_paths = np.vstack([np.zeros((1, n_sims)), np.cumsum(drift + shocks, axis=0)])
    paths = s0 * np.exp(log_paths)
    return SimulationResult(
        paths, paths[-1], s0, horizon, f"Student-t (df={df:g})", _percentile_frame(paths)
    )


def simulate_bootstrap(
    s0: float,
    historical_log_returns: np.ndarray,
    horizon: int = TRADING_DAYS,
    n_sims: int = 10_000,
    block: int = 1,
    seed: int = 42,
) -> SimulationResult:
    """Lấy mẫu có hoàn lại từ lợi suất lịch sử."""
    rng = np.random.default_rng(seed)
    r = np.asarray(historical_log_returns, dtype=float)
    r = r[np.isfinite(r)]

    if block <= 1:
        draws = rng.choice(r, size=(horizon, n_sims), replace=True)
    else:
        n_blocks = int(np.ceil(horizon / block))
        starts = rng.integers(0, max(1, len(r) - block), size=(n_blocks, n_sims))
        pieces = np.stack([r[s: s + block] for s in starts.ravel()])
        draws = pieces.reshape(n_blocks, n_sims, block).transpose(0, 2, 1)
        draws = draws.reshape(n_blocks * block, n_sims)[:horizon]

    log_paths = np.vstack([np.zeros((1, n_sims)), np.cumsum(draws, axis=0)])
    paths = s0 * np.exp(log_paths)
    label = "Bootstrap lịch sử" + (f" (block {block})" if block > 1 else "")
    return SimulationResult(paths, paths[-1], s0, horizon, label, _percentile_frame(paths))


def compare_methods(
    s0: float,
    log_returns: pd.Series,
    horizon: int = TRADING_DAYS,
    n_sims: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bảng so sánh ba phương pháp mô phỏng trên cùng dữ liệu đầu vào."""
    mu, sd = float(log_returns.mean()), float(log_returns.std(ddof=1))
    sims = [
        simulate_gbm(s0, mu, sd, horizon, n_sims, seed),
        simulate_student_t(s0, mu, sd, 4.0, horizon, n_sims, seed),
        simulate_bootstrap(s0, log_returns.values, horizon, n_sims, 5, seed),
    ]
    return pd.DataFrame({s.method: s.stats() for s in sims})

