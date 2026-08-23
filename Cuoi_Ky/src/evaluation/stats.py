# Kiểm định thống kê cho tỷ số Sharpe.

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import TRADING_DAYS

EULER = 0.577215664901532_9


def _sharpe_moi_phien(r: np.ndarray, rf_moi_phien: float = 0.0) -> float:
    # Sharpe tính theo đơn vị một phiên, chưa thường niên hoá.
    thua = r - rf_moi_phien
    sd = thua.std(ddof=1)
    return float(thua.mean() / sd) if sd > 0 else float("nan")


def sharpe_std_error(returns: pd.Series, hieu_chinh_tu_tuong_quan: bool = True) -> float:
    # Sai số chuẩn của Sharpe thường niên hoá, theo Lo (2002).
    r = np.asarray(returns.dropna(), dtype=float)
    n = len(r)
    if n < 30:
        return float("nan")

    sr = _sharpe_moi_phien(r)
    if not np.isfinite(sr):
        return float("nan")

    if hieu_chinh_tu_tuong_quan:
        lech = stats.skew(r, bias=False)
        nhon = stats.kurtosis(r, fisher=False, bias=False)
        phuong_sai = (1.0 - lech * sr + (nhon - 1.0) / 4.0 * sr ** 2) / (n - 1)
    else:
        phuong_sai = (1.0 + 0.5 * sr ** 2) / (n - 1)

    return float(np.sqrt(max(phuong_sai, 0.0)) * np.sqrt(TRADING_DAYS))


def probabilistic_sharpe(returns: pd.Series, nguong_nam: float = 0.0) -> float:
    # Xác suất Sharpe thật vượt `nguong_nam` (Bailey & López de Prado, 2014).
    r = np.asarray(returns.dropna(), dtype=float)
    n = len(r)
    if n < 30:
        return float("nan")

    sr = _sharpe_moi_phien(r)
    sr_nguong = nguong_nam / np.sqrt(TRADING_DAYS)
    lech = stats.skew(r, bias=False)
    nhon = stats.kurtosis(r, fisher=False, bias=False)

    mau = np.sqrt(max(1.0 - lech * sr + (nhon - 1.0) / 4.0 * sr ** 2, 1e-12))
    z = (sr - sr_nguong) * np.sqrt(n - 1) / mau
    return float(stats.norm.cdf(z))


def sharpe_ky_vong_toi_da(so_thu: int, do_lech_sharpe: float) -> float:
    # Sharpe cao nhất kỳ vọng thu được khi thử `so_thu` cấu hình vô dụng.
    if so_thu < 2 or not np.isfinite(do_lech_sharpe) or do_lech_sharpe <= 0:
        return 0.0
    a = stats.norm.ppf(1.0 - 1.0 / so_thu)
    b = stats.norm.ppf(1.0 - 1.0 / (so_thu * np.e))
    return float(do_lech_sharpe * ((1.0 - EULER) * a + EULER * b))


def deflated_sharpe(
    returns: pd.Series, so_thu: int, do_lech_sharpe: float
) -> tuple[float, float]:
    # Deflated Sharpe Ratio — trả về (xác suất, ngưỡng đã trừ).
    nguong = sharpe_ky_vong_toi_da(so_thu, do_lech_sharpe)
    return probabilistic_sharpe(returns, nguong), nguong


def block_bootstrap_ci(
    returns: pd.Series,
    ham=None,
    do_dai_khoi: int = 21,
    so_lan: int = 2_000,
    muc: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    # Khoảng tin cậy bằng bootstrap theo khối, giữ nguyên tự tương quan ngắn hạn.
    from src.evaluation.metrics import sharpe_ratio

    ham = ham or (lambda x: sharpe_ratio(pd.Series(x)))
    r = np.asarray(returns.dropna(), dtype=float)
    n = len(r)
    if n < do_dai_khoi * 4:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    so_khoi = int(np.ceil(n / do_dai_khoi))
    mau = np.empty(so_lan)

    for i in range(so_lan):
        dau = rng.integers(0, n - do_dai_khoi, size=so_khoi)
        ghep = np.concatenate([r[d:d + do_dai_khoi] for d in dau])[:n]
        mau[i] = ham(ghep)

    lo = float(np.nanpercentile(mau, (1 - muc) / 2 * 100))
    hi = float(np.nanpercentile(mau, (1 + muc) / 2 * 100))
    return float(ham(r)), lo, hi


@dataclass
class KetQuaPBO:
    # Kết quả kiểm định xác suất quá khớp khi backtest.

    pbo: float
    so_to_hop: int
    xep_hang_oos: np.ndarray

    @property
    def dien_giai(self) -> str:
        if not np.isfinite(self.pbo):
            return "không đủ dữ liệu để kết luận"
        if self.pbo < 0.20:
            return "quy trình chọn mô hình đáng tin"
        if self.pbo < 0.50:
            return "có dấu hiệu quá khớp, cần thận trọng"
        return "quy trình chọn mô hình gần như chỉ đang chọn nhiễu"


def pbo(bang_loi_suat: pd.DataFrame, so_khoi: int = 10) -> KetQuaPBO:
    # Xác suất quá khớp khi backtest, theo phương pháp CSCV.
    from src.evaluation.metrics import sharpe_ratio

    X = bang_loi_suat.dropna(how="any")
    n_cau_hinh = X.shape[1]
    if n_cau_hinh < 2 or len(X) < so_khoi * 20:
        return KetQuaPBO(float("nan"), 0, np.array([]))

    if so_khoi % 2:
        so_khoi -= 1
    khoi = np.array_split(np.arange(len(X)), so_khoi)

    xep_hang = []
    for trong_mau in combinations(range(so_khoi), so_khoi // 2):
        ngoai_mau = [k for k in range(so_khoi) if k not in trong_mau]
        i_in = np.concatenate([khoi[k] for k in trong_mau])
        i_out = np.concatenate([khoi[k] for k in ngoai_mau])

        sr_in = X.iloc[i_in].apply(sharpe_ratio)
        sr_out = X.iloc[i_out].apply(sharpe_ratio)
        if sr_in.isna().all() or sr_out.isna().all():
            continue

        thang = sr_in.idxmax()
        # Thứ hạng tương đối ngoài mẫu: 1 là tốt nhất, 0 là kém nhất.
        r = sr_out.rank(pct=True)[thang]
        xep_hang.append(float(r))

    if not xep_hang:
        return KetQuaPBO(float("nan"), 0, np.array([]))

    xep_hang = np.array(xep_hang)
    return KetQuaPBO(float((xep_hang < 0.5).mean()), len(xep_hang), xep_hang)


def reality_check(
    bang_loi_suat: pd.DataFrame,
    moc: pd.Series,
    do_dai_khoi: int = 21,
    so_lan: int = 2_000,
    seed: int = 0,
) -> tuple[float, float]:
    # Kiểm định White's Reality Check — trả về (thống kê quan sát, p-value).
    chung = bang_loi_suat.dropna(how="any").index.intersection(moc.dropna().index)
    if len(chung) < do_dai_khoi * 4:
        return float("nan"), float("nan")

    vuot = bang_loi_suat.loc[chung].sub(moc.loc[chung], axis=0).to_numpy()
    n, k = vuot.shape
    quan_sat = np.sqrt(n) * vuot.mean(axis=0).max()

    rng = np.random.default_rng(seed)
    so_khoi = int(np.ceil(n / do_dai_khoi))
    mau = np.empty(so_lan)
    trung_binh = vuot.mean(axis=0)

    for i in range(so_lan):
        dau = rng.integers(0, n - do_dai_khoi, size=so_khoi)
        chi_muc = np.concatenate([np.arange(d, d + do_dai_khoi) for d in dau])[:n]
        # Trừ trung bình mẫu để áp đặt giả thuyết không lên phân phối bootstrap.
        mau[i] = np.sqrt(n) * (vuot[chi_muc].mean(axis=0) - trung_binh).max()

    return float(quan_sat), float((mau >= quan_sat).mean())


def bang_do_tin_cay(
    ket_qua: dict[str, pd.Series],
    so_cau_hinh_da_do: int,
    do_lech_sharpe: float | None = None,
) -> pd.DataFrame:
    # Bảng độ tin cậy cho nhiều chiến lược — dùng trực tiếp trong báo cáo.
    from src.evaluation.metrics import sharpe_ratio

    sharpes = {k: sharpe_ratio(v) for k, v in ket_qua.items()}
    if do_lech_sharpe is None:
        do_lech_sharpe = float(np.std(list(sharpes.values()), ddof=1)) \
            if len(sharpes) > 1 else 0.0

    hang = {}
    for ten, r in ket_qua.items():
        sr, lo, hi = block_bootstrap_ci(r)
        dsr, nguong = deflated_sharpe(r, so_cau_hinh_da_do, do_lech_sharpe)
        hang[ten] = {
            "Sharpe": sharpes[ten],
            "Sai số chuẩn": sharpe_std_error(r),
            "KTC 95% dưới": lo,
            "KTC 95% trên": hi,
            "P(Sharpe > 0)": probabilistic_sharpe(r, 0.0),
            "Ngưỡng do dò tìm": nguong,
            "Deflated Sharpe": dsr,
        }
    return pd.DataFrame(hang).T.sort_values("Sharpe", ascending=False)
