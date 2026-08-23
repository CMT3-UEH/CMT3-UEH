# Phân rã nguồn gốc lợi nhuận và hồi quy alpha — phần biện luận đầu tư.

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS
from src.evaluation.metrics import annual_return, daily_rf, sharpe_ratio


@dataclass
class KetQuaHoiQuy:
    # Kết quả một hồi quy nhân tố, đã dùng sai số chuẩn vững.

    alpha_nam: float
    alpha_t: float
    alpha_p: float
    he_so: pd.Series
    t_stat: pd.Series
    p_value: pd.Series
    r2: float
    so_quan_sat: int

    def tom_tat(self) -> dict:
        return {
            "Alpha quy năm": self.alpha_nam,
            "t-stat của alpha": self.alpha_t,
            "p-value của alpha": self.alpha_p,
            "R²": self.r2,
            "Số quan sát": self.so_quan_sat,
        }


def hoi_quy_nhan_to(
    loi_suat: pd.Series, nhan_to: pd.DataFrame, rf_annual: float | None = None
) -> KetQuaHoiQuy:
    # Hồi quy lợi suất vượt trội theo các nhân tố, sai số chuẩn Newey–West.
    import statsmodels.api as sm

    chung = loi_suat.dropna().index.intersection(nhan_to.dropna().index)
    y = loi_suat.loc[chung].astype(float)
    X = nhan_to.loc[chung].astype(float)

    if rf_annual is not None:
        y = y - daily_rf(rf_annual)

    X = sm.add_constant(X, has_constant="add")
    do_tre = int(np.ceil(4 * (len(y) / 100.0) ** (2.0 / 9.0)))
    mo_hinh = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": do_tre})

    return KetQuaHoiQuy(
        alpha_nam=float(mo_hinh.params["const"]) * TRADING_DAYS,
        alpha_t=float(mo_hinh.tvalues["const"]),
        alpha_p=float(mo_hinh.pvalues["const"]),
        he_so=mo_hinh.params.drop("const"),
        t_stat=mo_hinh.tvalues.drop("const"),
        p_value=mo_hinh.pvalues.drop("const"),
        r2=float(mo_hinh.rsquared),
        so_quan_sat=int(mo_hinh.nobs),
    )


# Tên đọc được cho các nhân tố, thay cho mã cột thô trong bảng đặc trưng. Bảng hồi
# quy là thứ hội đồng đọc nên không được để lộ tên biến nội bộ ra ngoài.
TEN_NHAN_TO = {
    "mac_USDVND_r1": "tỷ giá USD/VND",
    "mac_OIL_r1": "dầu WTI",
    "mac_GOLD_r1": "vàng",
    "mac_SP500_r1": "S&P 500",
    "rel_excess_1": "lợi suất vượt trội",
    "rel_beta_60": "beta trượt 60 phiên",
}


def dat_ten_nhan_to(bang: pd.DataFrame) -> pd.DataFrame:
    # Đổi mã cột đặc trưng sang tên đọc được.
    return bang.rename(columns=TEN_NHAN_TO)


def bang_alpha(
    loi_suat: pd.Series,
    loi_suat_thi_truong: pd.Series,
    loi_suat_co_phieu: pd.Series,
    nhan_to_vi_mo: pd.DataFrame | None = None,
    rf_annual: float | None = None,
) -> pd.DataFrame:
    # Ba đặc tả hồi quy lồng nhau: CAPM, CAPM cộng chính cổ phiếu, rồi APT đa nhân tố.
    mo_hinh = {
        "CAPM (thị trường)": pd.DataFrame({"thị trường": loi_suat_thi_truong}),
        "CAPM mở rộng với lợi suất FPT": pd.DataFrame({
            "thị trường": loi_suat_thi_truong, "FPT": loi_suat_co_phieu,
        }),
    }
    if nhan_to_vi_mo is not None and not nhan_to_vi_mo.empty:
        mo_hinh["APT đa nhân tố"] = pd.concat(
            [pd.DataFrame({"thị trường": loi_suat_thi_truong,
                           "FPT": loi_suat_co_phieu}), nhan_to_vi_mo], axis=1
        )

    hang = {}
    for ten, X in mo_hinh.items():
        kq = hoi_quy_nhan_to(loi_suat, X, rf_annual)
        d = kq.tom_tat()
        for bien in X.columns:
            d[f"β {TEN_NHAN_TO.get(bien, bien)}"] = kq.he_so.get(bien, np.nan)
        hang[ten] = d
    return dat_ten_nhan_to(pd.DataFrame(hang).T)


def phan_ra_vao_ra(
    loi_suat_chien_luoc: pd.Series,
    loi_suat_tai_san: pd.Series,
    vi_the: pd.Series,
) -> pd.DataFrame:
    # Tách giá trị đến từ 'chọn đúng lúc vào' và từ 'tránh đúng lúc ra'.
    chung = loi_suat_tai_san.dropna().index.intersection(vi_the.dropna().index)
    r = loi_suat_tai_san.loc[chung]
    w = vi_the.loc[chung]

    ngoai = 1.0 - w
    tranh_duoc = float((ngoai * r.clip(upper=0.0)).sum())      # phần lỗ đã né được
    bo_lo = float((ngoai * r.clip(lower=0.0)).sum())           # phần lãi đã bỏ lỡ

    return pd.DataFrame({
        "Giá trị": {
            "Lỗ tránh được nhờ đứng ngoài": -tranh_duoc,
            "Lãi bỏ lỡ do đứng ngoài": -bo_lo,
            "Chênh lệch ròng của việc định thời điểm": -tranh_duoc - bo_lo,
            "Tỷ lệ thời gian đứng ngoài": float(ngoai.mean()),
        }
    })


def bang_theo_nam(loi_suat: pd.Series, moc: pd.Series | None = None) -> pd.DataFrame:
    # Hiệu quả theo từng năm — phát hiện ngay trường hợp một năm gánh cả kỳ.
    hang = {}
    for nam, r in loi_suat.groupby(loi_suat.index.year):
        d = {
            "Số phiên": len(r),
            "Lợi suất": float((1 + r).prod() - 1),
            "Sharpe": sharpe_ratio(r),
        }
        if moc is not None:
            m = moc.loc[moc.index.year == nam]
            d["Mốc"] = float((1 + m).prod() - 1)
            d["Vượt mốc"] = d["Lợi suất"] - d["Mốc"]
        hang[nam] = d
    return pd.DataFrame(hang).T


def bang_theo_che_do(
    loi_suat: pd.Series, che_do: pd.Series, moc: pd.Series | None = None
) -> pd.DataFrame:
    # Hiệu quả theo chế độ thị trường: tăng, giảm, đi ngang.
    chung = loi_suat.dropna().index.intersection(che_do.dropna().index)
    r, c = loi_suat.loc[chung], che_do.loc[chung]

    hang = {}
    for ten, phan in r.groupby(c):
        d = {
            "Số phiên": len(phan),
            "Lợi suất quy năm": annual_return(phan),
            "Sharpe": sharpe_ratio(phan),
        }
        if moc is not None:
            m = moc.loc[phan.index]
            d["Mốc quy năm"] = annual_return(m)
            d["Vượt mốc"] = d["Lợi suất quy năm"] - d["Mốc quy năm"]
        hang[ten] = d
    return pd.DataFrame(hang).T


def gan_nhan_che_do(ds: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    # Gán nhãn chế độ thị trường bằng đặc trưng đã có, không ước lượng gì thêm.
    tren_ma = ds.loc[index, "reg_mkt_above_ma200"] > 0
    sut = ds.loc[index, "reg_mkt_dd"] < -0.10
    nhan = pd.Series("đi ngang", index=index)
    nhan[tren_ma & ~sut] = "tăng"
    nhan[~tren_ma & sut] = "giảm"
    return nhan
