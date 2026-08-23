# Trang 2 — dữ liệu và phân tích khám phá dữ liệu.

import pandas as pd
import streamlit as st

from src.config import MACRO_TICKERS, TICKER
from src.ui import bang_dep, ghi_chu, khoi_tao_trang, kho, nguyen, the_chi_so
from src.viz import ban_do_nhiet_thang, duong_tai_san
from src.evaluation.metrics import monthly_return_table, summary_table

khoi_tao_trang("Dữ liệu và phân tích khám phá dữ liệu",
               "Nguồn, độ phủ, phân phối lợi suất và kiểm định thống kê")

k = kho()
ds = k.ds

the_chi_so({
    "Số phiên": nguyen(len(ds)),
    "Từ": f"{ds.index.min():%d/%m/%Y}",
    "Đến": f"{ds.index.max():%d/%m/%Y}",
    "Phiên không khớp được lệnh": nguyen(int((~ds["tradable"]).sum())),
})

st.subheader("Nguồn dữ liệu")
st.markdown(
    """
Hai nguồn bù nhau vì không nguồn nào một mình đủ dùng:

* **vnstock** — có giá cổ phiếu từ ngày niêm yết 13/12/2006, nhưng thiếu 13 phiên chỉ số năm 2008
* **DNSE entrade** — có chỉ số liên tục từ 2006, nhưng giá cổ phiếu chỉ từ 2012

Tầng thu thập ghép hai nguồn, bù phiên khuyết và kiểm tra giá trị bất thường. Nhân tố vĩ mô
lấy từ Yahoo Finance, đã lọc các giá trị sai thang đo (tỷ giá ghi nhầm dấu phân cách) và giá dầu âm.
"""
)

st.divider()
st.subheader("Giá và khối lượng")
khoang = st.select_slider(
    "Khoảng thời gian",
    options=list(range(ds.index.year.min(), ds.index.year.max() + 1)),
    value=(ds.index.year.min(), ds.index.year.max()),
)
lat = ds.loc[str(khoang[0]):str(khoang[1])]

c1, c2 = st.columns([2, 1])
with c1:
    r = lat["close"].pct_change().dropna()
    st.plotly_chart(duong_tai_san({f"{TICKER}": (1 + r).cumprod()}, log=True),
                    width="stretch")
with c2:
    st.markdown("**Thống kê khoảng đang chọn**")
    bang_dep(summary_table(r, label=TICKER))

st.divider()
st.subheader("Phân phối lợi suất theo tháng")
st.plotly_chart(ban_do_nhiet_thang(monthly_return_table(ds["close"].pct_change().dropna())),
                width="stretch")

st.divider()
st.subheader("Kiểm định thống kê trên chuỗi lợi suất")


@st.cache_data(show_spinner=False)
def _kiem_dinh(r: pd.Series) -> pd.DataFrame:
    from scipy import stats
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.stattools import adfuller, kpss

    x = r.dropna()
    hang = {}

    adf = adfuller(x, autolag="AIC")
    hang["ADF (giả thuyết không: có nghiệm đơn vị)"] = {
        "Thống kê": adf[0], "p-value": adf[1],
        "Kết luận": "chuỗi dừng" if adf[1] < 0.05 else "chưa bác bỏ được nghiệm đơn vị",
    }

    kp = kpss(x, regression="c", nlags="auto")
    hang["KPSS (giả thuyết không: chuỗi dừng)"] = {
        "Thống kê": kp[0], "p-value": kp[1],
        "Kết luận": "bác bỏ tính dừng" if kp[1] < 0.05 else "không bác bỏ tính dừng",
    }

    jb = stats.jarque_bera(x)
    hang["Jarque–Bera (giả thuyết không: phân phối chuẩn)"] = {
        "Thống kê": jb[0], "p-value": jb[1],
        "Kết luận": "không phải phân phối chuẩn" if jb[1] < 0.05 else "chưa bác bỏ",
    }

    lb = acorr_ljungbox(x, lags=[10], return_df=True)
    hang["Ljung–Box bậc 10 (giả thuyết không: không tự tương quan)"] = {
        "Thống kê": float(lb["lb_stat"].iloc[0]), "p-value": float(lb["lb_pvalue"].iloc[0]),
        "Kết luận": "có tự tương quan" if lb["lb_pvalue"].iloc[0] < 0.05 else "chưa bác bỏ",
    }

    arch = het_arch(x, nlags=10)
    hang["ARCH-LM bậc 10 (giả thuyết không: phương sai đồng nhất)"] = {
        "Thống kê": arch[0], "p-value": arch[1],
        "Kết luận": "có cụm biến động" if arch[1] < 0.05 else "chưa bác bỏ",
    }
    return pd.DataFrame(hang).T


bang_dep(_kiem_dinh(ds["close"].pct_change()))
ghi_chu(
    "Ba kết quả đầu quyết định thiết kế của cả dự án: chuỗi lợi suất dừng nhưng "
    "không phân phối chuẩn và có cụm biến động. Vì vậy mọi khoảng tin cậy trong đồ án "
    "đều dùng bootstrap theo khối thay vì công thức giả định phân phối chuẩn, và "
    "Deflated Sharpe được dùng thay cho kiểm định t thông thường."
)

st.divider()
st.subheader("Tương quan với thị trường và nhân tố vĩ mô")
cot = ["close", "benchmark_close"] + [t for t in MACRO_TICKERS if t in ds.columns]
tuong_quan = ds[cot].pct_change().corr()
tuong_quan.index = ["FPT", "VNINDEX"] + [t for t in MACRO_TICKERS if t in ds.columns]
tuong_quan.columns = tuong_quan.index
bang_dep(tuong_quan)
