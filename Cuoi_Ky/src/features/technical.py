# Chỉ báo phân tích kỹ thuật cơ bản (tự cài đặt, không phụ thuộc thư viện ngoài).

import numpy as np
import pandas as pd


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()


def ema(s: pd.Series, window: int) -> pd.Series:
    return s.ewm(span=window, adjust=False).mean()


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    # Relative Strength Index theo Wilder.
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    no_loss = avg_loss.eq(0)
    out = out.mask(no_loss & avg_gain.gt(0), 100.0)
    out = out.mask(no_loss & avg_gain.eq(0), 50.0)
    out.iloc[0] = np.nan            # phiên đầu chưa có biến động để tính
    return out


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(s: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = sma(s, window)
    sd = s.rolling(window).std(ddof=1)
    return pd.DataFrame({"mid": mid, "upper": mid + n_std * sd, "lower": mid - n_std * sd})


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    # Average True Range — dùng cho phần định cỡ vị thế theo rủi ro.
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


# Nhóm đặc trưng suy ra từ các chỉ báo trên, dùng làm đầu vào cho mô hình.
# Mọi cột chỉ dùng thông tin tính đến giá đóng cửa của chính phiên đó.
def technical_features(px: pd.DataFrame) -> pd.DataFrame:
    # Đặc trưng xu hướng, hồi quy về trung bình, biến động và khối lượng.
    c, v = px["close"], px["volume"]
    out = pd.DataFrame(index=px.index)

    # Xu hướng: lợi suất tích luỹ ở nhiều thang thời gian.
    for k in (1, 5, 20, 60, 120, 250):
        out[f"mom_{k}"] = c.pct_change(k)

    # Vị trí giá so với các đường trung bình động.
    for k in (20, 50, 200):
        ma = sma(c, k)
        out[f"ma_dist_{k}"] = c / ma - 1.0
        out[f"ma_slope_{k}"] = ma.pct_change(20)
    out["ma_above_200_60"] = (c > sma(c, 200)).rolling(60).mean()

    # Hồi quy về trung bình.
    ma20, sd20 = sma(c, 20), c.rolling(20).std(ddof=1)
    out["rev_z20"] = (c - ma20) / sd20.replace(0.0, np.nan)
    bb = bollinger(c, 20, 2.0)
    out["rev_bb_pctb"] = (c - bb["lower"]) / (bb["upper"] - bb["lower"]).replace(0.0, np.nan)
    out["rev_rsi14"] = rsi(c, 14) / 100.0
    out["rev_ret_5_neg"] = -c.pct_change(5)

    # Động lượng theo MACD.
    md = macd(c)
    out["macd_hist"] = md["hist"] / c
    out["macd_line"] = md["macd"] / c

    # Biến động thực hiện ở nhiều cửa sổ và tỷ lệ giữa chúng.
    r = c.pct_change()
    for k in (20, 60, 120):
        out[f"vol_{k}"] = r.rolling(k).std(ddof=1) * np.sqrt(252)
    out["vol_ratio"] = out["vol_20"] / out["vol_120"].replace(0.0, np.nan)

    # Ước lượng biến động dùng cả giá cao và giá thấp, hiệu quả hơn dùng riêng giá đóng cửa.
    if {"high", "low", "open"}.issubset(px.columns):
        h, l, o = px["high"], px["low"], px["open"]
        hl = np.log((h / l).replace(0.0, np.nan))
        out["vol_parkinson"] = np.sqrt(
            (hl ** 2).rolling(20).mean() / (4 * np.log(2))
        ) * np.sqrt(252)
        co = np.log((c / o).replace(0.0, np.nan))
        gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
        out["vol_garman_klass"] = np.sqrt(gk.rolling(20).mean().clip(lower=0)) * np.sqrt(252)
        out["atr_pct"] = atr(px, 14) / c

    # Khối lượng và giá trị giao dịch. Dùng tiền tố riêng "liq_" để không lẫn với
    # nhóm biến động "vol_" — hai nhóm này bị tách riêng ở phần ablation.
    gia_tri = v * c
    out["liq_value_z20"] = (
        (gia_tri - gia_tri.rolling(20).mean()) / gia_tri.rolling(20).std(ddof=1).replace(0.0, np.nan)
    )
    out["liq_volume_ratio"] = v / v.rolling(20).mean().replace(0.0, np.nan)
    out["liq_obv_slope"] = (np.sign(c.diff()).fillna(0) * v).cumsum().pct_change(20)
    out["liq_vol_of_vol"] = v.pct_change().rolling(20).std(ddof=1)
    out["liq_amihud"] = (
        (c.pct_change().abs() / gia_tri.replace(0.0, np.nan)).rolling(20).mean() * 1e9
    )

    return out
