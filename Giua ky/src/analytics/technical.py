"""Chỉ báo phân tích kỹ thuật cơ bản (tự cài đặt, không phụ thuộc thư viện ngoài)."""

import numpy as np
import pandas as pd


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()


def ema(s: pd.Series, window: int) -> pd.Series:
    return s.ewm(span=window, adjust=False).mean()


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index theo Wilder."""
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
    """Average True Range — dùng cho phần định cỡ vị thế theo rủi ro."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm toàn bộ chỉ báo vào khung dữ liệu giá (index = thời gian)."""
    out = df.copy()
    c = out["close"]
    out["MA20"] = sma(c, 20)
    out["MA50"] = sma(c, 50)
    out["MA200"] = sma(c, 200)
    out["RSI14"] = rsi(c, 14)
    out = out.join(macd(c))
    out = out.join(bollinger(c).add_prefix("bb_"))
    if {"high", "low"}.issubset(out.columns):
        out["ATR14"] = atr(out, 14)
    if "volume" in out.columns:
        out["VOL_MA20"] = sma(out["volume"], 20)
    return out


def signal_summary(df: pd.DataFrame) -> dict[str, str]:
    """Diễn giải trạng thái kỹ thuật hiện tại — dùng cho chatbot và dashboard."""
    last = df.dropna(subset=["MA20", "MA50", "RSI14"]).iloc[-1]
    out = {}
    out["Xu hướng theo MA"] = (
        "Tăng (giá > MA20 > MA50)"
        if last["close"] > last["MA20"] > last["MA50"]
        else "Giảm (giá < MA20 < MA50)"
        if last["close"] < last["MA20"] < last["MA50"]
        else "Đi ngang / chưa rõ xu hướng"
    )
    r = last["RSI14"]
    out["RSI(14)"] = (
        f"{r:.1f} — vùng quá mua" if r >= 70
        else f"{r:.1f} — vùng quá bán" if r <= 30
        else f"{r:.1f} — vùng trung tính"
    )
    out["MACD"] = (
        "Trên đường tín hiệu (động lượng dương)"
        if last["macd"] > last["signal"]
        else "Dưới đường tín hiệu (động lượng âm)"
    )
    if "MA200" in last and not np.isnan(last["MA200"]):
        out["So với MA200"] = (
            f"Giá cao hơn MA200 {(last['close'] / last['MA200'] - 1) * 100:.1f}%"
            if last["close"] > last["MA200"]
            else f"Giá thấp hơn MA200 {(1 - last['close'] / last['MA200']) * 100:.1f}%"
        )
    return out
