# Thu thập giá FPT, chỉ số VNINDEX và nhân tố vĩ mô, cache ra data/raw.

import warnings
from datetime import date

import pandas as pd

from src.config import (
    END_DATE,
    MACRO_TICKERS,
    RAW_DIR,
    START_DATE,
    TICKER,
    VN_SOURCE,
)

warnings.filterwarnings("ignore")

NGAY_KET_THUC = END_DATE or date.today().isoformat()
INDEX_SYMBOLS = {"VNINDEX", "VN30", "HNXINDEX", "HNX30", "UPCOMINDEX"}

# Lọc giá trị sai đơn vị từ nguồn vĩ mô.
MACRO_VALID_RANGE = {
    "USDVND": (10_000, 40_000),
    "OIL": (0.01, 500),           # loại giá âm ngày 20/04/2020
    "GOLD": (100, 20_000),
    "SP500": (100, 100_000),
}


def read_cache(name: str, parse_time: bool = False) -> pd.DataFrame | None:
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["time"] if parse_time else None)
    return df if not df.empty else None


def write_cache(df: pd.DataFrame | None, name: str) -> None:
    (df if df is not None else pd.DataFrame()).to_csv(
        RAW_DIR / f"{name}.csv", index=False, encoding="utf-8-sig"
    )


# Nguồn giá
def _fetch_ohlcv_entrade(symbol: str) -> pd.DataFrame:
    # Giá từ DNSE entrade — chỉ số từ 2006, cổ phiếu chỉ từ 2012.
    import requests

    kind = "index" if symbol.upper() in INDEX_SYMBOLS else "stock"
    resp = requests.get(
        f"https://services.entrade.com.vn/chart-api/v2/ohlcs/{kind}",
        params={
            "symbol": symbol,
            "resolution": "1D",
            "from": int(pd.Timestamp(START_DATE).timestamp()),
            "to": int((pd.Timestamp(NGAY_KET_THUC) + pd.Timedelta(days=1)).timestamp()),
        },
        headers={"User-Agent": "Mozilla/5.0"}, timeout=45,
    )
    resp.raise_for_status()
    j = resp.json() or {}
    if not j.get("t"):
        raise ValueError(f"entrade không trả về dữ liệu cho {symbol}")

    return pd.DataFrame({
        "time": pd.to_datetime(j["t"], unit="s").normalize(),
        "open": pd.to_numeric(j["o"]),
        "high": pd.to_numeric(j["h"]),
        "low": pd.to_numeric(j["l"]),
        "close": pd.to_numeric(j["c"]),
        "volume": pd.to_numeric(j.get("v") or [0] * len(j["t"])),
    })


def _fetch_ohlcv_vnstock(symbol: str) -> pd.DataFrame:
    # Giá từ vnstock — cổ phiếu đủ từ ngày niêm yết.
    try:
        from vnstock.api.quote import Quote

        quote = Quote(symbol=symbol, source=VN_SOURCE)
    except Exception:                       # thư viện bản cũ
        from vnstock import Vnstock

        quote = Vnstock().stock(symbol=symbol, source=VN_SOURCE).quote

    df = quote.history(start=START_DATE, end=NGAY_KET_THUC, interval="1D")
    df = df.rename(columns=str.lower)
    df["time"] = pd.to_datetime(df["time"])
    return df[[c for c in ("time", "open", "high", "low", "close", "volume") if c in df]]


def _va_phien_thieu(df: pd.DataFrame, symbol: str, nguon_phu,
                    nguong: float = 0.20) -> pd.DataFrame:
    # Bù phiên nguồn chính bỏ sót; phiên khuyết làm lợi suất hai ngày gộp làm một.
    for fn in nguon_phu:
        try:
            extra = fn(symbol)
        except Exception:
            continue
        if extra is None or extra.empty:
            continue

        thieu = extra[
            ~extra["time"].isin(df["time"])
            & extra["time"].between(df["time"].min(), df["time"].max())
        ]
        if thieu.empty:
            continue

        df = pd.concat([df, thieu], ignore_index=True)
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

        # Nguồn đôi khi trả giá trị của chuỗi khác: VNINDEX 23/07/2007 báo 445
        # điểm trong khi các phiên quanh đó quanh 980.
        lan_can = pd.concat([df["close"].shift(1), df["close"].shift(-1)], axis=1).mean(axis=1)
        vo_ly = (df["time"].isin(set(thieu["time"])) & lan_can.notna()
                 & ((df["close"] - lan_can).abs() / lan_can > nguong))
        for _, hang in df[vo_ly].iterrows():
            print(f"[{symbol}] loại phiên vá {hang['time']:%d/%m/%Y}: "
                  f"giá {hang['close']:,.2f} lệch quá xa hai phiên liền kề")

        df = df[~vo_ly].reset_index(drop=True)
        print(f"[{symbol}] vá {len(thieu) - int(vo_ly.sum())} phiên thiếu từ {fn.__name__}")
    return df


def fetch_ohlcv(symbol: str = TICKER, refresh: bool = False) -> pd.DataFrame:
    # Giá OHLCV theo ngày, có bù phiên thiếu và dự phòng bằng cache.
    name = f"ohlcv_{symbol}"
    if not refresh:
        cached = read_cache(name, parse_time=True)
        if cached is not None:
            return cached

    if symbol.upper() in INDEX_SYMBOLS:
        thu_tu = [_fetch_ohlcv_entrade, _fetch_ohlcv_vnstock]
    else:
        thu_tu = [_fetch_ohlcv_vnstock, _fetch_ohlcv_entrade]

    df, loi_cuoi = None, None
    for fn in thu_tu:
        try:
            df = fn(symbol)
            if df is not None and not df.empty:
                break
        except Exception as exc:
            loi_cuoi = exc
            print(f"[{symbol}] nguồn {fn.__name__} lỗi: {str(exc)[:90]}")

    if df is None or df.empty:
        cached = read_cache(name, parse_time=True)
        if cached is not None:
            print(f"[{symbol}] dùng cache do mọi nguồn đều lỗi")
            return cached
        raise RuntimeError(f"Không lấy được dữ liệu giá cho {symbol}: {loi_cuoi}")

    df = _va_phien_thieu(df, symbol, thu_tu[1:])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    write_cache(df, name)
    return df


# Nhân tố vĩ mô
def fetch_macro(refresh: bool = False) -> pd.DataFrame:
    # Nhân tố vĩ mô từ Yahoo Finance, đã lọc giá trị sai đơn vị.
    if not refresh:
        cached = read_cache("macro", parse_time=True)
        if cached is not None:
            return cached

    import yfinance as yf

    chuoi = []
    for nhan, ma_yahoo in MACRO_TICKERS.items():
        try:
            raw = yf.download(ma_yahoo, start=START_DATE, end=NGAY_KET_THUC,
                              progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                continue
            s = raw["Close"]
            if isinstance(s, pd.DataFrame):      # yfinance có lúc trả MultiIndex
                s = s.iloc[:, 0]
            chuoi.append(s.rename(nhan))
        except Exception as exc:
            print(f"[macro] bỏ qua {nhan}: {str(exc)[:80]}")

    if not chuoi:
        return pd.DataFrame(columns=["time"])

    df = pd.concat(chuoi, axis=1).reset_index().rename(columns={"Date": "time"})
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.sort_values("time").reset_index(drop=True)

    # Ví dụ tỷ giá 20.885 bị ghi thành 20,885, tạo biến động giả gần 100.000%.
    for col, (lo, hi) in MACRO_VALID_RANGE.items():
        if col not in df.columns:
            continue
        bad = ~df[col].between(lo, hi) & df[col].notna()
        if bad.any():
            print(f"[macro] {col}: loại {int(bad.sum())} giá trị ngoài khoảng "
                  f"[{lo:,}–{hi:,}] — ví dụ {df.loc[bad, col].round(3).tolist()[:8]}")
        df.loc[bad, col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill()

    write_cache(df, "macro")
    return df
