
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for _d in (RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Tài sản phân tích
TICKER = "FPT"
BENCHMARK = "VNINDEX"
START_DATE = "2006-01-01"
END_DATE = None 

VN_SOURCE = "vci"

# Nhân tố vĩ mô cho mô hình APT (lấy qua yfinance)
MACRO_TICKERS = {
    "USDVND": "USDVND=X",      # tỷ giá USD/VND
    "OIL": "CL=F",             # dầu WTI
    "GOLD": "GC=F",            # vàng
    "SP500": "^GSPC",          # chứng khoán Mỹ
}


TRADING_DAYS = 252 # số phiên giao dịch một năm
RISK_FREE_ANNUAL = 0.045 # lãi suất phi rủi ro ~ TPCP VN 10 năm (4,5%/năm)

# Chi phí giao dịch minh hoạ (mua/bán) — dùng ở phần backtest phân bổ
FEE_RATE = 0.0015              # 0,15% mỗi chiều

APP_TITLE = "Dashboard Quản lý Đầu tư"
APP_ICON = "📈"
COLOR_UP = "#16a34a"
COLOR_DOWN = "#dc2626"
COLOR_PRIMARY = "#2563eb"
COLOR_ACCENT = "#f59e0b"
COLOR_MUTED = "#64748b"
