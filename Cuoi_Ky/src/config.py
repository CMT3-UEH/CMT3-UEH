# Cấu hình tập trung cho đồ án Giao dịch thuật toán trên cổ phiếu FPT.

from pathlib import Path

# Đường dẫn
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = ROOT / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Tài sản
TICKER = "FPT"                 # tài sản duy nhất được giao dịch
BENCHMARK = "VNINDEX"          # biến ngoại sinh, không giao dịch
START_DATE = "2006-01-01"      # FPT niêm yết HOSE ngày 13/12/2006
END_DATE = None                # None = tới ngày hiện tại

VN_SOURCE = "vci"              # nguồn vnstock cho giá cổ phiếu

# Nhân tố vĩ mô, dùng làm biến ngoại sinh (lấy qua yfinance)
MACRO_TICKERS = {
    "USDVND": "USDVND=X",
    "OIL": "CL=F",
    "GOLD": "GC=F",
    "SP500": "^GSPC",
}

# Tham số tài chính
TRADING_DAYS = 252
RISK_FREE_ANNUAL = 0.045       # lãi suất phi rủi ro ~ TPCP VN 10 năm

# Chi phí giao dịch thực tế trên HOSE
FEE_RATE = 0.0015              # phí công ty chứng khoán, mỗi chiều
SELL_TAX = 0.0010              # thuế thu nhập, chỉ khi bán
SLIPPAGE = 0.0005              # trượt giá giả định, mỗi chiều

# Đơn vị giá: nguồn dữ liệu trả giá cổ phiếu Việt Nam theo NGHÌN đồng
# (FPT đóng cửa 72,0 nghĩa là 72.000 đồng). Mọi phép quy đổi ra tiền thật phải
# nhân với hằng số này, nếu không ràng buộc thanh khoản sẽ chặt gấp 1.000 lần.
PRICE_UNIT_VND = 1_000

# Ràng buộc thị trường Việt Nam
PRICE_LIMIT = 0.07             # biên độ dao động HOSE ±7%
SETTLEMENT_DAYS = 3            # T+2,5, làm tròn lên để mô phỏng thận trọng
MAX_ADV_PARTICIPATION = 0.10   # lệnh tối đa 10% giá trị giao dịch bình quân ngày 20 phiên
CAPITAL_VND = 1_000_000_000    # quy mô vốn giả định: 1 tỷ đồng

# Bài toán học
LOOKBACK = 60                  # số phiên trong cửa sổ đầu vào
HORIZONS = (1, 5, 10, 20)      # các tầm dự báo dự báo song song
EMBARGO = 10                   # số phiên cách ly quanh ranh giới chia dữ liệu

# Mốc chia dữ liệu theo thời gian, cố định để tránh chia lại theo tỷ lệ
TRAIN_END = "2017-12-31"
VALID_END = "2020-12-31"

# Quản lý vị thế
VOL_TARGET = 0.15              # biến động mục tiêu của chiến lược, 15%/năm
MAX_WEIGHT = 1.0               # chỉ mua hoặc đứng ngoài, không đòn bẩy
MIN_WEIGHT = 0.0

# Huấn luyện
SEEDS = (0, 1, 2, 3, 4)        # số hạt giống cho ensemble

# Mục tiêu của đề bài
SHARPE_TARGET = 1.8

# Hiển thị
APP_TITLE = "Giao dịch thuật toán — FPT"
APP_ICON = "🤖"
COLOR_UP = "#16a34a"
COLOR_DOWN = "#dc2626"
COLOR_PRIMARY = "#2563eb"
COLOR_ACCENT = "#f59e0b"
COLOR_MUTED = "#64748b"
