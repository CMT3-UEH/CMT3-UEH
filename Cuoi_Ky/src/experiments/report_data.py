# Báo cáo dữ liệu và mốc tham chiếu — kết quả bàn giao của giai đoạn 1 và 2.

from src.console import setup as _setup_console

_setup_console()

import numpy as np
import pandas as pd

from fetch_data import load_panel
from src.backtest.costs import BASE_COST, NO_COST
from src.backtest.engine import BacktestConfig, buy_and_hold, run_backtest
from src.config import HORIZONS, REPORT_DIR, SHARPE_TARGET, TICKER
from src.dataset import make_dataset
from src.evaluation.metrics import (
    annual_return,
    annual_volatility,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from src.features.builder import build_dataset, feature_columns, feature_groups

DONG_NGAN = "─" * 78


def _ghi(bang: pd.DataFrame, ten: str) -> None:
    duong = REPORT_DIR / f"{ten}.csv"
    bang.to_csv(duong, encoding="utf-8-sig")
    print(f"\n  → đã ghi {duong.relative_to(REPORT_DIR.parent)}")


def mo_ta_du_lieu(ds: pd.DataFrame) -> pd.DataFrame:
    cot = feature_columns(ds)
    nhom = feature_groups(cot)
    bang = pd.DataFrame(
        {"Số đặc trưng": {k: len(v) for k, v in nhom.items()}}
    )
    bang.loc["Tổng cộng"] = bang["Số đặc trưng"].sum()
    return bang


def hieu_qua(r: pd.Series, ten: str) -> dict:
    return {
        "Giai đoạn": ten,
        "Số phiên": len(r),
        "CAGR": annual_return(r),
        "Độ biến động": annual_volatility(r),
        "Sharpe": sharpe_ratio(r),
        "Sortino": sortino_ratio(r),
        "Sụt giảm tối đa": max_drawdown(r),
    }


def moc_mua_va_giu(ds: pd.DataFrame, d) -> pd.DataFrame:
    # Mua và nắm giữ FPT trên từng giai đoạn — mốc mà mọi mô hình phải vượt qua.
    cfg = BacktestConfig(cost=BASE_COST)
    hang = []
    for ten in ("train", "valid", "test"):
        idx = d.index(ten)
        kq = buy_and_hold(ds.loc[idx], cfg)
        hang.append(hieu_qua(kq.returns.iloc[1:], ten))

    kq = buy_and_hold(ds, cfg)
    hang.append(hieu_qua(kq.returns.iloc[1:], "toàn kỳ"))
    return pd.DataFrame(hang).set_index("Giai đoạn")


def do_nhay_chi_phi(ds: pd.DataFrame, d) -> pd.DataFrame:
    # Chi phí ăn bao nhiêu lợi nhuận ở mỗi nhịp giao dịch.
    test = ds.loc[d.index("test")]
    n = len(test)
    hang = []

    for ten, chu_ky in (("mỗi phiên", 1), ("mỗi tuần", 5),
                        ("mỗi tháng", 20), ("mỗi quý", 60)):
        # Chiến lược trung tính: đảo giữa nắm giữ và đứng ngoài theo đúng chu kỳ.
        # Lấy trung bình trên mọi pha khởi đầu, nếu không thì con số đo được chỉ
        # phản ánh việc chiến lược tình cờ nằm trong hay ngoài thị trường ở những
        # đoạn tăng mạnh, chứ không phản ánh chi phí.
        gop, rong, so_lenh, sharpe = [], [], [], []
        for pha in range(2 * chu_ky):
            tin_hieu = pd.Series(
                (((np.arange(n) + pha) // chu_ky) % 2).astype(float), index=test.index
            )
            a = run_backtest(test, tin_hieu, BacktestConfig(cost=NO_COST))
            b = run_backtest(test, tin_hieu, BacktestConfig(cost=BASE_COST))
            gop.append(annual_return(a.returns.iloc[1:]))
            rong.append(annual_return(b.returns.iloc[1:]))
            sharpe.append(sharpe_ratio(b.returns.iloc[1:]))
            so_lenh.append(b.tempo()["Số lần khớp lệnh mỗi năm"])

        hang.append({
            "Nhịp đảo vị thế": ten,
            "Số lệnh mỗi năm": float(np.mean(so_lenh)),
            "CAGR trước phí": float(np.mean(gop)),
            "CAGR sau phí": float(np.mean(rong)),
            "Chi phí bào mòn mỗi năm": float(np.mean(gop) - np.mean(rong)),
            "Sharpe sau phí": float(np.mean(sharpe)),
        })
    return pd.DataFrame(hang).set_index("Nhịp đảo vị thế")


def main() -> None:
    panel = load_panel()
    ds = build_dataset(panel)

    print(DONG_NGAN)
    print(f"BỘ DỮ LIỆU — {TICKER}")
    print(DONG_NGAN)
    print(f"  Nguồn thô      : {len(panel):,} phiên, "
          f"{panel.index.min():%d/%m/%Y} – {panel.index.max():%d/%m/%Y}")
    print(f"  Sau khởi động  : {len(ds):,} phiên, "
          f"{ds.index.min():%d/%m/%Y} – {ds.index.max():%d/%m/%Y}")
    print(f"  Phiên không khớp lệnh được: {int((~ds['tradable']).sum()):,}")

    bang_dt = mo_ta_du_lieu(ds)
    print(f"\n  Bộ đặc trưng theo nhóm:")
    print(bang_dt.to_string())
    _ghi(bang_dt, "01_dac_trung")

    print(f"\n{DONG_NGAN}")
    print("CHIA DỮ LIỆU CHỐNG RÒ RỈ")
    print(DONG_NGAN)
    bang_chia = {}
    for h in HORIZONS:
        d = make_dataset(ds, horizon=h)
        hy_sinh = (len(d.split.purged) + len(d.split.embargoed)) / len(ds)
        bang_chia[f"h={h}"] = {
            "Huấn luyện": len(d.split.train),
            "Kiểm định": len(d.split.valid),
            "Kiểm tra": len(d.split.test),
            "Thanh lọc": len(d.split.purged),
            "Cách ly": len(d.split.embargoed),
            "Dữ liệu hy sinh": hy_sinh,
        }
    bang_chia = pd.DataFrame(bang_chia).T
    print(bang_chia.to_string())
    _ghi(bang_chia, "02_chia_du_lieu")

    d1 = make_dataset(ds, horizon=1)
    print(f"\n  Chi tiết lần chia dùng cho tầm dự báo 1 phiên:")
    print(d1.report().to_string())

    print(f"\n{DONG_NGAN}")
    print("MỐC THAM CHIẾU — MUA VÀ NẮM GIỮ FPT, ĐÃ TRỪ PHÍ")
    print(DONG_NGAN)
    bang_moc = moc_mua_va_giu(ds, d1)
    print(bang_moc.to_string(float_format=lambda x: f"{x:.4f}"))
    _ghi(bang_moc, "03_moc_mua_va_giu")

    sharpe_test = bang_moc.loc["test", "Sharpe"]
    print(f"\n  Chỉ tiêu đề bài là Sharpe ≥ {SHARPE_TARGET}. "
          f"Mốc mua và nắm giữ trên tập kiểm tra đang là {sharpe_test:.3f}.")
    print(f"  Mô hình cần tạo thêm {SHARPE_TARGET - sharpe_test:+.3f} đơn vị Sharpe "
          f"so với việc chỉ mua rồi để yên.")

    print(f"\n{DONG_NGAN}")
    print("CHI PHÍ GIAO DỊCH THEO NHỊP — TRÊN TẬP KIỂM TRA")
    print(DONG_NGAN)
    bang_phi = do_nhay_chi_phi(ds, d1)
    print(bang_phi.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print("  Cột đáng đọc là 'Chi phí bào mòn mỗi năm'. Mọi dòng đều nắm giữ khoảng một nửa")
    print("  thời gian nên chênh lệch giữa các dòng đúng bằng cái giá của việc giao dịch")
    print("  dày hơn. Đây là lập luận kinh tế nền cho việc để mô hình tự chọn nhịp:")
    print("  giao dịch mỗi phiên tốn gấp nhiều lần giao dịch mỗi tháng, nên tín hiệu phải")
    print("  đủ mạnh mới bù nổi chi phí — và mô hình cần học được chính điều đó.")
    _ghi(bang_phi, "04_chi_phi_theo_nhip")

    print(f"\n{DONG_NGAN}")


if __name__ == "__main__":
    main()
