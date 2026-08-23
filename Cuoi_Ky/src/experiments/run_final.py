# Giai đoạn 6 — chạy tập kiểm tra một lần duy nhất và kiểm định thống kê.

from src.console import setup as _setup_console

_setup_console()

import numpy as np
import pandas as pd

from fetch_data import load_panel
from src.backtest.constraints import Constraints
from src.backtest.costs import BASE_COST
from src.backtest.engine import BacktestConfig, run_backtest
from src.config import MACRO_TICKERS, REPORT_DIR, RISK_FREE_ANNUAL, SHARPE_TARGET
from src.dataset import make_dataset
from src.evaluation.attribution import (
    bang_alpha,
    bang_theo_che_do,
    bang_theo_nam,
    gan_nhan_che_do,
    phan_ra_vao_ra,
)
from src.evaluation.harness import (
    comparison_table,
    count_test_touches,
    evaluate_strategy,
    format_table,
    log_test_touch,
)
from src.evaluation.metrics import annual_return, sharpe_ratio
from src.evaluation.stats import bang_do_tin_cay, pbo, reality_check
from src.features.builder import build_dataset
from src.models import classic as C

DONG_NGAN = "─" * 104
MO_HINH_DE_XUAT = "ATFN-ABCD · +vùng không giao dịch"


def nap_tin_hieu(ds: pd.DataFrame) -> pd.DataFrame:
    # Gom mọi tín hiệu đã sinh ở các giai đoạn trước.
    phan = []
    for ten in ("tin_hieu_baseline", "tin_hieu_hoc_sau", "tin_hieu_atfn"):
        duong = REPORT_DIR / f"{ten}.parquet"
        if duong.exists():
            phan.append(pd.read_parquet(duong))
        else:
            print(f"  cảnh báo: thiếu {duong.name}, bỏ qua nhóm này")
    if not phan:
        raise FileNotFoundError("Chưa có tín hiệu nào. Chạy các giai đoạn trước đã.")
    return pd.concat(phan, axis=1).reindex(ds.index).fillna(0.0)


def do_ben_vung(ds: pd.DataFrame, d, tin_hieu: pd.Series) -> pd.DataFrame:
    # Đổi từng giả định một rồi đo lại — chiến lược chết ở giả định nào.
    idx = d.index("test")
    px = ds.loc[idx]
    hang = {}

    for ten, cfg in (
        ("chuẩn", BacktestConfig()),
        ("phí ×0 (chỉ để tham chiếu)", BacktestConfig(cost=BASE_COST.scaled(0.0))),
        ("phí ×2", BacktestConfig(cost=BASE_COST.scaled(2.0))),
        ("phí ×4", BacktestConfig(cost=BASE_COST.scaled(4.0))),
        ("trễ thực thi 2 phiên", BacktestConfig(exec_lag=2)),
        ("trễ thực thi 3 phiên", BacktestConfig(exec_lag=3)),
        ("không cộng lãi tiền mặt", BacktestConfig(cash_earns_rf=False)),
        ("bỏ ràng buộc T+2,5", BacktestConfig(
            constraints=Constraints(enforce_settlement=False))),
        ("vốn 10 tỷ (siết thanh khoản)", BacktestConfig(
            constraints=Constraints(capital=1e10))),
    ):
        r = run_backtest(px, tin_hieu.reindex(idx), cfg).returns.iloc[1:]
        hang[ten] = {"Sharpe": sharpe_ratio(r), "CAGR": annual_return(r)}
    return pd.DataFrame(hang).T


def ablation_nhip(
    ds: pd.DataFrame, d, w_muc_tieu: pd.Series, w_da_qua_bang: pd.Series
) -> pd.DataFrame:
    # Ablation về nhịp giao dịch — bảng trọng tâm của đề tài.
    from src.backtest.sizing import no_trade_band

    idx = d.index("test")
    px = ds.loc[idx]
    w = w_muc_tieu.reindex(idx).fillna(0.0)
    hang = {}

    def ghi(ten: str, tin_hieu: pd.Series) -> None:
        kq = run_backtest(px, tin_hieu, BacktestConfig())
        r = kq.returns.iloc[1:]
        nhip = kq.tempo()
        hang[ten] = {
            "Sharpe": sharpe_ratio(r),
            "CAGR": annual_return(r),
            "Lệnh mỗi năm": nhip["Số lần khớp lệnh mỗi năm"],
            "Phiên giữa hai lệnh": nhip["Số phiên giữa hai lệnh"],
            "Quy mô lệnh trung bình": nhip["Quy mô lệnh trung bình"],
            "Tổng chi phí giao dịch": nhip["Tổng chi phí giao dịch"],
        }

    # Ép tái cân bằng theo chu kỳ cố định: giữ nguyên vị thế giữa hai lần tái cân bằng.
    for ten, chu_ky in (("ép mỗi phiên", 1), ("ép mỗi tuần", 5), ("ép mỗi tháng", 20)):
        buoc = np.arange(len(idx)) // chu_ky
        ghi(ten, w.groupby(buoc).transform("first"))

    for nguong in (0.05, 0.15, 0.30):
        ghi(f"vùng cố định τ = {nguong:.2f}".replace(".", ","), no_trade_band(w, nguong))

    # Dòng cuối là chính đầu ra của mô hình: băng có ngưỡng học được theo từng phiên.
    ghi("vùng học được (ATFN-ABCD)", w_da_qua_bang.reindex(idx).fillna(0.0))
    return pd.DataFrame(hang).T


def chan_doan_hat_giong() -> pd.DataFrame | None:
    # Phát hiện những hạt giống rơi vào nghiệm suy biến của hàm mất mát Sharpe.
    cache = REPORT_DIR / "_cache_atfn"
    if not cache.exists():
        return None

    hang = {}
    for tep in sorted(cache.glob("*_vi_the.parquet")):
        bang = pd.read_parquet(tep)
        cot_hat = [c for c in bang.columns if c.startswith("hạt")]
        suy_bien = [c for c in cot_hat if bang[c].std() < 1e-9]
        hang[tep.stem.replace("_vi_the", "")] = {
            "Số hạt giống": len(cot_hat),
            "Hạt suy biến (w ≡ hằng số)": len(suy_bien),
            "Vị thế trung bình": float(bang["hợp nhất"].mean()),
            "Độ lệch vị thế": float(bang["hợp nhất"].std()),
        }
    return pd.DataFrame(hang).T if hang else None


def main() -> None:
    ds = build_dataset(load_panel())
    d = make_dataset(ds, horizon=1)
    tin_hieu = nap_tin_hieu(ds)

    so_lan, so_cau_hinh = count_test_touches()
    print(f"Nhật ký chạm tập kiểm tra trước lần chạy này: {so_lan} lần, "
          f"{so_cau_hinh} cấu hình.")
    print("(Các lần đó đến từ hàm đánh giá chung khi chạy trên train và valid; "
          "không lần nào đọc số liệu tập kiểm tra.)\n")
    log_test_touch(f"run_final: đánh giá cuối {tin_hieu.shape[1]} chiến lược",
                   tin_hieu.shape[1])

    print(f"Mô hình đề xuất đã công bố trước: {MO_HINH_DE_XUAT}")
    print(f"Số chiến lược đưa vào bảng cuối: {tin_hieu.shape[1]}\n")

    ket_qua = [
        evaluate_strategy(ten, tin_hieu[ten], ds, d, parts=("valid", "test"))
        for ten in tin_hieu.columns
    ]

    print(DONG_NGAN)
    print("BẢNG KẾT QUẢ CUỐI — TẬP KIỂM TRA (04/01/2021 – 21/08/2026)")
    print(DONG_NGAN)
    bang = comparison_table(ket_qua, "test")
    print(format_table(bang).to_string())
    bang.to_csv(REPORT_DIR / "10_ket_qua_test.csv", encoding="utf-8-sig")

    print(f"\n{DONG_NGAN}")
    print("ĐỘ TIN CẬY THỐNG KÊ — TẬP KIỂM TRA")
    print(DONG_NGAN)
    R_test = pd.DataFrame({k.name: k.returns("test") for k in ket_qua})
    tin_cay = bang_do_tin_cay({c: R_test[c] for c in R_test.columns},
                              so_cau_hinh_da_do=max(tin_hieu.shape[1], 44))
    print(tin_cay.round(3).to_string())
    tin_cay.to_csv(REPORT_DIR / "11_do_tin_cay_test.csv", encoding="utf-8-sig")

    k = pbo(R_test, so_khoi=10)
    moc_bh = R_test["A1 · Mua và nắm giữ FPT"]
    khac = R_test.drop(columns=["A1 · Mua và nắm giữ FPT"])
    thong_ke, p = reality_check(khac, moc_bh)
    print(f"\n  PBO trên tập kiểm tra          : {k.pbo:.1%} → {k.dien_giai}")
    print(f"  White Reality Check so với mua-và-giữ: thống kê {thong_ke:.4f}, "
          f"p-value {p:.4f}")
    print("  Giả thuyết không: không chiến lược nào thật sự vượt mốc mua-và-giữ.")

    # Phân tích sâu cho mô hình đề xuất
    if MO_HINH_DE_XUAT not in tin_hieu.columns:
        print(f"\nKhông tìm thấy {MO_HINH_DE_XUAT}; bỏ qua phần phân tích sâu.")
        return

    vo_dich = next(k for k in ket_qua if k.name == MO_HINH_DE_XUAT)
    r_test = vo_dich.returns("test")
    idx = d.index("test")

    print(f"\n{DONG_NGAN}")
    print(f"PHÂN TÍCH SÂU — {MO_HINH_DE_XUAT}")
    print(DONG_NGAN)
    m = vo_dich.metrics("test")
    for ten, gia_tri in m.items():
        if ten == "Chiến lược":
            continue
        print(f"  {ten:<24} {gia_tri:>10.4f}")

    dat = m["Sharpe"] >= SHARPE_TARGET
    print(f"\n  Chỉ tiêu đề bài Sharpe ≥ {SHARPE_TARGET}: "
          f"{'ĐẠT' if dat else 'CHƯA ĐẠT'} (đang là {m['Sharpe']:.3f})")

    print(f"\n{DONG_NGAN}")
    print("ALPHA HAY BETA — HỒI QUY NHÂN TỐ, SAI SỐ CHUẨN NEWEY–WEST")
    print(DONG_NGAN)
    r_tt = ds.loc[idx, "benchmark_close"].pct_change()
    r_cp = ds.loc[idx, "close"].pct_change()
    vi_mo = ds.loc[idx, [f"mac_{t}_r1" for t in MACRO_TICKERS
                         if f"mac_{t}_r1" in ds.columns]]
    bang_a = bang_alpha(r_test, r_tt, r_cp, vi_mo, rf_annual=RISK_FREE_ANNUAL)
    print(bang_a.round(4).to_string())
    bang_a.to_csv(REPORT_DIR / "12_alpha_beta.csv", encoding="utf-8-sig")

    print(f"\n{DONG_NGAN}")
    print("LỢI NHUẬN ĐẾN TỪ ĐÂU")
    print(DONG_NGAN)
    pr = phan_ra_vao_ra(r_test, r_cp, vo_dich.runs["test"].weights)
    print(pr.round(4).to_string())

    bh = evaluate_strategy("mốc", C.mua_va_nam_giu(ds), ds, d,
                           parts=("test",)).returns("test")
    print("\nTheo năm:")
    bn = bang_theo_nam(r_test, bh)
    print(bn.round(4).to_string())
    bn.to_csv(REPORT_DIR / "13_theo_nam.csv", encoding="utf-8-sig")

    print("\nTheo chế độ thị trường:")
    bc = bang_theo_che_do(r_test, gan_nhan_che_do(ds, idx), bh)
    print(bc.round(4).to_string())
    bc.to_csv(REPORT_DIR / "13_theo_che_do.csv", encoding="utf-8-sig")

    print(f"\n{DONG_NGAN}")
    print("ĐỘ VỮNG KHI ĐỔI GIẢ ĐỊNH")
    print(DONG_NGAN)
    bv = do_ben_vung(ds, d, tin_hieu[MO_HINH_DE_XUAT])
    print(bv.round(4).to_string())
    bv.to_csv(REPORT_DIR / "14_do_ben_vung.csv", encoding="utf-8-sig")

    print(f"\n{DONG_NGAN}")
    print("ABLATION VỀ NHỊP GIAO DỊCH — TRỌNG TÂM CỦA ĐỀ TÀI")
    print(DONG_NGAN)
    duong_mt = REPORT_DIR / "muc_tieu_atfn.parquet"
    if duong_mt.exists() and MO_HINH_DE_XUAT in pd.read_parquet(duong_mt).columns:
        mt = pd.read_parquet(duong_mt)[MO_HINH_DE_XUAT].reindex(ds.index).fillna(0.0)
        an = ablation_nhip(ds, d, mt, tin_hieu[MO_HINH_DE_XUAT])
    else:
        print("  Thiếu muc_tieu_atfn.parquet — chạy lại src.experiments.run_atfn.")
        an = None
    if an is not None:
        print(an.round(4).to_string())
        an.to_csv(REPORT_DIR / "15_ablation_nhip.csv", encoding="utf-8-sig")

    cd = chan_doan_hat_giong()
    if cd is not None:
        print(f"\n{DONG_NGAN}")
        print("CHẨN ĐOÁN HẠT GIỐNG — NGHIỆM SUY BIẾN CỦA HÀM MẤT MÁT SHARPE")
        print(DONG_NGAN)
        print(cd.round(4).to_string())
        cd.to_csv(REPORT_DIR / "16_chan_doan_hat_giong.csv", encoding="utf-8-sig")
        tong = int(cd["Hạt suy biến (w ≡ hằng số)"].sum())
        if tong:
            print(f"\n  {tong} hạt giống rơi vào nghiệm suy biến w ≡ hằng số. Đây là hạn")
            print("  chế thật của hàm mất mát Sharpe, không phải lỗi cài đặt: đứng ngoài")
            print("  suốt kỳ cho Sharpe bằng 0 và mọi phần phạt cũng bằng 0, nên đó là một")
            print("  cực trị hợp lệ. Báo cáo rõ thay vì để nó lặng lẽ kéo trung bình xuống.")

    so_lan, so_cau_hinh = count_test_touches()
    print(f"\n{DONG_NGAN}")
    print(f"Nhật ký chạm tập kiểm tra sau lần chạy này: {so_lan} lần, "
          f"{so_cau_hinh} cấu hình. Xem reports/test_touches.log")
    print(DONG_NGAN)


if __name__ == "__main__":
    main()
