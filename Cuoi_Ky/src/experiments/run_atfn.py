# Giai đoạn 5 — mô hình đề xuất ATFN và toàn bộ thí nghiệm ablation.

from src.console import setup as _setup_console

_setup_console()

import sys
import time

import numpy as np
import pandas as pd

from fetch_data import load_panel
from src.config import LOOKBACK, REPORT_DIR, SEEDS
from src.dataset import make_dataset
from src.evaluation.harness import comparison_table, evaluate_strategy, format_table
from src.evaluation.selection import sharpe_an_toan
from src.features.builder import build_dataset, feature_groups
from src.models.atfn import ATFN, BAC_ABLATION
from src.models.deep import build_windows
from src.models.losses import MatMatGiaoDich
from src.models.trainer import CauHinhHuanLuyen, huan_luyen_atfn_ensemble

DONG_NGAN = "─" * 104
DUONG_TIN_HIEU = REPORT_DIR / "tin_hieu_atfn.parquet"
DUONG_NHIP = REPORT_DIR / "nhip_atfn.parquet"
CACHE = REPORT_DIR / "_cache_atfn"


def chi_muc_che_do(features: list[str]) -> list[int]:
    # Vị trí của nhóm đặc trưng chế độ thị trường trong danh sách đặc trưng.
    ten = set(feature_groups(features).get("chế độ", []))
    return [i for i, c in enumerate(features) if c in ten]


def bac_thi_nghiem() -> list[dict]:
    # Danh sách các cấu hình ablation, kèm chế độ huấn luyện tương ứng.
    bac = [
        {"ten": ten, "co": co, "che_do": "mse"}
        for ten, co in BAC_ABLATION.items()
        if not ten.startswith("ATFN-ABCD")
    ]
    bac.append({
        "ten": "ATFN-ABC+L · +mất mát Sharpe có phí",
        "co": dict(dung_B=True, dung_C=True, dung_D=False),
        "che_do": "sharpe",
    })
    bac.append({
        "ten": "ATFN-ABCD · +vùng không giao dịch",
        "co": dict(dung_B=True, dung_C=True, dung_D=True),
        "che_do": "sharpe",
    })
    return bac


def _ten_tep(ten: str) -> str:
    # Tên tệp an toàn suy ra từ tên cấu hình ablation.
    return ten.split("·")[0].strip().replace(" ", "_").replace("+", "p")


def nap_hoac_huan_luyen(ten, tao, tap_train, tap_valid, tap_all, cfg, lam_lai=False):
    # Nạp kết quả đã lưu nếu có, ngược lại huấn luyện rồi lưu lại ngay.
    CACHE.mkdir(parents=True, exist_ok=True)
    tep_w = CACHE / f"{_ten_tep(ten)}_vi_the.parquet"
    tep_hat = CACHE / f"{_ten_tep(ten)}_hat_giong.csv"
    tep_nhip = CACHE / f"{_ten_tep(ten)}_nhip.parquet"

    if not lam_lai and tep_w.exists() and tep_hat.exists():
        print(f"{DONG_NGAN}")
        print(f"{ten}   [đã có kết quả, bỏ qua]")
        bang = pd.read_parquet(tep_w)
        rieng = [bang[c] for c in bang.columns if c.startswith("hạt")]
        nhip = pd.read_parquet(tep_nhip) if tep_nhip.exists() else None
        mt = bang["mục tiêu"] if "mục tiêu" in bang.columns else bang["hợp nhất"]
        return bang["hợp nhất"].rename(ten), pd.read_csv(tep_hat), nhip, rieng, mt

    print(f"{DONG_NGAN}")
    print(f"{ten}   [chế độ huấn luyện: {cfg.che_do}]")
    t0 = time.time()
    w_tb, hg, cuoi, tung_hat, muc_tieu = huan_luyen_atfn_ensemble(
        tao, ten, tap_train, tap_valid, tap_all, cfg, seeds=SEEDS
    )
    hg.insert(0, "mô hình", ten)

    pd.concat([w_tb.rename("hợp nhất"), muc_tieu.rename("mục tiêu")]
              + [w.rename(f"hạt {i}") for i, w in enumerate(tung_hat)],
              axis=1).to_parquet(tep_w)
    hg.to_csv(tep_hat, index=False, encoding="utf-8-sig")

    nhip = None
    if cuoi is not None:
        nhip = cuoi.cong_nhip.copy()
        nhip["nguong"] = cuoi.nguong
        nhip["ty_trong_nam_giu"] = cuoi.he_so_ty_trong
        nhip.to_parquet(tep_nhip)

    print(f"      đã lưu sau {time.time() - t0:.0f} giây")
    return w_tb, hg, nhip, tung_hat, muc_tieu


def main() -> None:
    lam_lai = "--lam-lai" in sys.argv

    ds = build_dataset(load_panel())
    d = make_dataset(ds, horizon=1)

    tap_all = build_windows(ds, d.features, d.scaler, horizon=d.horizon, lookback=LOOKBACK)
    tap_train = tap_all.select(d.usable("train"))
    tap_valid = tap_all.select(d.usable("valid"))
    chi_muc = chi_muc_che_do(d.features)

    print(f"Cửa sổ {LOOKBACK} phiên × {tap_all.X.shape[2]} đặc trưng | "
          f"huấn luyện {len(tap_train):,} | kiểm định {len(tap_valid):,}")
    print(f"Nhóm đặc trưng chế độ cấp cho cổng (C): {len(chi_muc)} biến")
    print(f"Nhãn đa tầm dự báo: {tap_all.horizons}\n")

    # Phạt vòng quay vị thế đặt theo cùng bậc độ lớn với chi phí một vòng mua bán: mô hình
    # chỉ nên đổi vị thế khi lợi ích kỳ vọng vượt được cái giá thật của việc đó.
    loss = MatMatGiaoDich(phat_turnover=0.5, phat_muot=0.1)

    ket_qua, bang_hat_giong, tin_hieu, muc_tieu_bac = [], [], {}, {}
    chan_doan = None

    for b in bac_thi_nghiem():
        cfg = CauHinhHuanLuyen(che_do=b["che_do"], loss_giao_dich=loss)

        def tao(dim, co=b["co"]):
            return ATFN(dim, chi_muc_che_do=chi_muc, **co)

        w_tb, hg, nhip_bac, tung_hat, muc_tieu = nap_hoac_huan_luyen(
            b["ten"], tao, tap_train, tap_valid, tap_all, cfg, lam_lai
        )
        muc_tieu_bac[b["ten"]] = muc_tieu.reindex(ds.index).fillna(0.0)
        bang_hat_giong.append(hg)

        # Kết quả của từng hạt giống riêng lẻ, để tách đóng góp của việc hợp nhất.
        # Dùng Sharpe an toàn: hạt giống rơi vào nghiệm suy biến w ≡ 0 cho chuỗi lợi
        # suất vượt trội hằng số, và Sharpe của nó là 0 chứ không phải "không xác
        # định". Nếu để rỗng thì một hạt giống hỏng sẽ làm cả trung bình thành NaN.
        sharpe_rieng = [
            sharpe_an_toan(
                evaluate_strategy("tạm", w.reindex(ds.index).fillna(0.0), ds, d)
                .returns("valid")
            )
            for w in tung_hat
        ]
        so_suy_bien = sum(1 for w in tung_hat if w.std() < 1e-9)

        s_tb = w_tb.reindex(ds.index).fillna(0.0)
        tin_hieu[b["ten"]] = s_tb
        kq = evaluate_strategy(b["ten"], s_tb, ds, d)
        kq.note = (f"hạt giống đơn lẻ {np.mean(sharpe_rieng):+.3f}"
                   f"±{np.std(sharpe_rieng, ddof=1):.3f}")
        ket_qua.append(kq)

        m = kq.metrics("valid")
        print(f"      → từng hạt giống : Sharpe {np.mean(sharpe_rieng):+.3f} "
              f"± {np.std(sharpe_rieng, ddof=1):.3f}"
              + (f"  [{so_suy_bien} hạt suy biến w≡const]" if so_suy_bien else ""))
        print(f"      → hợp nhất 5 hạt : Sharpe {m['Sharpe']:+.3f} | "
              f"{m['Lệnh mỗi năm']:.0f} lệnh/năm | "
              f"cứ {m['Phiên giữa hai lệnh']:.1f} phiên một lệnh | "
              f"trong thị trường {m['Tỷ lệ thời gian có vị thế']:.0%}")

        if nhip_bac is not None:
            cot_h = [c for c in nhip_bac.columns if c.startswith("h=")]
            if cot_h:
                tb = nhip_bac[cot_h].mean()
                print("      → cổng nhịp TB   : "
                      + ", ".join(f"{c} {v:.0%}" for c, v in tb.items()))
            if "nguong" in nhip_bac and nhip_bac["nguong"].notna().any():
                ng = nhip_bac["nguong"]
                print(f"      → ngưỡng vùng    : trung bình {ng.mean():.3f}, "
                      f"khoảng [{ng.min():.3f}, {ng.max():.3f}]")
                chan_doan = nhip_bac

        # Ghi hợp nhất sau mỗi bậc để phần sau dùng được ngay cả khi dừng giữa chừng.
        pd.DataFrame(tin_hieu).to_parquet(DUONG_TIN_HIEU)
        pd.DataFrame(muc_tieu_bac).to_parquet(REPORT_DIR / "muc_tieu_atfn.parquet")

    print(f"\n{DONG_NGAN}")
    print("BẢNG ABLATION — TẬP KIỂM ĐỊNH")
    print(DONG_NGAN)
    bang = comparison_table(ket_qua, "valid")
    print(format_table(bang).to_string())
    bang.to_csv(REPORT_DIR / "09_ablation_valid.csv", encoding="utf-8-sig")

    goc = bang["Sharpe"].get("ATFN-A · TCN+GRU", np.nan)
    if np.isfinite(goc):
        print(f"\n  Mức tăng Sharpe cộng dồn so với bậc A ({goc:+.3f}):")
        for b in bac_thi_nghiem():
            ten = b["ten"]
            if ten in bang.index:
                print(f"    {ten:<46} {bang.loc[ten, 'Sharpe']:+.3f}  "
                      f"(Δ {bang.loc[ten, 'Sharpe'] - goc:+.3f})")

    pd.concat(bang_hat_giong, ignore_index=True).to_csv(
        REPORT_DIR / "09_hat_giong_atfn.csv", index=False, encoding="utf-8-sig")

    if chan_doan is not None:
        chan_doan.to_parquet(DUONG_NHIP)
        print(f"\nĐã ghi chẩn đoán nhịp -> {DUONG_NHIP.name}")

    print(f"Đã ghi tín hiệu -> {DUONG_TIN_HIEU.name}")
    print(DONG_NGAN)


if __name__ == "__main__":
    main()
