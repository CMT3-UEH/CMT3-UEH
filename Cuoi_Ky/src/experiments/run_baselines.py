# Giai đoạn 3 và 4 — toàn bộ baseline nhóm A và nhóm B chưa dùng học sâu.

from src.console import setup as _setup_console

_setup_console()

import json

import pandas as pd

from fetch_data import load_panel
from src.config import REPORT_DIR, VALID_END
from src.dataset import make_dataset
from src.evaluation.harness import (
    StrategyResult,
    comparison_table,
    evaluate_strategy,
    format_table,
)
from src.evaluation.metrics import sharpe_ratio
from src.evaluation.selection import (
    bang_ung_vien,
    cham_diem_mo_hinh,
    cham_diem_tin_hieu,
    chon_tot_nhat,
    walk_forward_folds,
)
from src.evaluation.stats import pbo, sharpe_std_error
from src.features.builder import build_dataset, feature_columns
from src.models import classic as C
from src.models import ml as M

DONG_NGAN = "─" * 104
DUONG_TIN_HIEU = REPORT_DIR / "tin_hieu_baseline.parquet"
DUONG_CAU_HINH = REPORT_DIR / "cau_hinh_baseline.json"

# Nhà máy sinh mô hình cho từng họ, dùng chung giữa phần chọn và phần khớp cuối.
NHA_MAY = {
    "A10 · Ridge": M.hoi_quy_tuyen_tinh,
    "A10 · Lasso": M.hoi_quy_tuyen_tinh,
    "A10 · OLS": M.hoi_quy_tuyen_tinh,
    "A11 · Logistic": M.phan_loai_logistic,
    "B1a · Rừng ngẫu nhiên": M.rung_ngau_nhien,
    "B1b · LightGBM": M.lightgbm,
}


def _mo_ta(tham_so: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in tham_so.items()) or "mặc định"


def chon_luat(ds: pd.DataFrame, d, folds) -> tuple[list[StrategyResult], dict, list]:
    # Dò lưới các chiến lược dựa trên quy tắc bằng kiểm định cuốn chiếu.
    ket_qua, chon, moi_ung_vien = [], {}, []

    for ten, luoi in C.LUOI_CO_DIEN.items():
        ham = C.HAM_CO_DIEN[ten]
        ung_vien = []
        for tham_so in luoi:
            tin_hieu = ham(ds, **tham_so)
            ung_vien.append(cham_diem_tin_hieu(ten, tham_so, tin_hieu, ds, folds))

        thang = chon_tot_nhat(ung_vien)
        moi_ung_vien.extend(ung_vien)
        chon[ten] = thang.tham_so

        tin_hieu = ham(ds, **thang.tham_so)
        kq = evaluate_strategy(ten, tin_hieu, ds, d, note=json.dumps(thang.tham_so))
        ket_qua.append(kq)

        print(f"  {ten:<38} {_mo_ta(thang.tham_so):<28} "
              f"WF gộp {thang.sharpe_gop:+.3f} "
              f"| ổn định {thang.ty_le_thang:.0%} "
              f"| valid {kq.metrics('valid')['Sharpe']:+.3f}")

    return ket_qua, chon, moi_ung_vien


def chon_mo_hinh(ds: pd.DataFrame, d, folds) -> tuple[list[StrategyResult], dict, list]:
    # Dò lưới các mô hình có tham số, khớp lại từ đầu ở từng cửa sổ cuốn chiếu.
    ket_qua, chon, moi_ung_vien = [], {}, []
    cot = feature_columns(ds)

    apt = M.apt_da_nhan_to(d, ds)
    kq = evaluate_strategy(apt.name, (apt.scores > 0).astype(float).fillna(0.0), ds, d)
    ket_qua.append(kq)
    print(f"  {apt.name:<38} {'không có siêu tham số':<28} "
          f"{'':<22} | valid {kq.metrics('valid')['Sharpe']:+.3f}")

    for ten, luoi in M.LUOI_MAY_HOC.items():
        phan_loai = "Logistic" in ten
        ung_vien = [
            cham_diem_mo_hinh(ten, tham_so, NHA_MAY[ten], ds, cot, folds,
                              horizon=d.horizon, phan_loai=phan_loai)
            for tham_so in luoi
        ]
        thang = chon_tot_nhat(ung_vien)
        moi_ung_vien.extend(ung_vien)
        chon[ten] = thang.tham_so

        # Khớp lần cuối trên toàn bộ tập huấn luyện với cấu hình đã chọn.
        mo_hinh = M.fit_predict(NHA_MAY[ten](**thang.tham_so), d, ten, phan_loai=phan_loai)
        tin_hieu = (mo_hinh.scores > 0).astype(float).fillna(0.0)
        kq = evaluate_strategy(ten, tin_hieu, ds, d, note=json.dumps(thang.tham_so))
        ket_qua.append(kq)

        print(f"  {ten:<38} {_mo_ta(thang.tham_so):<28} "
              f"WF gộp {thang.sharpe_gop:+.3f} "
              f"| ổn định {thang.ty_le_thang:.0%} "
              f"| valid {kq.metrics('valid')['Sharpe']:+.3f}")

    return ket_qua, chon, moi_ung_vien


def do_do_tin_cay(ket_qua: list[StrategyResult], part: str = "valid") -> pd.DataFrame:
    # Bảng cho thấy bảng xếp hạng đáng tin tới đâu.
    R = pd.DataFrame({k.name: k.returns(part) for k in ket_qua if part in k.runs})
    bang = pd.DataFrame({
        "Sharpe": R.apply(sharpe_ratio),
        "Sai số chuẩn": R.apply(sharpe_std_error),
    }).sort_values("Sharpe", ascending=False)
    bang["t-stat"] = bang["Sharpe"] / bang["Sai số chuẩn"]
    return bang, R


def main() -> None:
    ds = build_dataset(load_panel())
    d = make_dataset(ds, horizon=1)

    # Vùng chọn mô hình: mọi phiên trước ranh giới tập kiểm tra.
    vung_chon = ds.index[ds.index <= pd.Timestamp(VALID_END)]
    folds = walk_forward_folds(vung_chon, horizon=d.horizon)

    print(f"Dữ liệu {len(ds):,} phiên | huấn luyện {len(d.split.train):,} "
          f"| kiểm định {len(d.split.valid):,} | kiểm tra {len(d.split.test):,} (chưa đụng)")
    print(f"Vùng chọn mô hình: {len(vung_chon):,} phiên, chia thành {len(folds)} cửa sổ cuốn chiếu "
          f"({folds[0].test[0]:%m/%Y} → {folds[-1].test[-1]:%m/%Y})\n")

    print(DONG_NGAN)
    print("NHÓM A — MÔ HÌNH CỔ ĐIỂN VÀ DẠNG LUẬT")
    print(DONG_NGAN)
    moc = [
        evaluate_strategy("A1 · Mua và nắm giữ FPT", C.mua_va_nam_giu(ds), ds, d),
        evaluate_strategy("A8b · Cổ phiếu và thị trường cùng trên MA200",
                          C.theo_xu_huong_thi_truong(ds), ds, d),
    ]
    for m in moc:
        print(f"  {m.name:<38} {'không có siêu tham số':<28} "
              f"{'':<22} | valid {m.metrics('valid')['Sharpe']:+.3f}")
    luat, chon_l, uv_luat = chon_luat(ds, d, folds)

    print(f"\n{DONG_NGAN}")
    print("BASELINE CÓ THAM SỐ ƯỚC LƯỢNG — HỒI QUY, PHÂN LOẠI, CÂY")
    print(DONG_NGAN)
    mh, chon_m, uv_mh = chon_mo_hinh(ds, d, folds)

    tat_ca = moc + luat + mh

    print(f"\n{DONG_NGAN}")
    print("BẢNG SO SÁNH — TẬP KIỂM ĐỊNH")
    print(DONG_NGAN)
    bang = comparison_table(tat_ca, "valid")
    print(format_table(bang).to_string())
    bang.to_csv(REPORT_DIR / "05_baseline_valid.csv", encoding="utf-8-sig")
    comparison_table(tat_ca, "train").to_csv(
        REPORT_DIR / "05_baseline_train.csv", encoding="utf-8-sig")

    print(f"\n{DONG_NGAN}")
    print("ĐỘ TIN CẬY CỦA BẢNG XẾP HẠNG")
    print(DONG_NGAN)
    tin_cay, R = do_do_tin_cay(tat_ca, "valid")
    print(tin_cay.round(3).to_string())
    tin_cay.to_csv(REPORT_DIR / "06_do_tin_cay_valid.csv", encoding="utf-8-sig")

    k = pbo(R, so_khoi=10)
    print(f"\n  Sai số chuẩn trung bình của Sharpe : {tin_cay['Sai số chuẩn'].mean():.3f}")
    print(f"  Biên độ toàn bảng xếp hạng         : {tin_cay['Sharpe'].max() - tin_cay['Sharpe'].min():.3f}")
    print(f"  Số chiến lược có t-stat > 2        : {int((tin_cay['t-stat'] > 2).sum())}/{len(tin_cay)}")
    print(f"  PBO trên tập kiểm định             : {k.pbo:.1%} → {k.dien_giai}")
    print("\n  Đây là lý do phần chọn siêu tham số đã chuyển sang kiểm định cuốn chiếu:")
    print("  một cửa sổ 738 phiên không đủ để phân biệt các cấu hình với nhau.")

    bang_wf = bang_ung_vien(uv_luat + uv_mh)
    bang_wf.to_csv(REPORT_DIR / "07_ung_vien_tien_dan.csv", encoding="utf-8-sig")
    print(f"\n{DONG_NGAN}")
    print("MƯỜI HAI CẤU HÌNH TỐT NHẤT THEO SHARPE GỘP NGOÀI MẪU")
    print(DONG_NGAN)
    print(bang_wf.head(12).round(3).to_string())

    so_cau_hinh = C.dem_cau_hinh() + M.dem_cau_hinh() + 1
    pd.DataFrame({k.name: k.signal for k in tat_ca}).to_parquet(DUONG_TIN_HIEU)
    DUONG_CAU_HINH.write_text(
        json.dumps({"luat": chon_l, "uoc_luong": chon_m,
                    "so_cau_hinh_da_do": so_cau_hinh,
                    "so_cua_so_tien_dan": len(folds)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nTổng số cấu hình đã dò: {so_cau_hinh} — dùng để hiệu chỉnh Deflated Sharpe.")
    print(f"Đã ghi tín hiệu -> {DUONG_TIN_HIEU.name}, cấu hình -> {DUONG_CAU_HINH.name}")
    print(DONG_NGAN)


if __name__ == "__main__":
    main()
