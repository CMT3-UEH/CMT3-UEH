# Giai đoạn 4 — baseline học sâu thông dụng: MLP, CNN, RNN, LSTM, GRU, Transformer.

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
from src.evaluation.metrics import sharpe_ratio
from src.features.builder import build_dataset
from src.models.deep import BASELINE_HOC_SAU, build_windows
from src.models.trainer import CauHinhHuanLuyen, huan_luyen_ensemble

DONG_NGAN = "─" * 104
CACHE = REPORT_DIR / "_cache_deep"
DUONG_TIN_HIEU = REPORT_DIR / "tin_hieu_hoc_sau.parquet"


def _ten_tep(ten: str) -> str:
    # Tên tệp an toàn suy ra từ tên mô hình.
    return ten.split("·")[0].strip().replace(" ", "_")


def nap_hoac_huan_luyen(ten, tao, tap_train, tap_valid, tap_all, cfg, lam_lai=False):
    # Nạp kết quả đã lưu nếu có, ngược lại huấn luyện rồi lưu lại ngay.
    CACHE.mkdir(parents=True, exist_ok=True)
    tep_tin_hieu = CACHE / f"{_ten_tep(ten)}_tin_hieu.parquet"
    tep_hat = CACHE / f"{_ten_tep(ten)}_hat_giong.csv"

    if not lam_lai and tep_tin_hieu.exists() and tep_hat.exists():
        print(f"{DONG_NGAN}\n{ten}   [đã có kết quả, bỏ qua]")
        bang = pd.read_parquet(tep_tin_hieu)
        rieng = [bang[c] for c in bang.columns if c != "hợp nhất"]
        return bang["hợp nhất"].rename(ten), pd.read_csv(tep_hat), rieng

    print(f"{DONG_NGAN}\n{ten}")
    t0 = time.time()
    diem, hat_giong, rieng = huan_luyen_ensemble(tao, ten, tap_train, tap_valid,
                                                 tap_all, cfg, seeds=SEEDS)
    hat_giong.insert(0, "mô hình", ten)

    luu = pd.concat(
        [diem.rename("hợp nhất")] + [r.rename(f"hạt {i}") for i, r in enumerate(rieng)],
        axis=1,
    )
    luu.to_parquet(tep_tin_hieu)
    hat_giong.to_csv(tep_hat, index=False, encoding="utf-8-sig")
    print(f"      đã lưu sau {time.time() - t0:.0f} giây")
    return diem, hat_giong, rieng


def main() -> None:
    lam_lai = "--lam-lai" in sys.argv

    ds = build_dataset(load_panel())
    d = make_dataset(ds, horizon=1)

    print("Dựng tensor cửa sổ trượt ...")
    tap_all = build_windows(ds, d.features, d.scaler, horizon=d.horizon, lookback=LOOKBACK)
    tap_train = tap_all.select(d.usable("train"))
    tap_valid = tap_all.select(d.usable("valid"))
    print(f"  cửa sổ {LOOKBACK} phiên × {tap_all.X.shape[2]} đặc trưng | "
          f"huấn luyện {len(tap_train):,} | kiểm định {len(tap_valid):,} | "
          f"toàn chuỗi {len(tap_all):,}\n")

    cfg = CauHinhHuanLuyen(che_do="mse")
    ket_qua, bang_hat_giong, tin_hieu = [], [], {}

    for ten, tao in BASELINE_HOC_SAU.items():
        diem, hat_giong, rieng = nap_hoac_huan_luyen(
            ten, tao, tap_train, tap_valid, tap_all, cfg, lam_lai
        )
        bang_hat_giong.append(hat_giong)

        # Sharpe thật của từng hạt giống, đo bằng bộ máy backtest đầy đủ. Cần tách
        # bạch phần này với phần hợp nhất: nếu hợp nhất tốt hơn hẳn thì cải thiện
        # đến từ việc giảm phương sai chứ không từ kiến trúc.
        sharpe_rieng = [
            sharpe_ratio(
                evaluate_strategy(
                    "tạm", (r > 0).astype(float).reindex(ds.index).fillna(0.0), ds, d
                ).returns("valid")
            )
            for r in rieng
        ]

        # Ngưỡng 0: mô hình dự báo lợi suất, nắm giữ khi dự báo dương.
        s = (diem > 0).astype(float).reindex(ds.index).fillna(0.0)
        tin_hieu[ten] = s
        kq = evaluate_strategy(ten, s, ds, d)
        ket_qua.append(kq)

        m = kq.metrics("valid")
        print(f"      → từng hạt giống (backtest thật): Sharpe "
              f"{np.mean(sharpe_rieng):+.3f} ± {np.std(sharpe_rieng, ddof=1):.3f}")
        print(f"      → hợp nhất 5 hạt : Sharpe {m['Sharpe']:+.3f} | "
              f"CAGR {m['CAGR']:.2%} | {m['Lệnh mỗi năm']:.0f} lệnh/năm | "
              f"trong thị trường {m['Tỷ lệ thời gian có vị thế']:.0%}")

        # Ghi hợp nhất sau mỗi mô hình để phần sau dùng được ngay cả khi dừng giữa chừng.
        pd.DataFrame(tin_hieu).to_parquet(DUONG_TIN_HIEU)

    print(f"\n{DONG_NGAN}")
    print("BẢNG SO SÁNH NHÓM B — TẬP KIỂM ĐỊNH")
    print(DONG_NGAN)
    bang = comparison_table(ket_qua, "valid")
    print(format_table(bang).to_string())
    bang.to_csv(REPORT_DIR / "08_hoc_sau_valid.csv", encoding="utf-8-sig")

    hg = pd.concat(bang_hat_giong, ignore_index=True)
    hg.to_csv(REPORT_DIR / "08_hat_giong_hoc_sau.csv", index=False, encoding="utf-8-sig")

    print(f"\n{DONG_NGAN}")
    print("DAO ĐỘNG GIỮA CÁC HẠT GIỐNG")
    print(DONG_NGAN)
    print(hg.groupby("mô hình")["Sharpe kiểm định"]
          .agg(["mean", "std", "min", "max"]).round(3).to_string())
    print("\n  Cột std là con số đáng chú ý nhất: nếu nó lớn hơn khoảng cách giữa các")
    print("  kiến trúc thì bảng xếp hạng kiến trúc không có ý nghĩa, và cách duy nhất")
    print("  để báo cáo trung thực là ghi trung bình kèm độ lệch chứ không ghi một số.")

    print(f"\nĐã ghi tín hiệu -> {DUONG_TIN_HIEU.name}")
    print(DONG_NGAN)


if __name__ == "__main__":
    main()
