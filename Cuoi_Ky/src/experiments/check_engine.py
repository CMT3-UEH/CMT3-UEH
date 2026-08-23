# Tự kiểm bộ máy backtest trước khi cho bất kỳ mô hình nào chạy qua nó.

from src.console import setup as _setup_console

_setup_console()

import numpy as np
import pandas as pd

from fetch_data import load_panel
from src.backtest.constraints import Constraints
from src.backtest.costs import BASE_COST, NO_COST
from src.backtest.engine import BacktestConfig, buy_and_hold, run_backtest
from src.config import TRADING_DAYS
from src.evaluation.metrics import annual_return, sharpe_ratio
from src.features.builder import build_dataset

DONG_NGAN = "─" * 72


def _cau_hinh_tran(**kw) -> BacktestConfig:
    # Cấu hình trần trụi: không phí, không lãi tiền mặt, không ràng buộc.
    mac_dinh = dict(
        cost=NO_COST,
        constraints=Constraints(
            max_adv=None, enforce_settlement=False, min_trade=0.0
        ),
        cash_earns_rf=False,
        respect_tradable=False,
    )
    return BacktestConfig(**{**mac_dinh, **kw})


def kiem_nam_giu_toan_phan(ds: pd.DataFrame) -> None:
    # Nắm giữ toàn thời gian phải trùng khớp lợi suất của chính cổ phiếu.
    kq = buy_and_hold(ds, _cau_hinh_tran())

    that = ds["close"].pct_change().fillna(0.0)
    that.iloc[0] = 0.0
    lech = (kq.returns - that).abs().max()

    print(f"  Lệch lớn nhất so với lợi suất FPT: {lech:.2e}")
    assert lech < 1e-12, (
        f"Nắm giữ toàn phần không khớp lợi suất cổ phiếu (lệch {lech:.2e}). "
        "Sai ở khâu ghép hai đoạn qua đêm và trong phiên."
    )

    tong_phi = kq.costs.sum()
    assert tong_phi == 0.0, f"Không đặt phí mà vẫn tính ra chi phí {tong_phi}"
    so_lenh = int(kq.trade_days().sum())
    assert so_lenh == 1, f"Mua và giữ phải khớp đúng 1 lệnh, đang là {so_lenh}"
    print(f"  Số lệnh: {so_lenh} — đúng như kỳ vọng của chiến lược mua và giữ")


def kiem_dung_ngoai(ds: pd.DataFrame) -> None:
    # Đứng ngoài hoàn toàn phải cho lợi suất bằng đúng lãi suất phi rủi ro.
    tin_hieu = pd.Series(0.0, index=ds.index)

    kq0 = run_backtest(ds, tin_hieu, _cau_hinh_tran())
    assert kq0.returns.abs().max() < 1e-15, "Đứng ngoài mà vẫn có lãi lỗ"

    kq1 = run_backtest(ds, tin_hieu, BacktestConfig(cost=NO_COST, cash_earns_rf=True))
    lai_nam = annual_return(kq1.returns.iloc[1:])
    print(f"  Đứng ngoài, có lãi tiền mặt: {lai_nam:.4%}/năm (đặt 4,5000%)")
    assert abs(lai_nam - 0.045) < 1e-4, "Lãi tiền mặt không khớp mức đã đặt"


def kiem_chien_luoc_ngau_nhien(ds: pd.DataFrame, so_lan: int = 200) -> None:
    # Tín hiệu ngẫu nhiên không được sinh ra kỹ năng định thời điểm.
    rng = np.random.default_rng(20260822)
    cfg_co_phi = BacktestConfig(cost=BASE_COST, cash_earns_rf=False)
    cfg_khong_phi = BacktestConfig(cost=NO_COST, cash_earns_rf=False)

    bh = sharpe_ratio(buy_and_hold(ds, cfg_khong_phi).returns.iloc[1:])
    print(f"  Mốc: mua và nắm giữ FPT có Sharpe {bh:.3f}")

    chenh_gop, sharpe_rong = [], []
    for _ in range(so_lan):
        tin_hieu = pd.Series(rng.integers(0, 2, len(ds)).astype(float), index=ds.index)

        ngau_nhien = run_backtest(ds, tin_hieu, cfg_khong_phi)
        ty_trong_nam_giu = float(ngau_nhien.weights.mean())

        # Mốc tĩnh cùng tỷ trọng nắm giữ: nắm đúng tỷ trọng đó suốt kỳ, không định thời điểm.
        tinh = run_backtest(ds, pd.Series(ty_trong_nam_giu, index=ds.index), cfg_khong_phi)

        chenh_gop.append(ngau_nhien.returns.iloc[1:].mean() - tinh.returns.iloc[1:].mean())
        sharpe_rong.append(
            sharpe_ratio(run_backtest(ds, tin_hieu, cfg_co_phi).returns.iloc[1:])
        )

    chenh = pd.Series(chenh_gop) * TRADING_DAYS          # quy về đơn vị năm
    s = pd.Series(sharpe_rong)

    print(f"  Sharpe sau phí của {so_lan} lần thử ngẫu nhiên: "
          f"trung vị {s.median():.3f}, khoảng [{s.min():.2f}, {s.max():.2f}]")
    print(f"  Lợi suất trung bình lệch mốc tĩnh cùng tỷ trọng: "
          f"{chenh.mean():+.4%}/năm (sai số chuẩn {chenh.sem():.4%})")

    # Bất biến cần kiểm là kỳ vọng lợi suất số học, không phải lợi suất kép. Vị thế
    # ngẫu nhiên có cùng kỳ vọng số học với mốc tĩnh nhưng phương sai lớn hơn, nên
    # lợi suất kép của nó thấp hơn một cách có hệ thống. Đó là hao hụt do biến động,
    # một hiện tượng thật, không phải lỗi cài đặt.
    assert abs(chenh.mean()) < 3 * max(chenh.sem(), 1e-9), (
        f"Vị thế ngẫu nhiên lệch mốc tĩnh {chenh.mean():+.4%}/năm, vượt ba lần sai số "
        "chuẩn — bộ máy đang tự sinh hoặc tự huỷ lợi nhuận một cách có hệ thống."
    )
    assert s.median() < bh, (
        f"Chiến lược ngẫu nhiên (Sharpe {s.median():.3f}) không được vượt mốc mua và "
        f"nắm giữ ({bh:.3f}): giảm tỷ trọng một cách vô hướng thì phải giảm Sharpe."
    )


def kiem_phi_lam_giam_loi_nhuan(ds: pd.DataFrame) -> None:
    # Tăng phí phải làm lợi suất giảm, đơn điệu và không có ngoại lệ.
    rng = np.random.default_rng(7)
    tin_hieu = pd.Series(rng.integers(0, 2, len(ds)).astype(float), index=ds.index)

    truoc = None
    for he_so in (0.0, 1.0, 2.0, 4.0):
        cfg = BacktestConfig(cost=BASE_COST.scaled(he_so), cash_earns_rf=False)
        kq = run_backtest(ds, tin_hieu, cfg)
        tong = kq.returns.sum()
        print(f"  Phí ×{he_so:<4}  tổng lợi suất {tong:+.4f}  "
              f"chi phí đã trả {kq.costs.sum():.4f}")
        if truoc is not None:
            assert tong < truoc + 1e-12, "Tăng phí mà lợi nhuận không giảm"
        truoc = tong


def kiem_tre_tin_hieu(ds: pd.DataFrame) -> None:
    # Dịch tín hiệu chậm thêm một phiên phải làm kết quả xấu đi.
    # Tín hiệu cố ý rò rỉ: dùng chính lợi suất của phiên kế tiếp.
    ro_ri = (ds["close"].pct_change().shift(-1) > 0).astype(float)
    cfg = BacktestConfig(cost=NO_COST, cash_earns_rf=False)

    ket_qua = {}
    for tre in (0, 1, 2):
        kq = run_backtest(ds, ro_ri, BacktestConfig(
            cost=cfg.cost, cash_earns_rf=False, exec_lag=cfg.exec_lag + tre))
        ket_qua[tre] = sharpe_ratio(kq.returns.iloc[1:])
        print(f"  Trễ thêm {tre} phiên: Sharpe {ket_qua[tre]:>7.3f}")

    assert ket_qua[0] > 3.0, (
        "Tín hiệu biết trước tương lai lẽ ra phải cho Sharpe rất cao. "
        "Nếu không thì quy ước thời gian của bộ máy đang lệch."
    )
    assert ket_qua[1] < ket_qua[0] - 1.0, (
        "Làm chậm tín hiệu một phiên mà Sharpe không giảm mạnh — quy ước thời gian sai."
    )


def kiem_rang_buoc_thanh_toan(ds: pd.DataFrame) -> None:
    # Chu kỳ T+2,5 phải chặn bớt số lệnh của một chiến lược đảo vị thế liên tục.
    dao_lien_tuc = pd.Series(
        (np.arange(len(ds)) % 2).astype(float), index=ds.index
    )

    khong_rang_buoc = run_backtest(ds, dao_lien_tuc, BacktestConfig(
        constraints=Constraints(enforce_settlement=False, max_adv=None)))
    co_rang_buoc = run_backtest(ds, dao_lien_tuc, BacktestConfig(
        constraints=Constraints(enforce_settlement=True, max_adv=None)))

    a = int(khong_rang_buoc.trade_days().sum())
    b = int(co_rang_buoc.trade_days().sum())
    print(f"  Số lệnh khi bỏ qua T+: {a:,} — khi tôn trọng T+: {b:,} "
          f"(giảm {1 - b / max(a, 1):.1%})")
    assert b < a * 0.75, "Ràng buộc thanh toán không chặn được lệnh nào đáng kể"


def main() -> None:
    print("Dựng bảng dữ liệu ...")
    ds = build_dataset(load_panel())
    print(f"  {len(ds):,} phiên, {ds.index.min():%d/%m/%Y} – {ds.index.max():%d/%m/%Y}")

    phep_thu = [
        ("Nắm giữ toàn phần khớp lợi suất cổ phiếu", kiem_nam_giu_toan_phan),
        ("Đứng ngoài chỉ hưởng lãi phi rủi ro", kiem_dung_ngoai),
        ("Chiến lược ngẫu nhiên không sinh lợi nhuận", kiem_chien_luoc_ngau_nhien),
        ("Tăng phí làm giảm lợi nhuận", kiem_phi_lam_giam_loi_nhuan),
        ("Làm chậm tín hiệu làm hỏng kết quả", kiem_tre_tin_hieu),
        ("Chu kỳ thanh toán T+2,5 chặn bớt lệnh", kiem_rang_buoc_thanh_toan),
    ]

    for i, (ten, ham) in enumerate(phep_thu, 1):
        print(f"\n{DONG_NGAN}\n[{i}/{len(phep_thu)}] {ten}")
        ham(ds)

    print(f"\n{DONG_NGAN}\nToàn bộ {len(phep_thu)} phép thử của bộ máy backtest đều đạt.")


if __name__ == "__main__":
    main()
