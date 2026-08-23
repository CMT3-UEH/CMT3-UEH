# Khung huấn luyện dùng chung cho mọi mô hình học sâu trong dự án.

import os as _os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Dùng hết số nhân của máy. Mặc định của PyTorch trên Windows chỉ lấy một nửa.
torch.set_num_threads(_os.cpu_count() or 4)

from src.backtest.costs import BASE_COST
from src.config import SEEDS, TRADING_DAYS
from src.models.deep import WindowData, dem_tham_so
from src.models.losses import MatMatGiaoDich, mat_mat_mse


@dataclass
class CauHinhHuanLuyen:
    # Siêu tham số huấn luyện, cố định cho toàn dự án.

    che_do: str = "mse"              # "mse" hoặc "sharpe"

    # 60 vòng × 15 lô = 900 bước, thừa đủ cho mô hình vài chục nghìn tham số trên
    # 1.943 quan sát. Đo thực tế: mọi kiến trúc đều dừng sớm trước vòng 60.
    so_vong: int = 60
    hoc: float = 1e-3
    suy_giam: float = 1e-4
    lo: int = 128
    kien_nhan: int = 12
    cat_dao_ham: float = 1.0
    loss_giao_dich: MatMatGiaoDich = field(default_factory=MatMatGiaoDich)


@dataclass
class KetQuaHuanLuyen:
    # Điểm dự báo trên toàn chỉ mục, kèm nhật ký huấn luyện.

    ten: str
    scores: pd.Series
    lich_su: pd.DataFrame
    vong_tot_nhat: int
    sharpe_valid: float
    so_tham_so: int


def _sharpe_nhanh(w: np.ndarray, y: np.ndarray) -> float:
    # Sharpe xấp xỉ **đã trừ chi phí giao dịch**, dùng cho tiêu chí dừng sớm.
    thay_doi = np.abs(np.diff(w, prepend=0.0))
    r = w * y - thay_doi * (BASE_COST.round_trip / 2.0)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0


def _vi_the_tu_diem(diem: torch.Tensor, che_do: str) -> torch.Tensor:
    # Đổi đầu ra mô hình thành vị thế trong khoảng [0, 1].
    if che_do == "sharpe":
        return torch.sigmoid(diem)
    return (diem > 0).float()


def huan_luyen(
    tao_mo_hinh,
    ten: str,
    tap_train: WindowData,
    tap_valid: WindowData,
    tap_all: WindowData,
    cau_hinh: CauHinhHuanLuyen | None = None,
    seed: int = 0,
) -> KetQuaHuanLuyen:
    # Huấn luyện một mô hình, dừng sớm theo Sharpe của tập kiểm định.
    cfg = cau_hinh or CauHinhHuanLuyen()
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = tap_train.X.shape[2]
    mo_hinh = tao_mo_hinh(d)
    toi_uu = torch.optim.AdamW(mo_hinh.parameters(), lr=cfg.hoc,
                               weight_decay=cfg.suy_giam)
    lich_hoc = torch.optim.lr_scheduler.CosineAnnealingLR(toi_uu, T_max=cfg.so_vong)

    n = len(tap_train)
    y_valid = tap_valid.y.numpy()
    lich_su, tot_nhat, vong_tot, cho = [], -np.inf, 0, 0
    trang_thai_tot = None

    for vong in range(cfg.so_vong):
        mo_hinh.train()
        tong_mat_mat, so_lo = 0.0, 0

        if cfg.che_do == "sharpe":
            # Lô là đoạn phiên liên tiếp, lấy ngẫu nhiên điểm bắt đầu.
            so_lan = max(1, n // cfg.lo)
            for _ in range(so_lan):
                dau = np.random.randint(0, max(1, n - cfg.lo))
                lat = slice(dau, min(dau + cfg.lo, n))
                diem = mo_hinh(tap_train.X[lat])
                w = torch.sigmoid(diem)
                mat_mat, _ = cfg.loss_giao_dich(w, tap_train.y[lat])

                toi_uu.zero_grad()
                mat_mat.backward()
                torch.nn.utils.clip_grad_norm_(mo_hinh.parameters(), cfg.cat_dao_ham)
                toi_uu.step()
                tong_mat_mat += float(mat_mat.detach())
                so_lo += 1
        else:
            thu_tu = np.random.permutation(n)
            for i in range(0, n, cfg.lo):
                lay = thu_tu[i:i + cfg.lo]
                diem = mo_hinh(tap_train.X[lay])
                mat_mat = mat_mat_mse(diem, tap_train.y[lay])

                toi_uu.zero_grad()
                mat_mat.backward()
                torch.nn.utils.clip_grad_norm_(mo_hinh.parameters(), cfg.cat_dao_ham)
                toi_uu.step()
                tong_mat_mat += float(mat_mat.detach())
                so_lo += 1

        lich_hoc.step()

        mo_hinh.eval()
        with torch.no_grad():
            w_valid = _vi_the_tu_diem(mo_hinh(tap_valid.X), cfg.che_do).numpy()
        sharpe = _sharpe_nhanh(w_valid, y_valid)
        lich_su.append({"vòng": vong, "mất mát": tong_mat_mat / max(so_lo, 1),
                        "Sharpe kiểm định": sharpe})

        if sharpe > tot_nhat + 1e-6:
            tot_nhat, vong_tot, cho = sharpe, vong, 0
            trang_thai_tot = {k: v.clone() for k, v in mo_hinh.state_dict().items()}
        else:
            cho += 1
            if cho >= cfg.kien_nhan:
                break

    if trang_thai_tot is not None:
        mo_hinh.load_state_dict(trang_thai_tot)

    mo_hinh.eval()
    with torch.no_grad():
        diem_all = mo_hinh(tap_all.X).numpy()

    return KetQuaHuanLuyen(
        ten=ten,
        scores=pd.Series(diem_all, index=tap_all.index, name=ten),
        lich_su=pd.DataFrame(lich_su),
        vong_tot_nhat=vong_tot,
        sharpe_valid=tot_nhat,
        so_tham_so=dem_tham_so(mo_hinh),
    )


def huan_luyen_ensemble(
    tao_mo_hinh,
    ten: str,
    tap_train: WindowData,
    tap_valid: WindowData,
    tap_all: WindowData,
    cau_hinh: CauHinhHuanLuyen | None = None,
    seeds=SEEDS,
    im_lang: bool = False,
) -> tuple[pd.Series, pd.DataFrame, list[pd.Series]]:
    # Huấn luyện nhiều hạt giống rồi lấy trung bình điểm dự báo.
    diem, dong = [], []
    for s in seeds:
        kq = huan_luyen(tao_mo_hinh, ten, tap_train, tap_valid, tap_all, cau_hinh, seed=s)
        diem.append(kq.scores)
        dong.append({
            "hạt giống": s,
            "Sharpe kiểm định": kq.sharpe_valid,
            "vòng tốt nhất": kq.vong_tot_nhat,
            "số vòng đã chạy": len(kq.lich_su),
            "số tham số": kq.so_tham_so,
        })
        if not im_lang:
            print(f"      hạt giống {s}: Sharpe kiểm định {kq.sharpe_valid:+.3f} "
                  f"(dừng ở vòng {kq.vong_tot_nhat}/{len(kq.lich_su)})")

    trung_binh = pd.concat(diem, axis=1).mean(axis=1).rename(ten)
    return trung_binh, pd.DataFrame(dong), diem


# Huấn luyện mô hình đề xuất
@dataclass
class KetQuaATFN:
    # Đầu ra của một lần huấn luyện ATFN: vị thế, chẩn đoán nhịp và nhật ký.

    ten: str
    vi_the: pd.Series               # vị thế thực tế, đã đi qua vùng không giao dịch
    w_muc_tieu: pd.Series           # vị thế mong muốn, TRƯỚC khi qua vùng
    nguong: pd.Series               # ngưỡng vùng không giao dịch mô hình học được
    cong_nhip: pd.DataFrame         # trọng số cổng chọn tầm dự báo theo thời gian
    he_so_ty_trong: pd.Series     # hệ số điều tiết của cổng chế độ thị trường
    lich_su: pd.DataFrame
    vong_tot_nhat: int
    sharpe_valid: float
    so_tham_so: int


def _mat_mat_mse_atfn(ra: dict, lo: WindowData, lat) -> torch.Tensor:
    # Mất mát cho các cấu hình ablation chưa bật thành phần D.
    chinh = F.mse_loss(ra["diem"], lo.y[lat])
    if lo.Y is None or ra["du_bao"].shape[-1] != lo.Y.shape[-1]:
        return chinh
    phu = F.mse_loss(ra["du_bao"], lo.Y[lat])
    return chinh + 0.5 * phu


def huan_luyen_atfn(
    tao_mo_hinh,
    ten: str,
    tap_train: WindowData,
    tap_valid: WindowData,
    tap_all: WindowData,
    cau_hinh: CauHinhHuanLuyen | None = None,
    seed: int = 0,
) -> KetQuaATFN:
    # Huấn luyện ATFN ở một cấu hình ablation bất kỳ.
    cfg = cau_hinh or CauHinhHuanLuyen()
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = tap_train.X.shape[2]
    mo_hinh = tao_mo_hinh(d)
    toi_uu = torch.optim.AdamW(mo_hinh.parameters(), lr=cfg.hoc, weight_decay=cfg.suy_giam)
    lich_hoc = torch.optim.lr_scheduler.CosineAnnealingLR(toi_uu, T_max=cfg.so_vong)

    n = len(tap_train)
    y_valid = tap_valid.y.numpy()
    lich_su, tot_nhat, vong_tot, cho = [], -np.inf, 0, 0
    trang_thai_tot = None

    for vong in range(cfg.so_vong):
        mo_hinh.train()
        tong, so_lo = 0.0, 0

        for _ in range(max(1, n // cfg.lo)):
            dau = np.random.randint(0, max(1, n - cfg.lo))
            lat = slice(dau, min(dau + cfg.lo, n))
            x = tap_train.X[lat]

            if cfg.che_do == "sharpe":
                w = mo_hinh.vi_the(x)
                mat_mat, _ = cfg.loss_giao_dich(w, tap_train.y[lat])
            else:
                mat_mat = _mat_mat_mse_atfn(mo_hinh(x), tap_train, lat)

            toi_uu.zero_grad()
            mat_mat.backward()
            torch.nn.utils.clip_grad_norm_(mo_hinh.parameters(), cfg.cat_dao_ham)
            toi_uu.step()
            tong += float(mat_mat.detach())
            so_lo += 1

        lich_hoc.step()

        mo_hinh.eval()
        with torch.no_grad():
            if cfg.che_do == "sharpe":
                w_valid = mo_hinh.vi_the(tap_valid.X).numpy()
            else:
                w_valid = (mo_hinh(tap_valid.X)["diem"] > 0).float().numpy()
        sharpe = _sharpe_nhanh(w_valid, y_valid)
        lich_su.append({"vòng": vong, "mất mát": tong / max(so_lo, 1),
                        "Sharpe kiểm định": sharpe})

        if sharpe > tot_nhat + 1e-6:
            tot_nhat, vong_tot, cho = sharpe, vong, 0
            trang_thai_tot = {k: v.clone() for k, v in mo_hinh.state_dict().items()}
        else:
            cho += 1
            if cho >= cfg.kien_nhan:
                break

    if trang_thai_tot is not None:
        mo_hinh.load_state_dict(trang_thai_tot)

    mo_hinh.eval()
    with torch.no_grad():
        ra = mo_hinh(tap_all.X)
        if cfg.che_do == "sharpe":
            w = mo_hinh.vi_the(tap_all.X).numpy()
        else:
            w = (ra["diem"] > 0).float().numpy()

    idx = tap_all.index
    nguong = (pd.Series(ra["nguong"].numpy(), index=idx) if ra["nguong"] is not None
              else pd.Series(np.nan, index=idx))
    cong = (pd.DataFrame(ra["cong_nhip"].numpy(), index=idx,
                         columns=[f"h={h}" for h in tap_all.horizons])
            if ra["cong_nhip"] is not None else pd.DataFrame(index=idx))
    ty_trong_nam_giu = (pd.Series(ra["he_so_ty_trong"].numpy(), index=idx)
                  if ra["he_so_ty_trong"] is not None else pd.Series(np.nan, index=idx))

    return KetQuaATFN(
        ten=ten,
        vi_the=pd.Series(w, index=idx, name=ten),
        # Vị thế mong muốn trước khi qua vùng. Bắt buộc phải lưu riêng: bảng ablation
        # về nhịp giao dịch cần áp các cơ chế chọn nhịp khác nhau lên cùng một tín
        # hiệu gốc. Nếu chỉ có vị thế đã qua vùng thì mọi so sánh đều vô nghĩa vì
        # tín hiệu đã bị làm mượt sẵn.
        w_muc_tieu=pd.Series(ra["w_muc_tieu"].numpy(), index=idx, name=f"{ten} mục tiêu"),
        nguong=nguong,
        cong_nhip=cong,
        he_so_ty_trong=ty_trong_nam_giu,
        lich_su=pd.DataFrame(lich_su),
        vong_tot_nhat=vong_tot,
        sharpe_valid=tot_nhat,
        so_tham_so=dem_tham_so(mo_hinh),
    )


def huan_luyen_atfn_ensemble(
    tao_mo_hinh,
    ten: str,
    tap_train: WindowData,
    tap_valid: WindowData,
    tap_all: WindowData,
    cau_hinh: CauHinhHuanLuyen | None = None,
    seeds=SEEDS,
    im_lang: bool = False,
) -> tuple[pd.Series, pd.DataFrame, KetQuaATFN, list[pd.Series], pd.Series]:
    # Trung bình vị thế qua nhiều hạt giống.
    duong_di, muc_tieu, dong, cuoi = [], [], [], None
    for s in seeds:
        kq = huan_luyen_atfn(tao_mo_hinh, ten, tap_train, tap_valid, tap_all,
                             cau_hinh, seed=s)
        duong_di.append(kq.vi_the)
        muc_tieu.append(kq.w_muc_tieu)
        cuoi = kq
        dong.append({
            "hạt giống": s,
            "Sharpe kiểm định": kq.sharpe_valid,
            "vòng tốt nhất": kq.vong_tot_nhat,
            "số vòng đã chạy": len(kq.lich_su),
            "số tham số": kq.so_tham_so,
        })
        if not im_lang:
            print(f"      hạt giống {s}: Sharpe kiểm định {kq.sharpe_valid:+.3f} "
                  f"(dừng ở vòng {kq.vong_tot_nhat}/{len(kq.lich_su)})")

    trung_binh = pd.concat(duong_di, axis=1).mean(axis=1).rename(ten)
    tb_muc_tieu = pd.concat(muc_tieu, axis=1).mean(axis=1).rename(f"{ten} mục tiêu")
    return trung_binh, pd.DataFrame(dong), cuoi, duong_di, tb_muc_tieu
