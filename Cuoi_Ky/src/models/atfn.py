# ATFN — Adaptive-Tempo Fusion Network, mô hình đề xuất của đồ án.

import torch
import torch.nn as nn

from src.config import HORIZONS
from src.models.deep import AN, DROPOUT, TCN, ChuanHoaDauVao

# Số chuyên gia trong cổng chế độ thị trường. Ba là đủ để tách tăng, giảm, đi ngang
# mà không tạo ra quá nhiều tham số cho lượng dữ liệu đang có.
SO_CHUYEN_GIA = 3

# Trần của vùng không giao dịch. Ngưỡng lớn hơn nửa đơn vị vị thế thì mô hình gần
# như không bao giờ giao dịch, nên chặn ở đó cho quá trình tối ưu ổn định.
TRAN_NGUONG = 0.5


class BoMaHoaDaPhanGiai(nn.Module):
    # (A) — TCN giãn nở chạy song song GRU, hai nhánh nhìn hai kiểu nhịp khác nhau.

    def __init__(self, d: int, an: int = AN):
        super().__init__()
        self.chuan = ChuanHoaDauVao(d)
        self.tcn = TCN(d, an)
        self.gru = nn.GRU(d, an, batch_first=True)
        self.hop = nn.Sequential(
            nn.Linear(2 * an, an), nn.GELU(), nn.Dropout(DROPOUT), nn.LayerNorm(an)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.chuan(x)
        a = self.tcn(z)[:, -1, :]
        b, _ = self.gru(z)
        return self.hop(torch.cat([a, b[:, -1, :]], dim=-1))


class DauDaTamNhin(nn.Module):
    # (B) — dự báo ở nhiều tầm dự báo cùng lúc, kèm cổng chọn nhịp.

    def __init__(self, an: int = AN, horizons: tuple = HORIZONS):
        super().__init__()
        self.horizons = horizons
        self.du_bao = nn.Linear(an, len(horizons))
        self.cong = nn.Linear(an, len(horizons))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        du_bao = self.du_bao(h)                       # (N, H)
        cong = torch.softmax(self.cong(h), dim=-1)    # (N, H)
        diem = (du_bao * cong).sum(dim=-1)            # (N,)
        return diem, du_bao, cong


class CongCheDo(nn.Module):
    # (C) — hỗn hợp chuyên gia điều khiển bởi trạng thái thị trường.

    def __init__(self, an: int, so_dac_trung_che_do: int, so_chuyen_gia: int = SO_CHUYEN_GIA):
        super().__init__()
        self.chuyen_gia = nn.ModuleList(
            [nn.Sequential(nn.Linear(an, an // 2), nn.GELU(), nn.Linear(an // 2, 1))
             for _ in range(so_chuyen_gia)]
        )
        self.cong = nn.Sequential(
            nn.LayerNorm(so_dac_trung_che_do),
            nn.Linear(so_dac_trung_che_do, so_chuyen_gia),
        )
        self.ty_trong_nam_giu = nn.Sequential(
            nn.LayerNorm(so_dac_trung_che_do),
            nn.Linear(so_dac_trung_che_do, 1),
        )

    def forward(self, h: torch.Tensor, che_do: torch.Tensor):
        y = torch.cat([cg(h) for cg in self.chuyen_gia], dim=-1)   # (N, K)
        w = torch.softmax(self.cong(che_do), dim=-1)               # (N, K)
        diem = (y * w).sum(dim=-1)
        # Hệ số tỷ trọng trong khoảng [0, 1]: mô hình được phép thu hẹp toàn bộ vị
        # thế khi trạng thái thị trường bất lợi, kể cả lúc tín hiệu vẫn dương.
        he_so = torch.sigmoid(self.ty_trong_nam_giu(che_do)).squeeze(-1)
        return diem, w, he_so


def ap_vung_khong_giao_dich(
    w_muc_tieu: torch.Tensor,
    nguong: torch.Tensor,
    nhiet: float = 0.02,
    w_dau: float = 0.0,
) -> torch.Tensor:
    # (D) — triển khai đường đi vị thế qua vùng không giao dịch, có đạo hàm.
    duong_di = []
    hien_tai = w_muc_tieu.new_tensor(w_dau)
    for t in range(w_muc_tieu.shape[0]):
        chenh = w_muc_tieu[t] - hien_tai
        cong = torch.sigmoid((chenh.abs() - nguong[t]) / nhiet)
        hien_tai = hien_tai + cong * chenh
        duong_di.append(hien_tai)
    return torch.stack(duong_di)


class ATFN(nn.Module):
    # Mô hình đề xuất, bật tắt được từng thành phần để chạy ablation.

    def __init__(
        self,
        d: int,
        chi_muc_che_do: list[int] | None = None,
        dung_B: bool = True,
        dung_C: bool = True,
        dung_D: bool = True,
        an: int = AN,
        horizons: tuple = HORIZONS,
    ):
        super().__init__()
        self.dung_B, self.dung_C, self.dung_D = dung_B, dung_C, dung_D
        self.chi_muc_che_do = chi_muc_che_do or []

        self.ma_hoa = BoMaHoaDaPhanGiai(d, an)

        if dung_B:
            self.dau = DauDaTamNhin(an, horizons)
        else:
            self.dau = nn.Linear(an, 1)

        if dung_C:
            if not self.chi_muc_che_do:
                raise ValueError("Bật thành phần C thì phải truyền chỉ mục đặc trưng chế độ")
            self.che_do = CongCheDo(an, len(self.chi_muc_che_do))

        # Hệ số nhân đưa điểm dự báo (đơn vị lợi suất ngày, cỡ 0,01) về thang hợp lý
        # cho hàm sigmoid. Để mạng tự học thay vì cố định một hằng số tuỳ tiện.
        self.thang = nn.Parameter(torch.tensor(50.0))

        if dung_D:
            self.dau_nguong = nn.Sequential(nn.Linear(an, an // 4), nn.GELU(),
                                            nn.Linear(an // 4, 1))

    def forward(self, x: torch.Tensor) -> dict:
        h = self.ma_hoa(x)

        cong_nhip = None
        if self.dung_B:
            diem, du_bao, cong_nhip = self.dau(h)
        else:
            diem = self.dau(h).squeeze(-1)
            du_bao = diem.unsqueeze(-1)

        cong_che_do, he_so_ty_trong = None, None
        if self.dung_C:
            dac_trung_che_do = x[:, -1, self.chi_muc_che_do]
            diem_cd, cong_che_do, he_so_ty_trong = self.che_do(h, dac_trung_che_do)
            diem = diem + diem_cd

        w_muc_tieu = torch.sigmoid(diem * self.thang)
        if self.dung_C:
            w_muc_tieu = w_muc_tieu * he_so_ty_trong

        nguong = None
        if self.dung_D:
            nguong = torch.sigmoid(self.dau_nguong(h)).squeeze(-1) * TRAN_NGUONG

        return {
            "diem": diem,
            "du_bao": du_bao,
            "w_muc_tieu": w_muc_tieu,
            "nguong": nguong,
            "cong_nhip": cong_nhip,
            "cong_che_do": cong_che_do,
            "he_so_ty_trong": he_so_ty_trong,
        }

    def vi_the(self, x: torch.Tensor, w_dau: float = 0.0) -> torch.Tensor:
        # Đường đi vị thế thật sau khi đã qua vùng không giao dịch.
        ra = self.forward(x)
        if not self.dung_D:
            return ra["w_muc_tieu"]
        return ap_vung_khong_giao_dich(ra["w_muc_tieu"], ra["nguong"], w_dau=w_dau)


# Bốn cấu hình ablation cộng dồn, cộng thêm cấu hình đầy đủ có ensemble ở phần chạy thí nghiệm.
BAC_ABLATION = {
    "ATFN-A · TCN+GRU": dict(dung_B=False, dung_C=False, dung_D=False),
    "ATFN-AB · +đa tầm dự báo & cổng nhịp": dict(dung_B=True, dung_C=False, dung_D=False),
    "ATFN-ABC · +cổng chế độ": dict(dung_B=True, dung_C=True, dung_D=False),
    "ATFN-ABCD · +vùng không giao dịch": dict(dung_B=True, dung_C=True, dung_D=True),
}
