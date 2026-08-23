# Hàm mất mát — nơi thành phần (D) của mô hình đề xuất thật sự nằm.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.config import FEE_RATE, SELL_TAX, SLIPPAGE, TRADING_DAYS


def mat_mat_mse(du_bao: torch.Tensor, that: torch.Tensor) -> torch.Tensor:
    # Mức 1 — sai số bình phương trung bình, tiêu chí của mọi baseline nhóm B.
    return F.mse_loss(du_bao, that)


def sharpe_kha_vi(loi_suat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Tỷ số Sharpe thường niên hoá, khả vi theo chuỗi lợi suất danh mục.
    mu = loi_suat.mean()
    sd = loi_suat.std(unbiased=True)
    return mu / (sd + eps) * (TRADING_DAYS ** 0.5)


@dataclass
class MatMatGiaoDich:
    # Mức 3 — Sharpe sau chi phí, kèm phạt vòng quay vị thế và phạt vị thế giật cục.

    phi_mua: float = FEE_RATE + SLIPPAGE
    phi_ban: float = FEE_RATE + SELL_TAX + SLIPPAGE
    phat_turnover: float = 0.0
    phat_muot: float = 0.0

    # Hàm mất mát này có nghiệm suy biến tại w ≡ 0: đứng ngoài suốt kỳ cho Sharpe 0
    # và mọi phần phạt cũng 0. Thực tế 2/25 hạt giống đã rơi vào đó.

    def __call__(
        self,
        vi_the: torch.Tensor,
        loi_suat_tai_san: torch.Tensor,
        vi_the_dau: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        # Trả về (giá trị mất mát, các thành phần để ghi nhật ký).
        truoc = torch.cat([
            vi_the.new_zeros(1) if vi_the_dau is None else vi_the_dau.reshape(1),
            vi_the[:-1],
        ])
        thay_doi = vi_the - truoc

        mua = torch.clamp(thay_doi, min=0.0)
        ban = torch.clamp(-thay_doi, min=0.0)
        chi_phi = mua * self.phi_mua + ban * self.phi_ban

        loi_suat = vi_the * loi_suat_tai_san - chi_phi
        sharpe = sharpe_kha_vi(loi_suat)

        turnover = thay_doi.abs().mean()
        muot = (vi_the[1:] - vi_the[:-1]).pow(2).mean() if len(vi_the) > 1 \
            else vi_the.new_zeros(())

        mat_mat = -sharpe + self.phat_turnover * turnover + self.phat_muot * muot
        return mat_mat, {
            "sharpe": float(sharpe.detach()),
            "turnover": float(turnover.detach()),
            "chi_phi": float(chi_phi.sum().detach()),
            "vi_the_tb": float(vi_the.mean().detach()),
        }
