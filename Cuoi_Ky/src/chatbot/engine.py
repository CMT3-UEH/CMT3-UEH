# Bộ khớp câu hỏi tiếng Việt, có ngưỡng từ chối.

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.chatbot.questions import CauHoi, xay_dung_ngan_hang

NGUONG_TRA_LOI = 0.42


def bo_dau(s: str) -> str:
    # Bỏ dấu tiếng Việt để khớp được cả khi người dùng gõ không dấu.
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d")


def chuan_hoa(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", bo_dau(s))).strip()


@dataclass
class KetQuaKhop:
    # Câu hỏi khớp được, kèm điểm tin cậy.

    cau_hoi: CauHoi | None
    diem: float

    @property
    def du_tin_cay(self) -> bool:
        return self.cau_hoi is not None and self.diem >= NGUONG_TRA_LOI


def _diem(hoi: str, ch: CauHoi) -> float:
    # Điểm khớp giữa câu người dùng gõ và một mục trong ngân hàng.
    h = chuan_hoa(hoi)

    # Lớp 1: từ khoá đã khai báo. Khớp được cụm nào thì cộng điểm cụm đó.
    trung = sum(1 for tk in ch.tu_khoa if chuan_hoa(tk) in h)
    diem_tu_khoa = min(1.0, trung / max(1, min(2, len(ch.tu_khoa))))

    # Lớp 2: độ tương đồng chuỗi với chính câu hỏi mẫu.
    diem_chuoi = SequenceMatcher(None, h, chuan_hoa(ch.cau_hoi)).ratio()

    # Lớp 3: tỷ lệ từ chung, giúp bắt câu hỏi diễn đạt khác nhưng cùng nội dung.
    tu_hoi = set(h.split())
    tu_mau = set(chuan_hoa(ch.cau_hoi).split())
    chung = len(tu_hoi & tu_mau) / max(1, len(tu_mau))

    return max(diem_tu_khoa, 0.55 * diem_chuoi + 0.45 * chung)


class ChatbotHoiDap:
    # Hỏi đáp trên ngân hàng câu hỏi, mọi câu trả lời tính từ số liệu thật.

    def __init__(self, kho):
        self.kho = kho
        self.ngan_hang = xay_dung_ngan_hang()

    @property
    def nhom(self) -> dict[str, list[CauHoi]]:
        ra: dict[str, list[CauHoi]] = {}
        for ch in self.ngan_hang:
            ra.setdefault(ch.nhom, []).append(ch)
        return ra

    def khop(self, hoi: str) -> KetQuaKhop:
        if not hoi or not hoi.strip():
            return KetQuaKhop(None, 0.0)
        diem = [(ch, _diem(hoi, ch)) for ch in self.ngan_hang]
        ch, d = max(diem, key=lambda x: x[1])
        return KetQuaKhop(ch, d)

    def tra_loi(self, hoi: str) -> tuple[str, KetQuaKhop]:
        kq = self.khop(hoi)
        if not kq.du_tin_cay:
            goi_y = ", ".join(f"„{c.cau_hoi}“" for c in self.ngan_hang[:3])
            return (
                "Tôi không đủ chắc để trả lời câu này nên xin phép không đoán. "
                f"Ngân hàng câu hỏi hiện có {len(self.ngan_hang)} mục, ví dụ: {goi_y} "
                "Bạn thử chọn một câu gợi ý ở bên dưới.",
                kq,
            )
        return kq.cau_hoi.tra_loi(self.kho), kq
