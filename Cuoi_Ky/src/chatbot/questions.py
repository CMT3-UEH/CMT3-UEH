# Ngân hàng câu hỏi có kiểm soát cho phần hỏi đáp.

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.config import SHARPE_TARGET, TICKER
from src.evaluation.metrics import (
    annual_return,
    annual_volatility,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from src.formatting import num, pct

MO_HINH_DE_XUAT = "ATFN-ABCD"


@dataclass
class CauHoi:
    # Một câu hỏi trong ngân hàng, kèm hàm sinh câu trả lời.

    nhom: str
    cau_hoi: str
    tu_khoa: tuple[str, ...]
    tra_loi: Callable


def _ten_de_xuat(k) -> str | None:
    return next((c for c in k.tin_hieu.columns if c.startswith(MO_HINH_DE_XUAT)), None)


def _loi_suat(k, ten: str, part: str = "test") -> pd.Series:
    return k.chay(ten, part).returns.iloc[1:]


# Nhóm dữ liệu
def _so_phien(k) -> str:
    return (f"Bộ dữ liệu có **{num(len(k.ds), 0, vi=True)} phiên** dùng được, từ "
            f"{k.ds.index.min():%d/%m/%Y} đến {k.ds.index.max():%d/%m/%Y}. "
            f"Phần trước đó bị cắt vì các đặc trưng khung tháng cần 13 kỳ lịch sử.")


def _nguon_du_lieu(k) -> str:
    return ("Hai nguồn bù nhau: **vnstock** có giá cổ phiếu từ ngày niêm yết nhưng "
            "thiếu 13 phiên chỉ số năm 2008; **DNSE entrade** có chỉ số liên tục từ "
            "2006 nhưng cổ phiếu chỉ từ 2012. Nhân tố vĩ mô lấy từ Yahoo Finance, đã "
            "lọc giá trị sai thang đo và giá dầu âm ngày 20/04/2020.")


def _so_dac_trung(k) -> str:
    nhom = k.nhom_dac_trung
    chi_tiet = ", ".join(f"{t} ({len(c)})" for t, c in nhom.items())
    return (f"Có **{len(k.features)} đặc trưng** chia thành {len(nhom)} nhóm: {chi_tiet}. "
            f"Tất cả đã qua kiểm định tính nhân quả bằng thực nghiệm.")


def _chia_du_lieu(k) -> str:
    s = k.d.split
    hy_sinh = (len(s.purged) + len(s.embargoed)) / len(k.ds)
    return (f"Huấn luyện **{num(len(s.train), 0, vi=True)} phiên** "
            f"({s.train.min():%m/%Y}–{s.train.max():%m/%Y}), kiểm định "
            f"**{num(len(s.valid), 0, vi=True)}** ({s.valid.min():%m/%Y}–{s.valid.max():%m/%Y}), "
            f"kiểm tra **{num(len(s.test), 0, vi=True)}** "
            f"({s.test.min():%m/%Y}–{s.test.max():%m/%Y}). "
            f"Thanh lọc và vùng đệm làm hy sinh {pct(hy_sinh, 2, vi=True)} dữ liệu.")


# Nhóm mốc tham chiếu
def _moc_mua_giu(k) -> str:
    r = k.gia("test")["close"].pct_change().dropna()
    return (f"Mua và nắm giữ {TICKER} trên tập kiểm tra đạt CAGR "
            f"**{pct(annual_return(r), 2, vi=True)}**, độ biến động "
            f"{pct(annual_volatility(r), 2, vi=True)}, Sharpe "
            f"**{num(sharpe_ratio(r), 3, vi=True)}**, sụt giảm tối đa "
            f"{pct(max_drawdown(r), 2, vi=True)}. Đây là mốc mà mọi mô hình phải vượt.")


def _chi_phi_giao_dich(k) -> str:
    if not k.co_bang("chi phí theo nhịp"):
        return "Chưa chạy báo cáo chi phí. Chạy `python -m src.experiments.report_data`."
    b = k.bang["chi phí theo nhịp"]
    c = "Chi phí bào mòn mỗi năm"
    return (f"Với cùng tỷ trọng nắm giữ và cùng lợi suất trước phí, đảo vị thế mỗi phiên "
            f"tốn **{pct(b[c].max(), 2, vi=True)}/năm** còn đảo mỗi quý chỉ tốn "
            f"**{pct(b[c].min(), 2, vi=True)}/năm**. Chênh lệch đó chính là cái giá của "
            f"việc giao dịch dày, và là lý do đề tài để mô hình tự chọn nhịp.")


# Nhóm mô hình
def _mo_hinh_de_xuat(k) -> str:
    return ("**ATFN** — Adaptive-Tempo Fusion Network. Bốn thành phần: (A) bộ mã hoá đa "
            "phân giải TCN giãn nở song song GRU; (B) đầu ra đa tầm dự báo ở 1/5/10/20 phiên "
            "kèm cổng chọn nhịp; (C) cổng theo chế độ thị trường dạng hỗn hợp chuyên gia; "
            "(D) vùng không giao dịch có ngưỡng học được, huấn luyện bằng hàm mất mát "
            "Sharpe đã trừ chi phí.")


def _ket_qua_de_xuat(k) -> str:
    ten = _ten_de_xuat(k)
    if ten is None:
        return "Chưa huấn luyện mô hình đề xuất. Chạy `python -m src.experiments.run_atfn`."
    r = _loi_suat(k, ten, "test")
    s = sharpe_ratio(r)
    dat = "đạt" if s >= SHARPE_TARGET else "chưa đạt"
    return (f"Trên tập kiểm tra, {ten} cho Sharpe **{num(s, 3, vi=True)}**, CAGR "
            f"{pct(annual_return(r), 2, vi=True)}, Sortino {num(sortino_ratio(r), 3, vi=True)}, "
            f"sụt giảm tối đa {pct(max_drawdown(r), 2, vi=True)}. "
            f"So với chỉ tiêu đề bài ≥ {num(SHARPE_TARGET, 1, vi=True)} thì **{dat}**.")


def _nhip_giao_dich(k) -> str:
    ten = _ten_de_xuat(k)
    if ten is None:
        return "Chưa có mô hình đề xuất."
    n = k.chay(ten, "test").tempo()
    return (f"Mô hình tự chọn nhịp **{num(n['Số lần khớp lệnh mỗi năm'], 1, vi=True)} lệnh "
            f"mỗi năm**, mỗi lần nắm giữ trung bình "
            f"{num(n['Thời gian nắm giữ trung bình (phiên)'], 0, vi=True)} phiên, ở trong thị trường "
            f"{pct(n['Tỷ lệ thời gian có vị thế'], 1, vi=True)} thời gian. "
            f"Đây là kết quả của mô hình chứ không phải tham số cài sẵn.")


def _co_che_nhip(k) -> str:
    return ("Bốn cơ chế: (1) phạt vòng quay vị thế ngay trong hàm mất mát nên mỗi lần đổi vị thế "
            "đều phải trả giá; (2) vùng không giao dịch với ngưỡng τ do mạng sinh ra theo "
            "từng phiên, triển khai tuần tự nên có đạo hàm; (3) đầu ra đa tầm dự báo kèm cổng "
            "softmax chọn chu kỳ đáng tin; (4) cổng chế độ thị trường thu hẹp tỷ trọng nắm giữ khi "
            "trạng thái bất lợi.")


def _baseline_tot_nhat(k) -> str:
    if not k.co_bang("kết quả test"):
        return "Chưa chạy đánh giá cuối. Chạy `python -m src.experiments.run_final`."
    b = k.bang["kết quả test"].sort_values("Sharpe", ascending=False)
    dong = b.index[0]
    return (f"Trên tập kiểm tra, chiến lược có Sharpe cao nhất là **{dong}** với "
            f"{num(b.loc[dong, 'Sharpe'], 3, vi=True)}. Nhưng cần nhớ sai số chuẩn của "
            f"Sharpe trên 1.404 phiên vào khoảng 0,43 đơn vị, nên thứ hạng sát nhau "
            f"không kết luận được gì.")


# Nhóm phương pháp
def _chong_ro_ri(k) -> str:
    return ("Bảy lớp: quy ước tín hiệu ở close(t) khớp lệnh ở open(t+1); thanh lọc theo t1 "
            "của nhãn; vùng đệm quanh ranh giới; chuẩn hoá chỉ ước lượng trên tập huấn "
            "luyện; đặc trưng khung tuần/tháng và nhóm vĩ mô bị đẩy lùi một kỳ; kiểm định "
            "tính nhân quả bằng thực nghiệm cho cả 106 đặc trưng; và phép thử làm chậm tín "
            "hiệu — Sharpe phải giảm, nếu tăng thì chắc chắn có rò rỉ.")


def _do_tin_cay(k) -> str:
    if not k.co_bang("độ tin cậy valid"):
        return "Chưa chạy phần baseline."
    tc = k.bang["độ tin cậy valid"]
    return (f"Sai số chuẩn trung bình của Sharpe trên tập kiểm định là "
            f"**{num(tc['Sai số chuẩn'].mean(), 3, vi=True)}**, trong khi cả bảng xếp hạng "
            f"chỉ trải {num(tc['Sharpe'].max() - tc['Sharpe'].min(), 3, vi=True)} đơn vị. "
            f"Không chiến lược nào có t-stat vượt 2. Đây là lý do phần chọn siêu tham số "
            f"đã chuyển sang kiểm định cuốn chiếu trên 15 cửa sổ.")


def _tai_sao_khong_do_kien_truc(k) -> str:
    return ("Vì dò kiến trúc trên nền nhiễu này chỉ là xếp hạng may rủi. Sai số chuẩn của "
            "Sharpe ở vùng chọn mô hình khoảng 0,4–0,6 đơn vị, lớn hơn mọi khác biệt giữa "
            "các kiến trúc. Dò nhiều còn thổi phồng số phép thử, khiến hiệu chỉnh Deflated "
            "Sharpe về sau mất ý nghĩa. Thay vào đó kích thước được cố định theo quy mô dữ "
            "liệu và phương sai được giảm bằng cách trung bình 5 hạt giống.")


def _chi_phi_mo_phong(k) -> str:
    from src.backtest.costs import BASE_COST

    return (f"Phí công ty chứng khoán {pct(BASE_COST.fee, 2, vi=True)} mỗi chiều, thuế bán "
            f"{pct(BASE_COST.sell_tax, 2, vi=True)}, trượt giá "
            f"{pct(BASE_COST.slippage, 2, vi=True)} mỗi chiều. Một vòng mua rồi bán tốn "
            f"**{pct(BASE_COST.round_trip, 2, vi=True)}**. Ngoài ra còn mô phỏng chu kỳ "
            f"T+2,5, biên độ ±7%, giới hạn 10% giá trị giao dịch bình quân ngày, và phần vốn "
            f"đứng ngoài có hưởng lãi phi rủi ro.")


def _alpha_hay_beta(k) -> str:
    if not k.co_bang("alpha beta"):
        return "Chưa chạy đánh giá cuối. Chạy `python -m src.experiments.run_final`."
    b = k.bang["alpha beta"]
    dong = "CAPM mở rộng với lợi suất FPT"
    if dong not in b.index:
        dong = b.index[0]
    a = b.loc[dong, "Alpha quy năm"]
    p = b.loc[dong, "p-value của alpha"]
    ket = "có ý nghĩa thống kê" if p < 0.05 else "chưa có ý nghĩa thống kê"
    return (f"Hồi quy lợi suất chiến lược theo thị trường **và theo chính FPT** (sai số "
            f"chuẩn Newey–West) cho alpha {pct(a, 2, vi=True)}/năm với p-value "
            f"{num(p, 4, vi=True)} — **{ket}**. Đây là phép kiểm khắt khe nhất: nếu alpha "
            f"biến mất khi thêm lợi suất FPT vào vế phải thì cái gọi là kỹ năng định thời "
            f"điểm chỉ là việc nắm giữ một cổ phiếu đang tăng.")


def _rui_ro(k) -> str:
    return ("Bốn rủi ro chính: (1) tập trung toàn bộ vào một cổ phiếu duy nhất, không có đa "
            "dạng hoá; (2) thay đổi chế độ thị trường — mô hình học từ 2008–2017 có thể "
            "không còn đúng; (3) quá khớp, đã đo bằng PBO và Deflated Sharpe nhưng không "
            "loại trừ được hoàn toàn; (4) giới hạn năng lực — ràng buộc 10% giá trị giao "
            "dịch bình quân siết rất mạnh ở giai đoạn 2008–2012.")


def _han_che(k) -> str:
    return ("Ba hạn chế cần ghi rõ trong báo cáo: chỉ một tài sản nên không tận dụng được "
            "đa dạng hoá và không trung hoà được beta thị trường; chỉ mua hoặc đứng ngoài "
            "nên không kiếm được gì trong thị trường giảm; và tập kiểm tra 1.404 phiên vẫn "
            "quá ngắn để Sharpe có sai số chuẩn nhỏ hơn 0,4 đơn vị.")


def xay_dung_ngan_hang() -> list[CauHoi]:
    # Ngân hàng câu hỏi.
    return [
        CauHoi("Dữ liệu", "Bộ dữ liệu có bao nhiêu phiên và từ khi nào?",
               ("bao nhiêu phiên", "số phiên", "khoảng thời gian", "dữ liệu từ"), _so_phien),
        CauHoi("Dữ liệu", "Dữ liệu lấy từ nguồn nào?",
               ("nguồn dữ liệu", "lấy dữ liệu", "vnstock", "entrade"), _nguon_du_lieu),
        CauHoi("Dữ liệu", "Có bao nhiêu đặc trưng và chia thành những nhóm nào?",
               ("bao nhiêu đặc trưng", "nhóm đặc trưng", "feature"), _so_dac_trung),
        CauHoi("Dữ liệu", "Dữ liệu được chia huấn luyện, kiểm định và kiểm tra thế nào?",
               ("chia dữ liệu", "train valid test", "tập kiểm tra từ"), _chia_du_lieu),
        CauHoi("Mốc tham chiếu", "Mua và nắm giữ FPT cho kết quả bao nhiêu?",
               ("mua và nắm giữ", "buy and hold", "mốc tham chiếu"), _moc_mua_giu),
        CauHoi("Mốc tham chiếu", "Chi phí giao dịch ảnh hưởng thế nào theo nhịp?",
               ("chi phí theo nhịp", "phí ăn mất", "giao dịch dày"), _chi_phi_giao_dich),
        CauHoi("Mô hình", "Mô hình đề xuất là gì?",
               ("mô hình đề xuất", "atfn", "kiến trúc"), _mo_hinh_de_xuat),
        CauHoi("Mô hình", "Mô hình đề xuất đạt kết quả bao nhiêu trên tập kiểm tra?",
               ("kết quả mô hình", "sharpe của mô hình", "đạt chỉ tiêu"), _ket_qua_de_xuat),
        CauHoi("Mô hình", "Mô hình giao dịch bao nhiêu lần mỗi năm và nắm giữ bao lâu?",
               ("nhịp giao dịch", "bao nhiêu lệnh", "tần suất", "nắm giữ bao lâu"),
               _nhip_giao_dich),
        CauHoi("Mô hình", "Mô hình quyết định tần suất giao dịch bằng cách nào?",
               ("cơ chế nhịp", "làm sao tự quyết", "vùng không giao dịch"), _co_che_nhip),
        CauHoi("Mô hình", "Chiến lược nào tốt nhất trên tập kiểm tra?",
               ("tốt nhất", "cao nhất", "xếp hạng"), _baseline_tot_nhat),
        CauHoi("Phương pháp", "Đồ án chống rò rỉ dữ liệu bằng cách nào?",
               ("rò rỉ", "leakage", "chống rò"), _chong_ro_ri),
        CauHoi("Phương pháp", "Kết quả có đáng tin về mặt thống kê không?",
               ("đáng tin", "sai số chuẩn", "ý nghĩa thống kê", "t-stat"), _do_tin_cay),
        CauHoi("Phương pháp", "Vì sao không dò tìm siêu tham số kiến trúc?",
               ("không dò", "siêu tham số kiến trúc", "vì sao cố định"),
               _tai_sao_khong_do_kien_truc),
        CauHoi("Phương pháp", "Backtest mô phỏng những chi phí và ràng buộc gì?",
               ("chi phí mô phỏng", "phí bao nhiêu", "ràng buộc", "t+2"), _chi_phi_mo_phong),
        CauHoi("Biện luận", "Lợi nhuận là alpha thật hay chỉ là beta?",
               ("alpha hay beta", "alpha thật", "hồi quy nhân tố"), _alpha_hay_beta),
        CauHoi("Biện luận", "Chiến lược có những rủi ro gì?",
               ("rủi ro", "risk", "nguy cơ"), _rui_ro),
        CauHoi("Biện luận", "Đồ án có hạn chế gì?",
               ("hạn chế", "điểm yếu", "chưa làm được"), _han_che),
    ]
