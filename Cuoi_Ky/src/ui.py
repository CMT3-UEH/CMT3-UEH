# Thành phần giao diện dùng chung cho mọi trang Streamlit.

import pandas as pd
import streamlit as st

from src.config import SHARPE_TARGET
from src.formatting import cnt, num, pct

# Tên hiển thị của các phần dữ liệu. Mã nguồn dùng khoá tiếng Anh vì đề bài viết
# train/valid/test, nhưng giao diện thì không được để lộ khoá ra ngoài.
TEN_PHAN = {
    "train": "Huấn luyện",
    "valid": "Kiểm định",
    "test": "Kiểm tra",
    "all": "Toàn kỳ",
}

PHAN_TRAM = {
    "CAGR", "Độ biến động", "Sụt giảm tối đa", "Tỷ lệ phiên tăng",
    "Tỷ lệ thời gian có vị thế", "Tổng chi phí giao dịch", "Lợi suất", "Mốc", "Vượt mốc",
    "Dữ liệu hy sinh", "Tỷ lệ", "Lợi suất quy năm", "Mốc quy năm",
    "Lợi suất tích luỹ", "Lợi suất năm (CAGR)", "Độ biến động năm",
    "Alpha quy năm", "R²", "Tỷ lệ cửa sổ dương", "Chi phí bào mòn mỗi năm",
    "CAGR trước phí", "CAGR sau phí",
}


def khoi_tao_trang(tieu_de: str, mo_ta: str = "") -> None:
    # In tiêu đề trang.
    st.title(tieu_de)
    if mo_ta:
        st.caption(mo_ta)


def kho():
    # Kho dữ liệu, có bộ nhớ đệm của Streamlit.
    return _kho_cached()


@st.cache_resource(show_spinner="Đang nạp dữ liệu và kết quả thí nghiệm ...")
def _kho_cached():
    # Nhập muộn để tệp này không kéo theo cả tầng dữ liệu khi chỉ cần hàm định dạng.
    from src.pipeline import nap_kho

    return nap_kho()


def bang_dep(bang: pd.DataFrame, chieu_cao: int | None = None) -> None:
    # Hiển thị bảng số với định dạng Việt Nam, tự nhận cột phần trăm theo tên.
    hien = bang.copy()
    for cot in hien.columns:
        if not pd.api.types.is_numeric_dtype(hien[cot]):
            continue
        if cot in PHAN_TRAM:
            hien[cot] = hien[cot].map(lambda x: pct(x, 2, vi=True))
        elif hien[cot].abs().max() >= 1000:
            hien[cot] = hien[cot].map(lambda x: num(x, 0, vi=True))
        else:
            hien[cot] = hien[cot].map(lambda x: num(x, 3, vi=True))
    # Streamlit không nhận height=None: chỉ truyền khi thật sự có giá trị.
    them = {"height": chieu_cao} if chieu_cao else {}
    st.dataframe(hien, width="stretch", **them)


def the_chi_so(muc: dict[str, str], so_cot: int = 4) -> None:
    # Hàng thẻ chỉ số ở đầu trang.
    cot = st.columns(min(so_cot, len(muc)))
    for i, (ten, gia_tri) in enumerate(muc.items()):
        with cot[i % len(cot)]:
            st.metric(ten, gia_tri)


# Bộ định dạng số cho thẻ chỉ số. Trước đây mỗi trang tự viết f-string nên bảng thì
# hiện "0,526" theo chuẩn Việt Nam còn thẻ chỉ số lại hiện "0.526" theo chuẩn Anh —
# ngay trên cùng một màn hình. Bốn hàm dưới đây khép lại chỗ hở đó.
def so(x, chu_so: int = 3) -> str:
    # Số thực theo chuẩn Việt Nam: dấu phẩy thập phân, dấu chấm hàng nghìn.
    return num(x, chu_so, vi=True)


def phan_tram(x, chu_so: int = 2) -> str:
    return pct(x, chu_so, vi=True)


def nguyen(x) -> str:
    return cnt(x, vi=True)


def chon_phan(nhan: str, cac_phan: list[str], mac_dinh: int = 0,
              dang: str = "radio", khoa: str | None = None) -> str:
    # Bộ chọn giai đoạn dữ liệu, hiện tên tiếng Việt nhưng trả về khoá nội bộ.
    hien = [TEN_PHAN.get(p, p) for p in cac_phan]
    ham = st.radio if dang == "radio" else st.selectbox
    chon = ham(nhan, hien, index=mac_dinh, horizontal=(dang == "radio"), key=khoa)         if dang == "radio" else ham(nhan, hien, index=mac_dinh, key=khoa)
    return cac_phan[hien.index(chon)]


def canh_bao_thieu(ten_bang: str, lenh: str) -> bool:
    # Báo cho người dùng biết cần chạy giai đoạn nào để có dữ liệu cho trang này.
    if kho().co_bang(ten_bang):
        return False
    st.warning(
        f"Chưa có kết quả **{ten_bang}**. Chạy lệnh sau rồi tải lại trang:\n\n"
        f"```bash\n{lenh}\n```"
    )
    return True


def muc_tieu_sharpe(gia_tri: float) -> None:
    # Hiển thị khoảng cách tới chỉ tiêu của đề bài, không tô hồng.
    dat = gia_tri >= SHARPE_TARGET
    if dat:
        st.success(f"Sharpe {so(gia_tri)} — đạt chỉ tiêu ≥ {so(SHARPE_TARGET, 1)}.")
    else:
        st.info(
            f"Sharpe {so(gia_tri)} — chỉ tiêu đề bài là ≥ {so(SHARPE_TARGET, 1)}, "
            f"còn thiếu {so(SHARPE_TARGET - gia_tri)} đơn vị."
        )


def ghi_chu(noi_dung: str) -> None:
    # Khối ghi chú phương pháp, dùng để giải thích chỗ dễ hiểu nhầm.
    st.caption(noi_dung)


def thanh_ben_trang_thai() -> None:
    # Thanh bên hiển thị giai đoạn nào đã chạy xong.
    from src.pipeline import trang_thai_giai_doan

    with st.sidebar:
        st.subheader("Trạng thái thí nghiệm")
        st.dataframe(trang_thai_giai_doan(), width="stretch")
        st.caption(
            "Mỗi giai đoạn chạy bằng một lệnh trong `src/experiments/`. "
            "Trang nào thiếu dữ liệu sẽ báo đúng lệnh cần chạy."
        )
