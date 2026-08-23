# Trang 11 — biện luận đầu tư và hỏi đáp.

import streamlit as st

from src.chatbot.engine import ChatbotHoiDap
from src.config import CAPITAL_VND, MAX_ADV_PARTICIPATION, TICKER
from src.formatting import money_vnd, pct
from src.ui import bang_dep, ghi_chu, khoi_tao_trang, kho, phan_tram, the_chi_so

khoi_tao_trang("Biện luận đầu tư và hỏi đáp",
               "Lợi nhuận đến từ đâu, dùng được đến đâu, và hỏng khi nào")

k = kho()

st.subheader("Alpha hay chỉ là beta")
if k.co_bang("alpha beta"):
    bang_dep(k.bang["alpha beta"])
    st.markdown(
        """
Ba đặc tả hồi quy lồng nhau, cột giữa là cột khắt khe nhất. Chiến lược chỉ giao dịch FPT nên
mốc so sánh đúng **không phải chỉ số thị trường mà là chính FPT**: nếu alpha biến mất khi
thêm lợi suất FPT vào vế phải thì cái gọi là kỹ năng định thời điểm chỉ là việc nắm giữ
một cổ phiếu đang tăng.

Sai số chuẩn dùng Newey–West vì phần dư của chiến lược giao dịch gần như luôn có tự tương
quan và phương sai thay đổi; sai số chuẩn thường sẽ thổi phồng ý nghĩa thống kê của alpha,
đúng theo hướng có lợi cho người viết báo cáo.
"""
    )
else:
    st.info("Chạy `python -m src.experiments.run_final` để có bảng hồi quy nhân tố.")

if k.co_bang("theo năm"):
    st.divider()
    st.subheader("Lợi nhuận có đều không, hay chỉ một năm gánh cả kỳ")
    bang_dep(k.bang["theo năm"])

if k.co_bang("theo chế độ"):
    st.subheader("Hiệu quả theo chế độ thị trường")
    bang_dep(k.bang["theo chế độ"])
    ghi_chu(
        "Với chiến lược chỉ mua hoặc đứng ngoài, phần lớn giá trị thường đến từ việc "
        "**tránh đúng lúc** trong thị trường giảm chứ không từ việc chọn đúng điểm vào."
    )

st.divider()
st.subheader("Sức chứa vốn của chiến lược")
ten = next((c for c in k.tin_hieu.columns if c.startswith("ATFN-ABCD")), None)
if ten:
    from src.backtest.constraints import average_daily_value

    adv = average_daily_value(k.ds["close"], k.ds["volume"]).loc[k.d.index("test")]
    kq = k.chay(ten, "test")
    lenh_tb = kq.turnover[kq.turnover > 0].mean()
    suc_chua = MAX_ADV_PARTICIPATION * adv.median() / max(lenh_tb, 1e-9)

    the_chi_so({
        "Giá trị giao dịch bình quân ngày": money_vnd(adv.median()),
        "Quy mô lệnh trung bình": f"{phan_tram(lenh_tb, 1)} vốn",
        "Vốn tối đa ước tính": money_vnd(suc_chua),
        "Vốn đang giả định": money_vnd(CAPITAL_VND),
    })
    st.markdown(
        f"""
Với giới hạn tham gia {pct(MAX_ADV_PARTICIPATION, 0, vi=True)} giá trị giao dịch bình quân ngày
và quy mô lệnh trung bình {pct(lenh_tb, 1, vi=True)} vốn, chiến lược quản lý được khoảng
**{money_vnd(suc_chua)}** trước khi phải chia nhỏ lệnh qua nhiều phiên.

Con số này chỉ đúng cho giai đoạn gần đây. Năm 2008 giá trị giao dịch của {TICKER} chỉ quanh
1 tỷ đồng một phiên, nên cùng chiến lược đó chỉ chạy được với quy mô vốn nhỏ hơn khoảng hai bậc độ lớn.
"""
    )

st.divider()
st.subheader("Rủi ro và điều kiện thất bại")
st.markdown(
    """
| Rủi ro | Vì sao đáng lo | Dấu hiệu để dừng dùng mô hình |
|---|---|---|
| Tập trung một cổ phiếu | Không có đa dạng hoá, một sự kiện doanh nghiệp đủ phá cả chiến lược | Biến động riêng của FPT vượt xa mức lịch sử |
| Thay đổi chế độ thị trường | Mô hình học từ 2008–2017, cấu trúc thị trường Việt Nam đã đổi nhiều | Sharpe trượt 252 phiên âm liên tục hai quý |
| Quá khớp | Sai số chuẩn của Sharpe lớn hơn phần lớn chênh lệch giữa các mô hình | Deflated Sharpe tụt dưới 0,5 khi cập nhật dữ liệu |
| Giới hạn thanh khoản | Ràng buộc giá trị giao dịch siết rất mạnh ở giai đoạn đầu | Tỷ lệ phiên bị ràng buộc chặn lại tăng nhanh |
| Chỉ mua hoặc đứng ngoài | Không kiếm được gì trong thị trường giảm dài | Thị trường vào chu kỳ giảm nhiều năm |
"""
)

st.subheader("Khuyến nghị sử dụng")
st.markdown(
    """
1. **Dùng như công cụ hỗ trợ quyết định**, không phải hệ thống giao dịch tự động. Con số
   Sharpe trên một tài sản đơn lẻ có sai số chuẩn quá lớn để giao phó vốn thật một cách máy móc.
2. **Tái huấn luyện hằng năm** theo đúng lịch của kiểm định cuốn chiếu đã dùng trong đồ án.
3. **Giám sát ba chỉ số khi chạy thật**: Sharpe trượt 252 phiên, tỷ lệ phiên bị ràng buộc
   thanh khoản chặn, và độ lệch giữa nhịp giao dịch thực tế với nhịp mô hình dự kiến.
4. **Ghi rõ trong mọi báo cáo** rằng chiến lược chỉ mua hoặc đứng ngoài trên một cổ phiếu,
   nên không thay thế được một danh mục đa dạng hoá.
"""
)

st.divider()
st.subheader("Hỏi đáp")

bot = ChatbotHoiDap(k)
nhom = bot.nhom

c1, c2 = st.columns([1, 2])
with c1:
    nhom_chon = st.selectbox("Nhóm câu hỏi", list(nhom.keys()))
    goi_y = st.radio("Câu hỏi gợi ý", [c.cau_hoi for c in nhom[nhom_chon]],
                     label_visibility="collapsed")
with c2:
    hoi = st.text_input("Hoặc gõ câu hỏi của bạn", value="")
    cau = hoi.strip() or goi_y
    tra_loi, kq = bot.tra_loi(cau)

    st.markdown(f"**Câu hỏi:** {cau}")
    if kq.du_tin_cay:
        st.success(tra_loi)
        st.caption(f"Khớp với mục „{kq.cau_hoi.cau_hoi}“ — độ tin cậy {phan_tram(kq.diem, 0)}")
    else:
        st.warning(tra_loi)

ghi_chu(
    f"Ngân hàng có {len(bot.ngan_hang)} câu, mỗi câu ánh xạ tới một hàm tính toán thật trên "
    "kho dữ liệu chứ không có câu trả lời viết sẵn. Chatbot được phép từ chối khi không đủ "
    "chắc — thà nói không biết còn hơn bịa số liệu."
)
