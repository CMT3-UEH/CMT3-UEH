# Trang 4 — chia dữ liệu chống rò rỉ. Trang demo quan trọng nhất về phương pháp.

import pandas as pd
import streamlit as st

from src.config import TRAIN_END, VALID_END
from src.dataset import make_dataset
from src.ui import bang_dep, ghi_chu, khoi_tao_trang, kho, nguyen, phan_tram, the_chi_so
from src.viz import chia_du_lieu

khoi_tao_trang("Chia dữ liệu chống rò rỉ",
               "Kéo thanh trượt để thấy ngay bao nhiêu dữ liệu phải hy sinh")

k = kho()
ds = k.ds

st.markdown(
    """
Ranh giới chia đặt theo **mốc thời gian cố định**, không theo tỷ lệ. Thêm dữ liệu mới không
làm dịch ranh giới cũ, nên kết quả cũ vẫn so sánh được với kết quả mới.

Quanh mỗi ranh giới có hai lớp bảo vệ:

* **Thanh lọc** — bỏ quan sát huấn luyện có nhãn kéo dài sang khoảng đánh giá. Nhãn 20 phiên
  cần vùng thanh lọc rộng gấp hai mươi lần nhãn 1 phiên.
* **Vùng đệm** — bỏ thêm vài phiên ngay trước ranh giới. Thanh lọc xử lý phần nhãn chồng lấn,
  nhưng không xử lý được chuyện lợi suất và biến động có tự tương quan: phiên huấn luyện cuối
  cùng nằm sát ngay phiên kiểm tra đầu tiên nên hai bên vẫn chia sẻ chung một trạng thái thị trường.
"""
)

c1, c2 = st.columns(2)
with c1:
    tam_nhin = st.select_slider("Tầm dự báo của nhãn (phiên)", [1, 5, 10, 20], value=1)
with c2:
    dem = st.slider("Vùng đệm quanh ranh giới (phiên)", 0, 40, 10, step=5)

d = make_dataset(ds, horizon=tam_nhin, embargo=dem)
hy_sinh = (len(d.split.purged) + len(d.split.embargoed)) / len(ds)

the_chi_so({
    "Huấn luyện": nguyen(len(d.split.train)),
    "Kiểm định": nguyen(len(d.split.valid)),
    "Kiểm tra": nguyen(len(d.split.test)),
    "Dữ liệu hy sinh": phan_tram(hy_sinh),
})

st.plotly_chart(chia_du_lieu(ds.index, d.split, ds["close"]), width="stretch")

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("**Chi tiết lần chia**")
    bang_dep(d.report())
with c2:
    st.markdown("**Khoảng trống giữa các tập**")
    khoang = pd.DataFrame({
        "Số ngày lịch": {
            "Huấn luyện → Kiểm định":
                (d.split.valid.min() - d.split.train.max()).days,
            "Kiểm định → Kiểm tra":
                (d.split.test.min() - d.split.valid.max()).days,
        },
        "Số phiên bị loại": {
            "Huấn luyện → Kiểm định": len(d.split.purged),
            "Kiểm định → Kiểm tra": len(d.split.embargoed),
        },
    })
    bang_dep(khoang)
    ghi_chu(
        f"Mốc cố định: huấn luyện đến {TRAIN_END}, kiểm định đến {VALID_END}, "
        f"phần còn lại là kiểm tra."
    )

st.divider()
st.subheader("Kỷ luật với tập kiểm tra")
st.markdown(
    """
Toàn bộ việc chọn mô hình, siêu tham số, ngưỡng và kích cỡ vị thế đều quyết trên vùng huấn
luyện và kiểm định. Tập kiểm tra chỉ chạy khi cấu hình đã đóng băng.

Mỗi lần chạm tập kiểm tra đều bị ghi vào `reports/test_touches.log`. Đây là bằng chứng khi
hội đồng hỏi mô hình đã được dò tìm bao nhiêu lần trên dữ liệu đánh giá.
"""
)

from src.evaluation.harness import TEST_LOG, count_test_touches

so_lan, so_cau_hinh = count_test_touches()
the_chi_so({
    "Số lần chạm tập kiểm tra": nguyen(so_lan),
    "Số cấu hình đã đánh giá": nguyen(so_cau_hinh),
}, so_cot=2)

if TEST_LOG.exists():
    with st.expander("Xem nhật ký"):
        st.code(TEST_LOG.read_text(encoding="utf-8"), language=None)

if k.co_bang("chia dữ liệu"):
    st.divider()
    st.subheader("Lượng dữ liệu hy sinh theo tầm dự báo của nhãn")
    bang_dep(k.bang["chia dữ liệu"])
