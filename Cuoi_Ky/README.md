# Giao dịch thuật toán trên cổ phiếu FPT

Đồ án cuối kỳ học phần Nghiên cứu khoa học. Hệ thống giao dịch thuật toán trên dữ liệu thực của
thị trường Việt Nam, nghiên cứu điển hình cổ phiếu **FPT** với **4.623 phiên giao dịch** dùng được
kể từ 31/01/2008.

Điểm khác biệt của đề tài: **nhịp giao dịch không được cài sẵn mà do mô hình tự quyết**. Số lệnh
mỗi năm và thời gian nắm giữ trung bình là kết quả nghiên cứu chứ không phải siêu tham số.

Dự án này độc lập hoàn toàn với phần giữa kỳ. Các mô-đun tái dùng đã được sao chép sang, không
import chéo thư mục.

---

## Chạy nhanh

Yêu cầu **Python 3.10 trở lên**.

**Cách 1 — một lệnh (Windows):**

```bat
run.bat            chạy những bước còn thiếu rồi mở ứng dụng
run.bat app        chỉ mở ứng dụng          (~40 giây)
run.bat kiem-tra   chạy lại 6 phép tự kiểm bộ máy backtest
run.bat tat-ca     xoá cache, huấn luyện lại từ đầu  (~1,5 giờ)
```

**Cách 2 — notebook trình bày toàn bộ đồ án:**

```bat
.venv\Scripts\activate
jupyter notebook DoAn_CuoiKy.ipynb
```

Notebook đi qua đủ 10 bước theo đúng thứ tự đề bài. Mặc định dùng lại kết quả đã lưu nên chạy
hết trong vài phút; đổi `CHAY_LAI_HUAN_LUYEN = True` ở ô đầu để huấn luyện lại từ đầu (~1,5 giờ).

**Cách 3 — thủ công từng bước:**

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows
# source .venv/bin/activate                # macOS / Linux

pip install -r requirements.txt
python fetch_data.py                        # 1. tải dữ liệu          ~1 phút
python -m src.experiments.check_engine      # 2. tự kiểm backtest      ~3 phút
python -m src.experiments.report_data       # 3. dữ liệu và mốc        ~2 phút
python -m src.experiments.run_baselines     # 4. baseline cổ điển      ~5 phút
python -m src.experiments.run_deep          # 5. baseline học sâu      ~25 phút
python -m src.experiments.run_atfn          # 6. ATFN và ablation      ~50 phút
python -m src.experiments.run_final         # 7. đánh giá trên test    ~3 phút
streamlit run app.py                        # 8. mở ứng dụng
```

Bước 4, 5, 6 độc lập nhau nên chạy thứ tự nào cũng được, nhưng bước 7 cần cả ba.
Bước 2 là cổng chặn: bộ máy backtest không đạt thì mọi kết quả sau đó vô nghĩa.
Bước 5 và 6 lưu kết quả sau **từng mô hình**, dừng giữa chừng chạy lại chỉ mất mô hình đang dở.

---

## Bài toán

| | |
|---|---|
| **Input** | Đặc trưng `x[t]` tính từ thông tin đến hết giá đóng cửa phiên `t`; cửa sổ nhìn lại 60 phiên |
| **Output** | Vị thế `w[t+1] ∈ [0, 1]` — tỷ trọng vốn nắm FPT, phần còn lại giữ tiền mặt hưởng lãi phi rủi ro |
| **Thực thi** | Tín hiệu ở `close(t)` → lệnh khớp ở `open(t+1)` → lợi suất ghi cho phiên `t+1` |
| **Độ đo chính** | Sharpe ròng phí trên tập kiểm tra, mục tiêu ≥ 1,8 |
| **Độ đo nhịp** | Số lệnh mỗi năm, thời gian nắm giữ trung bình, turnover |

---

## Cấu trúc mã nguồn

```
final_algo_trading/
├── fetch_data.py            tải dữ liệu và dựng bảng giá tổng hợp
├── run.bat                  chạy toàn bộ dự án bằng một lệnh
└── src/
    ├── config.py            mọi hằng số của dự án nằm ở đúng một nơi
    ├── console.py           cho phép in tiếng Việt trên Windows
    ├── dataset.py           chia train/valid/test và chuẩn hoá chống rò rỉ
    ├── splits.py            thanh lọc, cách ly, kiểm định tiến dần
    ├── data/loaders.py      thu thập giá và nhân tố vĩ mô từ hai nguồn bù nhau
    ├── features/            7 nhóm đặc trưng + gán nhãn đa tầm dự báo
    ├── backtest/            bộ máy backtest, chi phí, ràng buộc, định cỡ vị thế
    ├── evaluation/          các độ đo hiệu quả và rủi ro
    └── experiments/         các kịch bản chạy thí nghiệm
```

---

## Bộ đặc trưng — 106 biến, 9 nhóm

| Nhóm | Số biến | Nội dung |
|---|---|---|
| Giá | 19 | Momentum 1–250 phiên, khoảng cách và độ dốc MA, hồi quy về trung bình, MACD |
| Biến động | 7 | Realized vol 20/60/120, Parkinson, Garman–Klass, ATR |
| Thanh khoản | 5 | z-score giá trị giao dịch, OBV, tỷ lệ khối lượng, kém thanh khoản Amihud |
| Vi cấu trúc | 11 | Vị trí đóng cửa trong biên độ, thân nến, râu nến, khoảng trống, phiên chạm trần/sàn |
| Đa khung | 14 | Bộ đặc trưng tính trên khung tuần và khung tháng, đã đẩy lùi một kỳ |
| Quan hệ thị trường | 13 | Sức mạnh tương đối so với VNINDEX, beta và alpha trượt, tương quan trượt |
| Chế độ | 16 | Biến động và sụt giảm của cổ phiếu lẫn thị trường, trạng thái so với MA200 |
| Vĩ mô | 16 | USD/VND, dầu, vàng, S&P 500 — đều trễ một phiên vì lệch múi giờ |
| Lịch | 5 | Thứ trong tuần và tháng, mã hoá vòng tròn |

Toàn bộ 106 đặc trưng đã qua kiểm định tính nhân quả bằng thực nghiệm: cắt chuỗi tại một phiên
bất kỳ rồi tính lại, giá trị phải không đổi. Xem `src/features/builder.py::assert_causal`.

---

## Chia dữ liệu

| | Số phiên | Khoảng thời gian |
|---|---|---|
| Huấn luyện | 2.457 | 31/01/2008 – 13/12/2017 |
| Kiểm định | 738 | 02/01/2018 – 15/12/2020 |
| Kiểm tra | 1.404 | 04/01/2021 – 21/08/2026 |

Ranh giới đặt theo mốc thời gian cố định, không theo tỷ lệ, để thêm dữ liệu mới không làm dịch
ranh giới cũ. Quanh mỗi ranh giới có thanh lọc theo `t1` của nhãn và vùng đệm 10 phiên. Tổng dữ
liệu hy sinh: 0,52% với nhãn 1 phiên, 1,34% với nhãn 20 phiên.

---

## Kết quả đã có

### Mốc mua và nắm giữ FPT, đã trừ phí

| Giai đoạn | CAGR | Độ biến động | Sharpe | Sụt giảm tối đa |
|---|---|---|---|---|
| Huấn luyện | 7,14% | 30,83% | 0,235 | −75,5% |
| Kiểm định | 16,06% | 26,95% | 0,524 | −33,8% |
| **Kiểm tra** | **20,39%** | **28,83%** | **0,635** | **−52,1%** |
| Toàn kỳ | 13,37% | 29,59% | 0,423 | −75,5% |

Chỉ tiêu đề bài là Sharpe ≥ 1,8, nên mô hình cần tạo thêm **+1,165 đơn vị Sharpe** so với việc
chỉ mua rồi để yên.

### Chi phí giao dịch theo nhịp — trên tập kiểm tra

Mọi dòng đều nắm giữ khoảng một nửa thời gian và có cùng lợi suất trước phí, nên chênh lệch giữa
các dòng đúng bằng cái giá của việc giao dịch dày hơn.

| Nhịp đảo vị thế | Số lệnh mỗi năm | CAGR trước phí | CAGR sau phí | Phí ăn mất mỗi năm |
|---|---|---|---|---|
| Mỗi phiên | 84,0 | 12,66% | −8,72% | **21,37%** |
| Mỗi tuần | 50,4 | 12,26% | −1,05% | **13,31%** |
| Mỗi tháng | 12,7 | 12,53% | 9,02% | **3,51%** |
| Mỗi quý | 4,3 | 12,60% | 11,40% | **1,20%** |

Đây là lập luận kinh tế nền của đề tài: giao dịch mỗi phiên tốn gấp gần 18 lần giao dịch mỗi quý.
Tín hiệu phải rất mạnh mới bù nổi chi phí, và mô hình cần học được chính điều đó thay vì bị ép
một nhịp cố định.

---

## Bộ máy backtest

Mọi mô hình đi qua đúng một hàm `run_backtest`. Bộ máy tự kiểm bằng 6 phép thử, chạy bằng
`python -m src.experiments.check_engine`:

| # | Phép thử | Kết quả |
|---|---|---|
| 1 | Nắm giữ toàn phần khớp lợi suất cổ phiếu | Lệch 2,2·10⁻¹⁶ |
| 2 | Đứng ngoài chỉ hưởng lãi phi rủi ro | Đúng 4,5000%/năm |
| 3 | Vị thế ngẫu nhiên không sinh kỹ năng định thời điểm | Lệch mốc tĩnh −0,10%/năm, trong sai số |
| 4 | Tăng phí làm giảm lợi nhuận | Đơn điệu qua 4 mức phí |
| 5 | Làm chậm tín hiệu làm hỏng kết quả | Sharpe 3,02 → −0,04 khi trễ thêm 1 phiên |
| 6 | Chu kỳ T+2.5 chặn bớt lệnh | Giảm 66,6% số lệnh |

Các giả định được mô phỏng:

- Phí 0,15%/chiều, thuế bán 0,1%, trượt giá 0,05%/chiều
- Lợi suất tách hai đoạn qua đêm và trong phiên rồi nhân với nhau, không cộng
- Vị thế trôi theo giá, chi phí chỉ tính trên phần thật sự phải khớp lệnh
- Chu kỳ thanh toán T+2.5 khoá riêng phần vừa mua và phần tiền vừa bán, không khoá cả tài khoản
- Giới hạn 10% giá trị giao dịch bình quân 20 phiên, vốn giả định 1 tỷ đồng
- Phần vốn đứng ngoài hưởng lãi suất phi rủi ro 4,5%/năm
- Phiên khoá trần hoặc khoá sàn không khớp được lệnh

---

## Kết quả cuối cùng trên tập kiểm tra

Tập kiểm tra 04/01/2021 – 21/08/2026, 1.404 phiên. Chạy đúng **một lần**, cấu hình đã đóng băng
từ trước. Mô hình đề xuất được công bố là **ATFN-ABCD** trước khi chạy, không phải chọn sau khi
nhìn kết quả.

| Chiến lược | CAGR | Sharpe | KTC 95% | Deflated Sharpe | MaxDD | Lệnh/năm |
|---|---|---|---|---|---|---|
| A6 · Momentum tuyệt đối 12–1 | 28,8% | **1,137** | [0,36 ; 1,86] | 0,925 | −19,4% | 3,2 |
| A8b · Cổ phiếu và thị trường trên MA200 | 21,6% | 0,926 | [0,05 ; 1,63] | 0,841 | −31,1% | 5,7 |
| A4 · Giao cắt trung bình động | 24,8% | 0,902 | [0,10 ; 1,66] | 0,794 | −29,2% | 2,5 |
| B4c · GRU *(học sâu tốt nhất)* | 18,8% | 0,726 | [−0,36 ; 1,51] | 0,675 | −42,0% | 24,6 |
| A1 · Mua và nắm giữ FPT *(mốc)* | 20,4% | 0,635 | [−0,22 ; 1,39] | 0,532 | −52,1% | 0,2 |
| **ATFN-ABCD** *(mô hình đề xuất)* | 9,3% | 0,526 | [−0,34 ; 1,31] | **0,719** | **−16,7%** | 65,5 |

### Ba kết luận trung thực

**1. Không chiến lược nào đạt chỉ tiêu Sharpe ≥ 1,8.** Cao nhất là 1,137. Sai số chuẩn của
Sharpe trên 1.404 phiên là 0,42 đơn vị, nên ngay cả con số cao nhất cũng chỉ cách 0 chưa tới ba
sai số chuẩn. Deflated Sharpe của nó là 0,925 — dưới ngưỡng 0,95 thường dùng để tuyên bố có kỹ năng.

**2. Mô hình học sâu không thắng được luật đơn giản.** Ba vị trí đầu bảng đều là chiến lược
dạng luật cổ điển. Mô hình đề xuất xếp dưới mốc mua-và-nắm-giữ về Sharpe, nhưng có **sụt giảm
tối đa thấp nhất bảng** (−16,7% so với −52,1%) và **Deflated Sharpe cao thứ hai** nhờ phân phối
lợi suất ít đuôi trái hơn.

**3. Định thời điểm phá giá trị chứ không tạo giá trị.** Phân rã cho thấy chiến lược tránh được
5,70 đơn vị lỗ nhưng bỏ lỡ 6,58 đơn vị lãi, ròng **−0,88**. Hồi quy có kiểm soát chính lợi suất
FPT cho alpha **−2,05%/năm** (t = −1,67; p = 0,094) — âm và không có ý nghĩa thống kê.

### Ablation của mô hình đề xuất

| Bậc | Sharpe (valid) | Δ | Lệnh/năm | Phí đã trả |
|---|---|---|---|---|
| A · TCN+GRU | 0,974 | — | 102,8 | 15,95% |
| A+B · +đa tầm dự báo & cổng nhịp | 0,636 | −0,338 | 42,3 | 5,72% |
| A+B+C · +cổng chế độ | 0,892 | −0,082 | 51,9 | 5,64% |
| A+B+C+L · +mất mát Sharpe có phí | 0,766 | −0,208 | 149,6 | 2,44% |
| A+B+C+D · +vùng không giao dịch | **0,997** | +0,023 | 69,3 | **2,73%** |

**Không thành phần nào cải thiện Sharpe một cách có ý nghĩa** — độ lệch giữa các hạt giống là
0,22–0,32, lớn hơn mọi khoảng cách trong bảng. Nhưng thành phần D làm đúng việc nó được thiết kế
để làm: phí từ 15,95% xuống 2,73%, độ lớn lệnh từ 21,2% xuống 5,4%, độ biến động từ 13,6% xuống
8,1%, sụt giảm tối đa từ −11,9% xuống −9,0%.

### Ablation về nhịp giao dịch

Cùng một vị thế mục tiêu, chỉ đổi cơ chế quyết định khi nào được đổi vị thế.

| Cơ chế chọn nhịp | Sharpe | Lệnh/năm | Phiên giữa hai lệnh | Phí đã trả |
|---|---|---|---|---|
| Ép mỗi phiên | 0,448 | 147,7 | 1,7 | 6,48% |
| Ép mỗi tuần | 0,466 | 51,7 | 4,9 | 3,58% |
| Ép mỗi tháng | **0,561** | 25,7 | 9,8 | 2,05% |
| Vùng cố định τ = 0,05 | 0,476 | 46,3 | 5,4 | 4,67% |
| Vùng cố định τ = 0,15 | 0,448 | 21,9 | 11,5 | 1,87% |
| Vùng cố định τ = 0,30 | 0,468 | 18,5 | 13,6 | 0,75% |
| **Vùng học được (ATFN-ABCD)** | **0,526** | 65,5 | 3,8 | 3,61% |

Vùng học được **thắng mọi vùng cố định** và thắng cả ép ngày lẫn ép tuần, nhưng **thua ép tháng**.
Ngưỡng τ mà mô hình học được nằm trong khoảng [0,219 ; 0,350], trung bình 0,293.

### Độ vững

| Đổi giả định | Sharpe |
|---|---|
| Chuẩn | 0,526 |
| Phí ×2 | 0,457 |
| Phí ×4 | 0,318 |
| Trễ thực thi 2 phiên | 0,539 |
| Trễ thực thi 3 phiên | 0,592 |
| **Không cộng lãi tiền mặt** | **0,203** |

Hai dòng cần đọc kỹ. **Lãi tiền mặt đóng góp hơn một nửa Sharpe** — mô hình đứng ngoài thị trường
68% thời gian nên phần lãi phi rủi ro rất lớn; đây là hiệu ứng thật nhưng phải nói rõ. **Làm chậm
tín hiệu lại làm Sharpe tăng** — không phải dấu hiệu rò rỉ (rò rỉ sẽ làm Sharpe sụt mạnh), mà cho
thấy tín hiệu không có lợi thế ngắn hạn nào: nó là tín hiệu chậm.

### Hạn chế đã ghi nhận

* Hàm mất mát Sharpe có **nghiệm suy biến tại w ≡ 0**; 2 trên 25 hạt giống rơi vào đó.
* Chỉ một tài sản nên không có đa dạng hoá, và chỉ mua/đứng ngoài nên không kiếm được gì khi thị trường giảm.
* PBO trên vùng chọn mô hình là 52–57%, nên mọi kết luận so sánh giữa các chiến lược đều phải kèm khoảng tin cậy.

---

## Trạng thái

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| 1 | Dữ liệu, đặc trưng, gán nhãn, EDA | Hoàn thành |
| 2 | Bộ máy backtest và chia dữ liệu chống rò rỉ | Hoàn thành |
| 3 | Baseline cổ điển và dạng luật | Hoàn thành |
| 4 | Baseline máy học và học sâu | Hoàn thành |
| 5 | Mô hình đề xuất ATFN và ablation | Hoàn thành |
| 6 | Chạy tập kiểm tra và kiểm định thống kê | Hoàn thành |
| 7 | Ứng dụng Streamlit 11 trang | Hoàn thành |
| — | Báo cáo Word và slide | Chưa làm |
