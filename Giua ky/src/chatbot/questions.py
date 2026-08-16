"""Ngân hàng câu hỏi gợi ý cho chatbot."""

CATEGORIES = [
    ("company", "🏢 Thông tin doanh nghiệp"),
    ("price", "📈 Giá và hiệu quả đầu tư"),
    ("risk", "⚠️ Rủi ro"),
    ("capm", "📐 Mô hình CAPM"),
    ("apt", "🧩 Mô hình APT"),
    ("mc", "🎲 Mô phỏng Monte Carlo"),
    ("tech", "📊 Phân tích kỹ thuật"),
    ("manage", "💼 Quản lý đầu tư"),
    ("theory", "📚 Kiến thức tài chính"),
]

# id, nhóm, câu hỏi, từ khoá phục vụ tìm kiếm khi người dùng tự gõ
QUESTIONS = [
    # ----------------------------- Doanh nghiệp -----------------------------
    dict(id="co_profile", cat="company", q="{t} là công ty gì, kinh doanh lĩnh vực nào?",
         kw="cong ty gi linh vuc nganh nghe gioi thieu ho so profile lam gi"),
    dict(id="co_listing", cat="company", q="{t} niêm yết từ khi nào và trên sàn nào?",
         kw="niem yet san hose khi nao ngay bao lau lich su"),
    dict(id="co_marketcap", cat="company", q="Vốn hoá thị trường và số cổ phiếu lưu hành là bao nhiêu?",
         kw="von hoa market cap so co phieu luu hanh issue share quy mo"),
    dict(id="co_shareholders", cat="company", q="Ai là những cổ đông lớn nhất?",
         kw="co dong lon so huu ai nam giu shareholders"),
    dict(id="co_foreign", cat="company", q="Room ngoại còn lại bao nhiêu?",
         kw="room ngoai nuoc ngoai foreigner so huu ty le"),
    dict(id="co_valuation", cat="company", q="P/E, P/B và ROE hiện tại là bao nhiêu?",
         kw="pe pb roe roa dinh gia chi so tai chinh valuation"),
    dict(id="co_growth", cat="company", q="Hiệu quả kinh doanh thay đổi thế nào qua các năm?",
         kw="roe bien loi nhuan tang truong hieu qua kinh doanh qua cac nam lich su"),

    # ------------------------------- Giá cả ---------------------------------
    dict(id="px_now", cat="price", q="Giá hiện tại của {t} là bao nhiêu?",
         kw="gia hien tai bao nhieu dong cua moi nhat now"),
    dict(id="px_change", cat="price", q="{t} tăng giảm bao nhiêu trong 1 tháng, 3 tháng, 1 năm qua?",
         kw="tang giam bao nhieu thang nam qua bien dong hieu suat gan day"),
    dict(id="px_cagr", cat="price", q="Lợi suất trung bình mỗi năm của {t} là bao nhiêu?",
         kw="loi suat trung binh nam cagr sinh loi bao nhieu mot nam"),
    dict(id="px_vs_index", cat="price", q="{t} so với VNINDEX thì hiệu quả hơn hay kém hơn?",
         kw="so sanh vnindex thi truong hieu qua hon kem benchmark"),
    dict(id="px_best_worst", cat="price", q="Năm nào {t} tăng mạnh nhất và năm nào giảm sâu nhất?",
         kw="nam nao tang manh nhat giam sau nhat tot te best worst"),
    dict(id="px_invest_100", cat="price", q="Nếu đầu tư 100 triệu từ đầu, bây giờ được bao nhiêu?",
         kw="dau tu 100 trieu tu dau bay gio bao nhieu von tang truong"),
    dict(id="px_monthly", cat="price", q="Lợi suất theo từng tháng trong năm ra sao?",
         kw="thang nao tot mua vu seasonality loi suat theo thang"),

    # -------------------------------- Rủi ro --------------------------------
    dict(id="rk_vol", cat="risk", q="Mức độ biến động của {t} là bao nhiêu?",
         kw="bien dong volatility do lech chuan rui ro bao nhieu"),
    dict(id="rk_var", cat="risk", q="VaR và CVaR 95% một ngày của {t} là bao nhiêu?",
         kw="var cvar gia tri chiu rui ro expected shortfall lo toi da ngay"),
    dict(id="rk_dd", cat="risk", q="Mức sụt giảm sâu nhất trong lịch sử là bao nhiêu?",
         kw="sut giam drawdown lo sau nhat dinh day phuc hoi"),
    dict(id="rk_sharpe", cat="risk", q="Tỷ số Sharpe của {t} là bao nhiêu, có tốt không?",
         kw="sharpe sortino calmar hieu qua dieu chinh rui ro"),
    dict(id="rk_split", cat="risk", q="Rủi ro hệ thống và rủi ro riêng chiếm bao nhiêu?",
         kw="rui ro he thong phi he thong rieng dac thu phan tach da dang hoa"),
    dict(id="rk_crisis", cat="risk", q="{t} diễn biến thế nào trong khủng hoảng 2008 và Covid 2020?",
         kw="khung hoang 2008 covid 2020 suy thoai giai doan xau stress"),

    # --------------------------------- CAPM ---------------------------------
    dict(id="capm_beta", cat="capm", q="Beta của {t} là bao nhiêu và có ý nghĩa gì?",
         kw="beta bao nhieu he so nhay thi truong y nghia"),
    dict(id="capm_alpha", cat="capm", q="Alpha Jensen của {t} là bao nhiêu?",
         kw="alpha jensen vuot troi sinh loi bat thuong"),
    dict(id="capm_expected", cat="capm", q="Theo CAPM, {t} nên mang lại lợi suất bao nhiêu?",
         kw="loi suat ky vong yeu cau capm doi hoi bao nhieu"),
    dict(id="capm_fair", cat="capm", q="{t} đang bị định giá cao hay thấp theo CAPM?",
         kw="dinh gia cao thap re dat sml duoi tren duong"),
    dict(id="capm_r2", cat="capm", q="Mô hình CAPM giải thích được bao nhiêu phần biến động?",
         kw="r binh phuong r2 giai thich muc do phu hop mo hinh"),
    dict(id="capm_rolling", cat="capm", q="Beta của {t} thay đổi thế nào theo thời gian?",
         kw="beta truot thay doi theo thoi gian rolling on dinh"),

    # ---------------------------------- APT ---------------------------------
    dict(id="apt_factors", cat="apt", q="Những nhân tố vĩ mô nào ảnh hưởng tới {t}?",
         kw="nhan to vi mo anh huong apt ty gia dau vang sp500 tac dong"),
    dict(id="apt_vs_capm", cat="apt", q="Mô hình APT có tốt hơn CAPM không?",
         kw="apt so voi capm tot hon so sanh mo hinh nao"),
    dict(id="apt_expected", cat="apt", q="Lợi suất kỳ vọng theo APT là bao nhiêu?",
         kw="loi suat ky vong apt bao nhieu phan bu rui ro"),
    dict(id="apt_usd", cat="apt", q="{t} nhạy cảm thế nào với tỷ giá USD/VND?",
         kw="ty gia usd vnd nhay cam ty gia do la anh huong"),

    # ------------------------------ Monte Carlo -----------------------------
    dict(id="mc_1y", cat="mc", q="Mô phỏng cho biết giá {t} sau 1 năm khoảng bao nhiêu?",
         kw="mo phong monte carlo gia sau 1 nam du bao tuong lai bao nhieu"),
    dict(id="mc_prob_loss", cat="mc", q="Xác suất thua lỗ sau 1 năm nắm giữ là bao nhiêu?",
         kw="xac suat lo thua lo sau 1 nam nam giu rui ro"),
    dict(id="mc_var", cat="mc", q="Nếu xấu nhất thì có thể mất bao nhiêu phần trăm vốn?",
         kw="xau nhat mat bao nhieu von var toan ky kich ban xau"),
    dict(id="mc_methods", cat="mc", q="Ba phương pháp mô phỏng cho kết quả khác nhau ra sao?",
         kw="so sanh phuong phap gbm student t bootstrap khac nhau"),
    dict(id="mc_double", cat="mc", q="Xác suất giá tăng gấp đôi trong 3 năm là bao nhiêu?",
         kw="gap doi x2 tang gap 3 nam xac suat dat muc gia"),

    # -------------------------------- Kỹ thuật ------------------------------
    dict(id="tc_trend", cat="tech", q="Xu hướng kỹ thuật hiện tại của {t} thế nào?",
         kw="xu huong ky thuat hien tai tang giam ma trend"),
    dict(id="tc_rsi", cat="tech", q="RSI hiện tại cho tín hiệu gì?",
         kw="rsi qua mua qua ban tin hieu chi bao"),
    dict(id="tc_macd", cat="tech", q="MACD đang báo hiệu điều gì?",
         kw="macd duong tin hieu histogram dong luong"),
    dict(id="tc_levels", cat="tech", q="Vùng hỗ trợ và kháng cự gần nhất ở đâu?",
         kw="ho tro khang cu vung gia dinh day support resistance"),

    # ----------------------------- Quản lý đầu tư ---------------------------
    dict(id="mg_weight", cat="manage", q="Nên phân bổ bao nhiêu phần trăm vốn vào {t}?",
         kw="phan bo bao nhieu phan tram von ty trong toi uu nen mua"),
    dict(id="mg_shares", cat="manage", q="Với 100 triệu đồng thì nên mua bao nhiêu cổ phiếu?",
         kw="100 trieu mua bao nhieu co phieu dinh co vi the so luong"),
    dict(id="mg_kelly", cat="manage", q="Tiêu chí Kelly khuyến nghị tỷ trọng bao nhiêu?",
         kw="kelly ty trong toi uu tang truong log"),
    dict(id="mg_dca", cat="manage", q="Nên mua một lần hay chia nhỏ hàng tháng (DCA)?",
         kw="dca binh quan gia mua mot lan lump sum chia nho hang thang"),
    dict(id="mg_horizon", cat="manage", q="Nắm giữ bao lâu thì xác suất có lãi cao?",
         kw="nam giu bao lau thoi gian dai han xac suat co lai"),
    dict(id="mg_mix", cat="manage", q="Danh mục pha trộn cổ phiếu và tiền gửi hiệu quả ra sao?",
         kw="pha tron tien gui tiet kiem danh muc hon hop backtest phan bo"),

    # ------------------------------- Lý thuyết ------------------------------
    dict(id="th_capm", cat="theory", q="Mô hình CAPM là gì?",
         kw="capm la gi khai niem cong thuc dinh nghia"),
    dict(id="th_beta", cat="theory", q="Beta là gì và đọc hiểu thế nào?",
         kw="beta la gi khai niem y nghia doc hieu"),
    dict(id="th_apt", cat="theory", q="Lý thuyết APT là gì, khác CAPM chỗ nào?",
         kw="apt la gi ly thuyet kinh doanh chenh lech khac capm"),
    dict(id="th_sharpe", cat="theory", q="Tỷ số Sharpe là gì, bao nhiêu là tốt?",
         kw="sharpe la gi bao nhieu la tot cong thuc y nghia"),
    dict(id="th_var", cat="theory", q="VaR và CVaR là gì?",
         kw="var cvar la gi gia tri chiu rui ro khai niem"),
    dict(id="th_mc", cat="theory", q="Mô phỏng Monte Carlo hoạt động thế nào?",
         kw="monte carlo la gi hoat dong the nao mo phong nguyen ly"),
    dict(id="th_dd", cat="theory", q="Maximum Drawdown là gì?",
         kw="drawdown la gi sut giam toi da khai niem"),
]

QUESTION_BY_ID = {q["id"]: q for q in QUESTIONS}


def questions_of(cat: str) -> list[dict]:
    return [q for q in QUESTIONS if q["cat"] == cat]


def render(q: dict, ticker: str) -> str:
    return q["q"].format(t=ticker)
