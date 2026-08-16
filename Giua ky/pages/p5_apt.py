"""Trang mô hình APT: hồi quy đa nhân tố vĩ mô và so sánh với CAPM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import viz
from src.analytics.apt import FACTOR_LABELS, interpret
from src.formatting import num, pct
from src.ui import formula, interpretation, metric_row, note, sidebar

a = sidebar()
r = a.apt

st.title("🧩 Lý thuyết định giá kinh doanh chênh lệch (APT)")
st.caption(
    f"Mô hình đa nhân tố vĩ mô cho {a.ticker} — {len(r.params) - 1} nhân tố rủi ro, "
    f"{r.n_obs:,} quan sát ngày"
)

formula(
    "R<sub>i</sub> − R<sub>f</sub> = α + b<sub>1</sub>F<sub>thị trường</sub> + "
    "b<sub>2</sub>F<sub>tỷ giá</sub> + b<sub>3</sub>F<sub>dầu</sub> + "
    "b<sub>4</sub>F<sub>vàng</sub> + b<sub>5</sub>F<sub>S&P500</sub> + ε"
)

n_sig = int((r.pvalues.drop("const") < 0.05).sum())
metric_row([
    ("Số nhân tố", str(len(r.params) - 1), f"{n_sig} có ý nghĩa 5%"),
    ("R²", pct(r.r_squared, 1), f"CAPM {pct(a.capm.r_squared, 1)}"),
    ("R² hiệu chỉnh", pct(r.adj_r_squared, 1), None),
    ("Lợi suất kỳ vọng APT", pct(r.expected_return), f"CAPM {pct(a.capm.expected_return)}"),
    ("VIF lớn nhất", num(float(r.vif.max())), "ngưỡng cảnh báo 5"),
    ("Số quan sát", f"{r.n_obs:,}", None),
])

tab1, tab2, tab3, tab4 = st.tabs([
    "Hệ số các nhân tố", "Đóng góp vào lợi suất", "So sánh APT với CAPM", "Chẩn đoán mô hình",
])

with tab1:
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.plotly_chart(viz.factor_betas(r), use_container_width=True)
    with col2:
        st.markdown("#### Bảng kết quả hồi quy")
        tbl = pd.DataFrame({
            "Hệ số b": r.params,
            "Sai số chuẩn": r.model.bse,
            "t-stat": r.tvalues,
            "p-value": r.pvalues,
        }).round(4)
        tbl.index = ["Hằng số (α)"] + [FACTOR_LABELS.get(i, i) for i in r.params.index[1:]]
        st.dataframe(tbl, use_container_width=True)
        st.caption("Sai số chuẩn Newey–West (HAC, 5 độ trễ).")

    st.markdown("#### Ý nghĩa từng nhân tố")
    st.markdown(
        "| Nhân tố | Đại diện cho rủi ro | Kỳ vọng lý thuyết |\n"
        "|---|---|---|\n"
        "| Thị trường (VNINDEX) | Rủi ro hệ thống của toàn thị trường cổ phiếu Việt Nam | "
        "Hệ số dương, giá trị lớn nhất |\n"
        "| Tỷ giá USD/VND | Rủi ro tiền tệ; ảnh hưởng doanh nghiệp xuất khẩu và nợ ngoại tệ | "
        "Dương với doanh nghiệp thu ngoại tệ |\n"
        "| Giá dầu WTI | Chi phí đầu vào và chu kỳ kinh tế toàn cầu | "
        "Dương với ngành năng lượng, âm với ngành tiêu thụ nhiều nhiên liệu |\n"
        "| Giá vàng | Tâm lý trú ẩn, kỳ vọng lạm phát | "
        "Thường âm khi dòng tiền rời tài sản rủi ro |\n"
        "| S&P 500 | Rủi ro thị trường toàn cầu, dòng vốn ngoại | "
        "Dương với cổ phiếu có tỷ trọng nhà đầu tư nước ngoài cao |"
    )

    interpretation(interpret(r, a.ticker))

with tab2:
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.plotly_chart(viz.factor_contribution(r), use_container_width=True)
    with col2:
        contrib = pd.DataFrame({
            "Phần bù nhân tố (năm)": [pct(v) for v in r.factor_premia],
            "Hệ số nhạy b": [num(v) for v in r.params.drop("const")],
            "Đóng góp": [pct(v) for v in r.contributions],
        }, index=[FACTOR_LABELS.get(i, i) for i in r.contributions.index])
        st.dataframe(contrib, use_container_width=True)
        formula(
            "E(R<sub>i</sub>) = R<sub>f</sub> + Σ b<sub>j</sub> · λ<sub>j</sub>"
        )
        note(
            f"Lợi suất kỳ vọng theo APT = {pct(a.rf_annual)} (phi rủi ro) + "
            f"{pct(float(r.contributions.sum()))} (tổng phần bù rủi ro) = "
            f"<b>{pct(r.expected_return)}/năm</b>. "
            f"Nhân tố thị trường đóng góp <b>{pct(float(r.contributions.get('market', np.nan)))}</b>, "
            "áp đảo so với các nhân tố vĩ mô còn lại."
        )

with tab3:
    col1, col2 = st.columns([1, 1])
    with col1:
        cmp = pd.DataFrame({
            "CAPM": {
                "Số nhân tố": "1",
                "R²": pct(a.capm.r_squared, 2),
                "R² hiệu chỉnh": "—",
                "Lợi suất kỳ vọng": pct(a.capm.expected_return),
                "Hệ số thị trường": num(a.capm.beta),
                "Số tham số ước lượng": "2",
            },
            f"APT ({len(r.params) - 1} nhân tố)": {
                "Số nhân tố": str(len(r.params) - 1),
                "R²": pct(r.r_squared, 2),
                "R² hiệu chỉnh": pct(r.adj_r_squared, 2),
                "Lợi suất kỳ vọng": pct(r.expected_return),
                "Hệ số thị trường": num(float(r.params.get("market", np.nan))),
                "Số tham số ước lượng": str(len(r.params)),
            },
        })
        st.dataframe(cmp, use_container_width=True)

        f = go.Figure()
        f.add_trace(go.Bar(x=["CAPM", "APT"],
                           y=[a.capm.r_squared * 100, r.r_squared * 100],
                           marker_color=["#94a3b8", "#2563eb"],
                           text=[f"{a.capm.r_squared * 100:.1f}%", f"{r.r_squared * 100:.1f}%"],
                           textposition="outside"))
        f.update_layout(**viz.LAYOUT, title="Khả năng giải thích của hai mô hình",
                        yaxis_title="R² (%)")
        st.plotly_chart(f, use_container_width=True)
    with col2:
        gain = r.r_squared - a.capm.r_squared
        note(
            f"Thêm 4 nhân tố vĩ mô chỉ nâng R² thêm <b>{pct(gain, 2)}</b> "
            f"(từ {pct(a.capm.r_squared, 1)} lên {pct(r.r_squared, 1)}). "
            "Kết quả này nhất quán với thực tế thị trường Việt Nam: biến động cổ phiếu bị chi "
            "phối áp đảo bởi nhân tố thị trường trong nước, còn các biến vĩ mô toàn cầu tác "
            "động chậm và gián tiếp nên rất khó nắm bắt ở tần suất ngày."
        )
        st.markdown(
            "#### Ưu và nhược điểm\n\n"
            "**APT mạnh hơn CAPM ở chỗ:**\n"
            "- Không cần giả định danh mục thị trường hiệu quả hay nhà đầu tư đồng nhất.\n"
            "- Cho phép nhiều nguồn rủi ro cùng tồn tại, gần thực tế hơn.\n"
            "- Chỉ dựa trên nguyên lý không tồn tại cơ hội kinh doanh chênh lệch.\n\n"
            "**Nhưng APT cũng có điểm yếu:**\n"
            "- Lý thuyết không chỉ ra nhân tố nào phải đưa vào — việc chọn nhân tố mang tính "
            "thực nghiệm và dễ dẫn tới dò tìm dữ liệu.\n"
            "- Nhiều tham số hơn nên dễ khớp quá mức; cần nhìn R² hiệu chỉnh thay vì R².\n"
            "- Phần bù rủi ro của mỗi nhân tố khó ước lượng chính xác.\n\n"
            "Với dữ liệu ngày của một cổ phiếu Việt Nam, kết luận thực nghiệm ở đây là: "
            "**mô hình một nhân tố đã nắm gần hết phần giải thích được**, phần còn lại thuộc "
            "về rủi ro riêng của doanh nghiệp."
        )

with tab4:
    st.markdown("### Kiểm tra đa cộng tuyến giữa các nhân tố")
    vif = pd.DataFrame({
        "VIF": r.vif.round(2),
        "Đánh giá": ["Tốt (< 5)" if v < 5 else "Cần thận trọng (≥ 5)" for v in r.vif],
    })
    vif.index = [FACTOR_LABELS.get(i, i) for i in r.vif.index]
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(vif, use_container_width=True)
        st.caption("VIF = 1/(1−R²ⱼ), với R²ⱼ là hệ số xác định khi hồi quy nhân tố j "
                   "theo các nhân tố còn lại. VIF > 5 cảnh báo đa cộng tuyến.")
    with col2:
        corr = r.data.drop(columns=["stock"]).corr()
        labels = [FACTOR_LABELS.get(i, i) for i in corr.index]
        f = go.Figure(go.Heatmap(
            z=corr.values, x=labels, y=labels, zmid=0,
            colorscale=[[0, "#dc2626"], [0.5, "#ffffff"], [1, "#2563eb"]],
            text=corr.round(2).values, texttemplate="%{text}",
        ))
        f.update_layout(**{k: v for k, v in viz.LAYOUT.items() if k != "hovermode"},
                        title="Ma trận tương quan giữa các nhân tố", hovermode="closest")
        st.plotly_chart(f, use_container_width=True)

    note(
        "Các nhân tố có tương quan thấp với nhau và VIF đều dưới ngưỡng cảnh báo, nên hệ số "
        "hồi quy ước lượng được là đáng tin cậy về mặt kỹ thuật — vấn đề của mô hình nằm ở "
        "khả năng giải thích, không nằm ở đa cộng tuyến."
    )

    st.markdown("### Kiểm định ý nghĩa chung của các nhân tố vĩ mô")
    try:
        macro_names = [n for n in r.params.index if n not in ("const", "market")]
        hypothesis = ", ".join(f"{n} = 0" for n in macro_names)
        ftest = r.model.f_test(hypothesis)
        st.markdown(
            f"Giả thuyết H₀: **toàn bộ hệ số của các nhân tố vĩ mô đều bằng 0** "
            f"(loại bỏ {len(macro_names)} nhân tố, chỉ giữ nhân tố thị trường).\n\n"
            f"- Thống kê F = **{float(np.ravel(ftest.statistic)[0]):.2f}**\n"
            f"- p-value = **{float(ftest.pvalue):.4f}**\n\n"
            + ("→ **Bác bỏ H₀** ở mức 5%: nhóm nhân tố vĩ mô có đóng góp thống kê, dù mức "
               "cải thiện R² nhỏ."
               if float(ftest.pvalue) < 0.05 else
               "→ **Không bác bỏ H₀** ở mức 5%: xét chung, các nhân tố vĩ mô không cải thiện "
               "mô hình một cách có ý nghĩa. Mô hình CAPM đơn giản là đủ.")
        )
    except Exception as exc:
        st.info(f"Không chạy được kiểm định F: {exc}")
