"""Trang chatbot hỏi đáp: 52 câu hỏi gợi ý được trả lời bằng số liệu tính trực tiếp."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.chatbot.engine import answer, match_questions, other_ticker_mentioned
from src.chatbot.questions import CATEGORIES, QUESTION_BY_ID, QUESTIONS, questions_of, render
from src.ui import note, sidebar

a = sidebar()

st.title("🤖 Chatbot hỏi đáp về cổ phiếu")
st.caption(
    f"Trả lời {len(QUESTIONS)} câu hỏi về {a.ticker} bằng số liệu tính trực tiếp từ dữ liệu "
    "và các mô hình đã ước lượng — hoạt động hoàn toàn ngoại tuyến, không gọi dịch vụ bên ngoài"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_qid" not in st.session_state:
    st.session_state.pending_qid = None


def ask(qid: str) -> None:
    st.session_state.pending_qid = qid


# Bảng câu hỏi gợi ý
with st.expander("📋 Danh mục câu hỏi gợi ý — bấm để hỏi", expanded=not st.session_state.chat_history):
    tabs = st.tabs([label for _, label in CATEGORIES])
    for tab, (cat, _) in zip(tabs, CATEGORIES):
        with tab:
            items = questions_of(cat)
            cols = st.columns(2)
            for i, q in enumerate(items):
                with cols[i % 2]:
                    st.button(render(q, a.ticker), key=f"btn_{q['id']}",
                              use_container_width=True, on_click=ask, args=(q["id"],))

# Lịch sử hội thoại
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["text"])
        if turn.get("table") is not None:
            st.dataframe(turn["table"], use_container_width=True)
        if turn.get("followups"):
            st.caption("Câu hỏi liên quan:")
            cols = st.columns(len(turn["followups"]))
            for col, fid in zip(cols, turn["followups"]):
                q = QUESTION_BY_ID.get(fid)
                if q:
                    with col:
                        st.button(render(q, a.ticker),
                                  key=f"fu_{turn['n']}_{fid}",
                                  use_container_width=True, on_click=ask, args=(fid,))

# Ô nhập tự do
typed = st.chat_input(f"Nhập câu hỏi về {a.ticker}, ví dụ: beta là bao nhiêu?")

qid = None
if st.session_state.pending_qid:
    qid = st.session_state.pending_qid
    st.session_state.pending_qid = None
    question_text = render(QUESTION_BY_ID[qid], a.ticker)
elif typed:
    other = other_ticker_mentioned(typed, a.ticker)
    matches = [] if other else match_questions(typed, a.ticker, 4)
    if matches:
        qid = matches[0][0]["id"]
        question_text = typed
        st.session_state["last_matches"] = [m[0]["id"] for m in matches[1:]]
    else:
        if other:
            reply = (
                f"Đề tài này chỉ phân tích cổ phiếu **{a.ticker}**, nên mình không có dữ liệu "
                f"về **{other}**.\n\nToàn bộ số liệu trong hệ thống được tải sẵn cho một mã duy "
                f"nhất. Nếu muốn phân tích mã khác, hãy đổi tham số `TICKER` trong "
                "`src/config.py` rồi chạy lại `fetch_data.py` — mọi trang phân tích và "
                "cả chatbot sẽ tự cập nhật theo mã mới."
            )
        else:
            reply = (
                "Mình chưa hiểu câu hỏi này. Chatbot được xây dựng theo hướng **hỏi đáp có "
                f"kiểm soát**: chỉ trả lời {len(QUESTIONS)} câu hỏi đã được lập trình sẵn, để "
                "bảo đảm mọi con số đưa ra đều chính xác và kiểm chứng được. Khi không chắc "
                "chắn, mình chọn nói không biết thay vì đoán.\n\nBạn hãy mở mục **Danh mục câu "
                "hỏi gợi ý** phía trên và chọn một câu, hoặc thử các từ khoá như: *beta, alpha, "
                "VaR, Sharpe, Monte Carlo, cổ đông, P/E, phân bổ vốn, DCA*."
            )
        st.session_state.chat_history.append({
            "n": len(st.session_state.chat_history),
            "question": typed,
            "text": reply,
            "table": None,
            "followups": ["capm_beta", "rk_var", "mc_1y"],
        })
        st.rerun()

if qid:
    res = answer(qid, a)
    st.session_state.chat_history.append({
        "n": len(st.session_state.chat_history),
        "question": question_text,
        "text": res.text,
        "table": res.table,
        "followups": res.followups[:3],
    })
    st.rerun()

if not st.session_state.chat_history:
    note(
        "<b>Cách hoạt động:</b> mỗi câu hỏi gợi ý được nối trực tiếp với một hàm tính toán "
        "trong mã nguồn. Khi bạn bấm một câu hỏi, chương trình chạy đúng hàm đó trên dữ liệu "
        "thật rồi diễn giải kết quả thành lời. Nhờ vậy chatbot không bao giờ bịa số — điểm "
        "yếu lớn nhất của các trợ lý dựa trên mô hình ngôn ngữ khi làm việc với dữ liệu tài chính."
    )
    c1, c2, c3 = st.columns(3)
    for col, qid_ in zip((c1, c2, c3), ("px_now", "capm_beta", "mc_1y")):
        with col:
            st.button("▶ " + render(QUESTION_BY_ID[qid_], a.ticker), key=f"start_{qid_}",
                      use_container_width=True, on_click=ask, args=(qid_,))
else:
    if st.button("🗑️ Xoá lịch sử hội thoại"):
        st.session_state.chat_history = []
        st.rerun()
