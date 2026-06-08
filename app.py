import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import warnings

warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AI Interview Simulator",
    page_icon="🎤",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_data():
    return pd.read_csv("INTERVIEW.csv")

df = load_data()

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ==========================
# FUNCTIONS
# ==========================

def analyze_answer(user_answer, ideal_answer):

    if len(user_answer.strip()) < 3:
        return 0

    emb1 = model.encode([user_answer])
    emb2 = model.encode([ideal_answer])

    score = cosine_similarity(emb1, emb2)[0][0]

    return round(float(score) * 100, 2)


def get_confidence_score(answer):


    fillers = [
        "um",
        "uh",
        "hmm",
        "like",
        "basically",
        "actually",
        "literally",
        "you know",
        "kind of"
    ]

    words = answer.lower().split()

    count = sum(words.count(w) for w in fillers)

    confidence = max(0, 100 - count * 10)

    return confidence, count


def get_wpm(word_count, time_seconds):

    if time_seconds < 1:
        return 0

    return round(word_count / (time_seconds / 60))


def speed_feedback(wpm):

    if wpm == 0:
        return "⚠️ Could not measure"

    elif wpm < 80:
        return "🐢 Too Slow"

    elif wpm <= 150:
        return "✅ Perfect Pace"

    elif wpm <= 180:
        return "🚀 Slightly Fast"

    else:
        return "⚡ Too Fast"


def overall_feedback(score):

    if score >= 85:
        return "🌟 Excellent! Interview Ready"

    elif score >= 70:
        return "👍 Good! Keep Practicing"

    elif score >= 50:
        return "📚 Average"

    else:
        return "💪 Needs Improvement"


# ==========================
# SESSION STATE
# ==========================

if "question_index" not in st.session_state:
    st.session_state.question_index = 0

if "question_row" not in st.session_state:
    st.session_state.question_row = None

if "answered" not in st.session_state:
    st.session_state.answered = False


# ==========================
# TITLE
# ==========================

st.title("🎤 AI Interview Simulator")

# ==========================
# CATEGORY
# ==========================

category = st.selectbox(
    "Select Category",
    ["All", "HR", "Technical"]
)

# ==========================
# FILTER DATA
# ==========================

if category == "HR":
    pool = df[df["category"] == "HR"]

elif category == "Technical":
    pool = df[df["category"] == "Technical"]

else:
    pool = df

# ==========================
# LOAD QUESTION
# ==========================

if st.session_state.question_row is None:

    st.session_state.question_row = (
        pool.sample(1).iloc[0]
    )


question = st.session_state.question_row["question"]
ideal_answer = st.session_state.question_row["ideal_answer"]

# ==========================
# QUESTION DISPLAY
# ==========================

st.subheader("❓ Question")

st.info(question)

answer = st.text_area(
    "Type Your Answer",
    height=180
)

# ==========================
# SUBMIT
# ==========================

if st.button("Submit Answer"):

    if answer.strip() == "":
        st.warning("Please enter answer")
        st.stop()

    word_count = len(answer.split())

    time_taken = 15

    answer_score = analyze_answer(
        answer,
        ideal_answer
    )

    confidence, fillers = get_confidence_score(
        answer
    )

    wpm = get_wpm(
        word_count,
        time_taken
    )

    final_score = round(
        (answer_score * 0.50)
        + (confidence * 0.30)
        + ((min(wpm, 150) / 150) * 100 * 0.20),
        2
    )

    st.session_state.answered = True

    # ==========================
    # RESULTS
    # ==========================

    st.success("Answer Evaluated")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Answer Quality",
        f"{answer_score}%"
    )

    col2.metric(
        "Confidence",
        f"{confidence}%"
    )

    col3.metric(
        "Final Score",
        f"{final_score}%"
    )

    st.write(
        f"🚀 WPM: {wpm} ({speed_feedback(wpm)})"
    )

    st.write(
        f"💬 Feedback: {overall_feedback(final_score)}"
    )

    st.subheader("💡 Ideal Answer")

    st.write(ideal_answer)

    # ==========================
    # SAVE REPORT
    # ==========================

    report = {
        "Question": question,
        "Category": st.session_state.question_row["category"],
        "User Answer": answer,
        "Words": word_count,
        "WPM": wpm,
        "Answer Score %": answer_score,
        "Confidence %": confidence,
        "Final Score %": final_score
    }

    os.makedirs("reports", exist_ok=True)

    report_path = "reports/interview_report.csv"

    new_df = pd.DataFrame([report])

    if os.path.exists(report_path):
        old = pd.read_csv(report_path)
        final = pd.concat(
            [old, new_df],
            ignore_index=True
        )
    else:
        final = new_df

    final.to_csv(
        report_path,
        index=False
    )

# ==========================
# NEXT QUESTION
# ==========================

if st.session_state.answered:

    if st.button("➡️ Next Question"):

        st.session_state.question_row = (
            pool.sample(1).iloc[0]
        )

        st.session_state.answered = False

        st.rerun()