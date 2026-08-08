import streamlit as st
import joblib
import re
from collections import Counter

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Student Comment Analyzer",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    return model


model = load_model()


# =========================================================
# TEXT CLEANING
# IMPORTANT:
# Same preprocessing used during training
# =========================================================

def clean_text(text):

    text = str(text).lower()

    # -----------------------------
    # Contractions
    # -----------------------------

    text = text.replace("don't", "do not")
    text = text.replace("doesn't", "does not")
    text = text.replace("didn't", "did not")
    text = text.replace("can't", "cannot")
    text = text.replace("couldn't", "could not")
    text = text.replace("isn't", "is not")
    text = text.replace("wasn't", "was not")
    text = text.replace("aren't", "are not")
    text = text.replace("weren't", "were not")

    # -----------------------------
    # Remove URLs
    # -----------------------------

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # -----------------------------
    # Negation phrases
    # -----------------------------

    patterns = [
        (r"\bnot\s+good\b", "not_good"),
        (r"\bnot\s+helpful\b", "not_helpful"),
        (r"\bnot\s+supportive\b", "not_supportive"),
        (r"\bnot\s+satisfied\b", "not_satisfied"),
        (r"\bnot\s+useful\b", "not_useful"),
        (r"\bnot\s+clear\b", "not_clear"),
        (r"\bnot\s+effective\b", "not_effective"),
        (r"\bnot\s+interesting\b", "not_interesting"),
        (r"\bnot\s+happy\b", "not_happy"),

        (r"\bdoes\s+not\s+explain\b", "does_not_explain"),
        (r"\bdoes\s+not\s+teach\b", "does_not_teach"),
        (r"\bdoes\s+not\s+help\b", "does_not_help"),
        (r"\bdoes\s+not\s+answer\b", "does_not_answer"),
        (r"\bdoes\s+not\s+support\b", "does_not_support"),

        (r"\bcannot\s+understand\b", "cannot_understand"),
        (r"\bcan\s+not\s+understand\b", "cannot_understand"),
    ]

    for pattern, replacement in patterns:

        text = re.sub(
            pattern,
            replacement,
            text
        )

    # -----------------------------
    # Remove punctuation
    # -----------------------------

    text = re.sub(
        r"[^a-zA-Z0-9_\s]",
        " ",
        text
    )

    # -----------------------------
    # Remove extra spaces
    # -----------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# COMMON WORDS
# =========================================================

def get_common_words(comments, top_n=5):

    if not comments:
        return []

    text = " ".join(comments).lower()

    words = re.findall(
        r"\b[a-zA-Z]{4,}\b",
        text
    )

    stop_words = {
        "this",
        "that",
        "with",
        "very",
        "they",
        "have",
        "from",
        "teacher",
        "teachers",
        "classes",
        "class",
        "students",
        "student",
        "the",
        "and",
        "are",
        "was",
        "were",
        "been",
        "being",
        "good",
        "really",
        "much",
        "more",
        "some",
        "about",
        "their",
        "there",
        "also",
        "teaching",
        "teach",
        "teacher"
    }

    filtered_words = [
        word
        for word in words
        if word not in stop_words
    ]

    return Counter(
        filtered_words
    ).most_common(top_n)


# =========================================================
# SUMMARY FUNCTION
# =========================================================

def generate_summary(
    total,
    positive_count,
    negative_count,
    neutral_count,
    positive_topics,
    negative_topics
):

    summary = []

    # -----------------------------------------
    # Overall sentiment
    # -----------------------------------------

    if (
        positive_count > negative_count
        and positive_count > neutral_count
    ):

        summary.append(
            "Overall, students have given mostly positive feedback about the teaching and learning experience."
        )

    elif (
        negative_count > positive_count
        and negative_count > neutral_count
    ):

        summary.append(
            "Overall, students have expressed more negative feedback, indicating that some areas of the learning experience need improvement."
        )

    elif (
        neutral_count > positive_count
        and neutral_count > negative_count
    ):

        summary.append(
            "Overall, the feedback is mostly neutral, with many comments describing the learning experience without strong positive or negative opinions."
        )

    else:

        summary.append(
            "Overall, the feedback is mixed, with students expressing positive, negative and neutral opinions."
        )

    # -----------------------------------------
    # Positive feedback
    # -----------------------------------------

    if positive_topics:

        topics = ", ".join(
            word
            for word, count in positive_topics[:3]
        )

        summary.append(
            f"Students appreciated aspects related to {topics}."
        )

    # -----------------------------------------
    # Negative feedback
    # -----------------------------------------

    if negative_topics:

        topics = ", ".join(
            word
            for word, count in negative_topics[:3]
        )

        summary.append(
            f"Some students raised concerns related to {topics}."
        )

    # -----------------------------------------
    # Improvement
    # -----------------------------------------

    if negative_count > 0:

        summary.append(
            "The negative feedback suggests that these areas should be improved to provide a better learning experience."
        )

    # -----------------------------------------
    # Total
    # -----------------------------------------

    summary.append(
        f"A total of {total} student comments were analyzed."
    )

    return summary


# =========================================================
# HEADER
# =========================================================

st.title("🎓 Student Comment Analyzer")

st.write(
    "AI-Based Student Feedback Sentiment Analysis using TF-IDF + Linear SVM"
)

st.divider()


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("🤖 Model Information"):

    st.write(
        "The application uses a TF-IDF + Calibrated Linear SVM pipeline."
    )

    st.write(
        "Supported sentiments:"
    )

    st.write(
        "😊 Positive  |  😞 Negative  |  😐 Neutral"
    )


# =========================================================
# SECTION 1
# SINGLE COMMENT
# =========================================================

st.header("🔍 Analyze Single Comment")

comment = st.text_area(
    "Enter Student Feedback",
    placeholder=(
        "Example: Teacher explains concepts very well."
    ),
    height=120
)


if st.button(
    "🔍 Analyze Comment",
    use_container_width=True
):

    if not comment.strip():

        st.warning(
            "⚠️ Please enter a comment."
        )

    else:

        try:

            # -----------------------------------------
            # Clean text
            # -----------------------------------------

            cleaned_comment = clean_text(
                comment
            )

            # -----------------------------------------
            # Prediction
            # IMPORTANT:
            # Pipeline handles TF-IDF internally
            # -----------------------------------------

            prediction = model.predict(
                [cleaned_comment]
            )[0]

            prediction_text = str(
                prediction
            ).lower()

            # -----------------------------------------
            # Confidence
            # -----------------------------------------

            confidence = None

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    [cleaned_comment]
                )[0]

                confidence = (
                    max(probabilities) * 100
                )

            # -----------------------------------------
            # Result
            # -----------------------------------------

            st.subheader(
                "📊 Prediction Result"
            )

            if prediction_text == "positive":

                st.success(
                    "😊 Sentiment: Positive"
                )

            elif prediction_text == "negative":

                st.error(
                    "😞 Sentiment: Negative"
                )

            else:

                st.info(
                    "😐 Sentiment: Neutral"
                )

            if confidence is not None:

                st.metric(
                    "🎯 Confidence",
                    f"{confidence:.2f}%"
                )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


# =========================================================
# SECTION 2
# MULTIPLE COMMENTS
# =========================================================

st.divider()

st.header(
    "📚 Analyze Multiple Student Comments"
)

st.write(
    "Enter one student comment per line."
)


bulk_comments = st.text_area(
    "Student Comments",
    placeholder="""Teacher explains concepts very clearly.
The teacher is very supportive.
The classes are boring.
Teacher does not explain topics properly.
We need more practical examples.
The teacher clears our doubts.
The lectures are informative.""",
    height=250
)


# =========================================================
# ANALYZE ALL
# =========================================================

if st.button(
    "📊 Analyze All Comments",
    use_container_width=True
):

    if not bulk_comments.strip():

        st.warning(
            "⚠️ Please enter student comments."
        )

    else:

        try:

            # -----------------------------------------
            # Convert into list
            # -----------------------------------------

            comments = [
                c.strip()
                for c in bulk_comments.split("\n")
                if c.strip()
            ]

            # -----------------------------------------
            # Clean comments
            # -----------------------------------------

            cleaned_comments = [
                clean_text(c)
                for c in comments
            ]

            # -----------------------------------------
            # Prediction
            # Pipeline automatically applies TF-IDF
            # -----------------------------------------

            predictions = model.predict(
                cleaned_comments
            )

            # -----------------------------------------
            # Probabilities
            # -----------------------------------------

            probabilities = None

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    cleaned_comments
                )

            # -----------------------------------------
            # Count sentiments
            # -----------------------------------------

            positive_count = sum(
                str(p).lower() == "positive"
                for p in predictions
            )

            negative_count = sum(
                str(p).lower() == "negative"
                for p in predictions
            )

            neutral_count = sum(
                str(p).lower() == "neutral"
                for p in predictions
            )

            total_comments = len(
                comments
            )

            # =================================================
            # SENTIMENT OVERVIEW
            # =================================================

            st.divider()

            st.subheader(
                "📊 Sentiment Overview"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total",
                total_comments
            )

            col2.metric(
                "😊 Positive",
                positive_count
            )

            col3.metric(
                "😞 Negative",
                negative_count
            )

            col4.metric(
                "😐 Neutral",
                neutral_count
            )

            # =================================================
            # PERCENTAGES
            # =================================================

            if total_comments > 0:

                positive_percentage = (
                    positive_count
                    / total_comments
                    * 100
                )

                negative_percentage = (
                    negative_count
                    / total_comments
                    * 100
                )

                neutral_percentage = (
                    neutral_count
                    / total_comments
                    * 100
                )

                st.subheader(
                    "📈 Sentiment Distribution"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "😊 Positive %",
                    f"{positive_percentage:.1f}%"
                )

                col2.metric(
                    "😞 Negative %",
                    f"{negative_percentage:.1f}%"
                )

                col3.metric(
                    "😐 Neutral %",
                    f"{neutral_percentage:.1f}%"
                )

            # =================================================
            # COMMENTS BY SENTIMENT
            # =================================================

            positive_comments = [
                c
                for c, p in zip(
                    comments,
                    predictions
                )
                if str(p).lower() == "positive"
            ]

            negative_comments = [
                c
                for c, p in zip(
                    comments,
                    predictions
                )
                if str(p).lower() == "negative"
            ]

            neutral_comments = [
                c
                for c, p in zip(
                    comments,
                    predictions
                )
                if str(p).lower() == "neutral"
            ]

            # =================================================
            # COMMON WORDS
            # =================================================

            positive_topics = get_common_words(
                positive_comments
            )

            negative_topics = get_common_words(
                negative_comments
            )

            # =================================================
            # SUMMARY
            # =================================================

            summary = generate_summary(
                total_comments,
                positive_count,
                negative_count,
                neutral_count,
                positive_topics,
                negative_topics
            )

            st.divider()

            st.subheader(
                "📝 Overall Feedback Summary"
            )

            for sentence in summary:

                st.write(
                    "• " + sentence
                )

            # =================================================
            # COMMON TOPICS
            # =================================================

            st.divider()

            st.subheader(
                "🔑 Common Feedback Topics"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    "😊 Positive Feedback"
                )

                if positive_topics:

                    for word, count in positive_topics:

                        st.write(
                            f"• {word} ({count})"
                        )

                else:

                    st.write(
                        "No positive topics found."
                    )

            with col2:

                st.write(
                    "😞 Negative Feedback"
                )

                if negative_topics:

                    for word, count in negative_topics:

                        st.write(
                            f"• {word} ({count})"
                        )

                else:

                    st.write(
                        "No negative topics found."
                    )

            # =================================================
            # INDIVIDUAL RESULTS
            # =================================================

            st.divider()

            st.subheader(
                "📋 Individual Predictions"
            )

            for index, (comment, prediction) in enumerate(
                zip(comments, predictions)
            ):

                prediction_text = str(
                    prediction
                )

                # -----------------------------------------
                # Confidence
                # -----------------------------------------

                confidence = None

                if probabilities is not None:

                    confidence = (
                        max(probabilities[index])
                        * 100
                    )

                # -----------------------------------------
                # Display
                # -----------------------------------------

                if prediction_text.lower() == "positive":

                    message = (
                        f"😊 **Positive** — {comment}"
                    )

                    if confidence is not None:

                        message += (
                            f"  \n🎯 Confidence: "
                            f"{confidence:.2f}%"
                        )

                    st.success(message)

                elif prediction_text.lower() == "negative":

                    message = (
                        f"😞 **Negative** — {comment}"
                    )

                    if confidence is not None:

                        message += (
                            f"  \n🎯 Confidence: "
                            f"{confidence:.2f}%"
                        )

                    st.error(message)

                else:

                    message = (
                        f"😐 **Neutral** — {comment}"
                    )

                    if confidence is not None:

                        message += (
                            f"  \n🎯 Confidence: "
                            f"{confidence:.2f}%"
                        )

                    st.info(message)

        except Exception as e:

            st.error(
                "❌ Analysis failed."
            )

            st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Built using Machine Learning | "
    "TF-IDF + Calibrated Linear SVM"
)