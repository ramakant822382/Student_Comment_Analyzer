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
    layout="centered"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    data = joblib.load("sentiment_model.pkl")

    model = data["model"]
    tfidf = data["tfidf"]

    return model, tfidf


model, tfidf = load_model()


# =========================================================
# COMMON WORDS FUNCTION
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
        "also"
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

    if positive_count > negative_count:

        summary.append(
            "Overall, students have given mostly positive feedback about the teaching and learning experience."
        )

    elif negative_count > positive_count:

        summary.append(
            "Overall, students have expressed more negative feedback, indicating that some areas of the learning experience need improvement."
        )

    else:

        summary.append(
            "Overall, the feedback is mixed, with students expressing both positive and negative opinions."
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
    # Total comments
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
    "AI Based Student Feedback Sentiment Analysis"
)

st.divider()


# =========================================================
# SECTION 1 - SINGLE COMMENT
# =========================================================

st.header("🔍 Analyze Single Comment")

comment = st.text_area(
    "Enter Student Feedback",
    placeholder="Example: Teacher explains concepts very well.",
    height=120
)


if st.button(
    "Analyze Comment",
    use_container_width=True
):

    if not comment.strip():

        st.warning(
            "⚠️ Please enter a comment."
        )

    else:

        try:

            # TF-IDF
            comment_vector = tfidf.transform(
                [comment]
            )

            # Prediction
            prediction = model.predict(
                comment_vector
            )[0]

            prediction_text = str(
                prediction
            )


            # Confidence
            confidence = None

            if hasattr(
                model,
                "predict_proba"
            ):

                probability = model.predict_proba(
                    comment_vector
                )

                confidence = max(
                    probability[0]
                ) * 100


            # Result
            st.subheader(
                "📊 Result"
            )


            if prediction_text.lower() == "positive":

                st.success(
                    f"😊 Sentiment: {prediction_text}"
                )

            elif prediction_text.lower() == "negative":

                st.error(
                    f"😞 Sentiment: {prediction_text}"
                )

            else:

                st.info(
                    f"😐 Sentiment: {prediction_text}"
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
# SECTION 2 - MULTIPLE COMMENTS
# =========================================================

st.divider()

st.header("📚 Analyze Multiple Student Comments")

st.write(
    "Enter multiple comments. Write one student comment per line."
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
# ANALYZE ALL COMMENTS
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
            # Convert text into list
            # -----------------------------------------

            comments = [
                c.strip()
                for c in bulk_comments.split("\n")
                if c.strip()
            ]


            # -----------------------------------------
            # TF-IDF
            # -----------------------------------------

            comment_vectors = tfidf.transform(
                comments
            )


            # -----------------------------------------
            # Predictions
            # -----------------------------------------

            predictions = model.predict(
                comment_vectors
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
            # SENTIMENT COUNTS
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
            # FIND COMMON WORDS
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


            for comment, prediction in zip(
                comments,
                predictions
            ):

                prediction = str(
                    prediction
                )


                if prediction.lower() == "positive":

                    st.success(
                        f"😊 {prediction} — {comment}"
                    )

                elif prediction.lower() == "negative":

                    st.error(
                        f"😞 {prediction} — {comment}"
                    )

                else:

                    st.info(
                        f"😐 {prediction} — {comment}"
                    )


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
    "Built using Machine Learning | TF-IDF + Logistic Regression"
)