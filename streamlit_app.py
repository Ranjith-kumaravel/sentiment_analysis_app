import streamlit as st
import pandas as pd
import re
from PyPDF2 import PdfReader
from textblob import TextBlob
import plotly.express as px

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("📊 Product Review Sentiment Analysis")

# -----------------------------
# Functions
# -----------------------------

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def split_reviews(text):
    reviews = re.split(r'\n\s*\n', text)
    reviews = [r.strip() for r in reviews if r.strip() and len(r.strip()) > 20]
    return reviews


def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        label = "Positive"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return label, round(polarity, 3)


# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    reviews = split_reviews(text)

    st.success(f"✅ Extracted {len(reviews)} reviews")

    # -----------------------------
    # Create DataFrame
    # -----------------------------
    data = []

    for i, review in enumerate(reviews):
        label, polarity = get_sentiment(review)

        data.append({
            "ID": i + 1,
            "Preview": review[:200],
            "Full Review": review,
            "Sentiment": label,
            "Polarity": polarity
        })

    df = pd.DataFrame(data)

    # -----------------------------
    # Metrics
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    total = len(df)
    pos = len(df[df["Sentiment"] == "Positive"])
    neg = len(df[df["Sentiment"] == "Negative"])
    neu = len(df[df["Sentiment"] == "Neutral"])

    col1.metric("Positive", f"{pos}", f"{pos/total*100:.1f}%")
    col2.metric("Negative", f"{neg}", f"{neg/total*100:.1f}%")
    col3.metric("Neutral", f"{neu}", f"{neu/total*100:.1f}%")

    # -----------------------------
    # Charts
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(df, names="Sentiment", title="Sentiment Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.bar(df["Sentiment"].value_counts().reset_index(),
                         x="Sentiment", y="count",
                         title="Sentiment Count")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Histogram
    fig_hist = px.histogram(df, x="Polarity", nbins=30, title="Polarity Distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

    # -----------------------------
    # Filter Table
    # -----------------------------
    selected = st.multiselect(
        "Filter by Sentiment",
        options=["Positive", "Negative", "Neutral"],
        default=["Positive", "Negative", "Neutral"]
    )

    filtered_df = df[df["Sentiment"].isin(selected)]

    st.dataframe(filtered_df)

    # -----------------------------
    # Review Inspector
    # -----------------------------
    review_id = st.selectbox("Select Review ID", df["ID"])

    selected_review = df[df["ID"] == review_id].iloc[0]

    st.info(f"""
    **Sentiment:** {selected_review['Sentiment']}  
    **Polarity:** {selected_review['Polarity']}  

    **Review:**  
    {selected_review['Full Review']}
    """)

    # -----------------------------
    # CSV Download
    # -----------------------------
    csv = df.to_csv(index=False)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="sentiment_results.csv",
        mime="text/csv"
    )