import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import sklearn

st.set_page_config(
    page_title="Retail Review Intelligence",
    page_icon="🧸",
    layout="wide"
)

st.title("🧸 Retail Review Intelligence")
st.caption(
    "Analyze customer sentiment, investigate negative feedback "
    "and compare product reviews."
)

# Load saved predictions and model
@st.cache_data
def load_data():
    return pd.read_csv("reviews_with_predictions.csv")

@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.joblib")

df = load_data()
model = load_model()

# Ensure review text is usable
df["review_text"] = df["review_text"].fillna("").astype(str)

# Sidebar filters
st.sidebar.header("Filter reviews")

sentiments = st.sidebar.multiselect(
    "Predicted sentiment",
    ["positive", "neutral", "negative"],
    default=["positive", "neutral", "negative"]
)

filtered = df[df["predicted_sentiment"].isin(sentiments)].copy()

if "rating" in filtered.columns:
    ratings = sorted(filtered["rating"].dropna().unique())
    selected_ratings = st.sidebar.multiselect(
        "Star rating",
        ratings,
        default=ratings
    )
    filtered = filtered[filtered["rating"].isin(selected_ratings)]

# Overview metrics
total = len(filtered)
negative_rate = (
    (filtered["predicted_sentiment"] == "negative").mean() * 100
    if total else 0
)
positive_rate = (
    (filtered["predicted_sentiment"] == "positive").mean() * 100
    if total else 0
)
disagreement_rate = (
    filtered["sentiment_disagreement"].mean() * 100
    if total and "sentiment_disagreement" in filtered.columns
    else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews in view", f"{total:,}")
c2.metric("Predicted positive", f"{positive_rate:.1f}%")
c3.metric("Predicted negative", f"{negative_rate:.1f}%")
c4.metric("Rating-label disagreements", f"{disagreement_rate:.1f}%")

# Sentiment distribution
st.subheader("Customer sentiment distribution")

if total:
    sentiment_counts = (
        filtered["predicted_sentiment"]
        .value_counts()
        .reindex(["positive", "neutral", "negative"], fill_value=0)
        .rename_axis("Sentiment")
        .reset_index(name="Reviews")
    )

    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Reviews",
        color="Sentiment",
        text="Reviews",
        title="Predicted sentiment across selected reviews"
    )
    st.plotly_chart(fig, use_container_width=True)

# Rating versus predicted sentiment
st.subheader("Sentiment by star rating")

if total and "rating" in filtered.columns:
    rating_counts = (
        filtered.groupby(["rating", "predicted_sentiment"])
        .size()
        .reset_index(name="Reviews")
    )
    fig = px.bar(
        rating_counts,
        x="rating",
        y="Reviews",
        color="predicted_sentiment",
        barmode="group",
        title="Predicted sentiment for each star rating"
    )
    st.plotly_chart(fig, use_container_width=True)

# Product-level insights
if "parent_asin" in filtered.columns:
    st.subheader("Products requiring investigation")

    product_summary = (
        filtered.groupby("parent_asin")
        .agg(
            review_count=("review_text", "size"),
            negative_reviews=(
                "predicted_sentiment",
                lambda s: (s == "negative").sum()
            ),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )

    product_summary["negative_rate_pct"] = (
        100 * product_summary["negative_reviews"]
        / product_summary["review_count"]
    )

    # Avoid ranking products with very few reviews
    product_summary = product_summary[
        product_summary["review_count"] >= 20
    ]

    product_summary = product_summary.sort_values(
        "negative_rate_pct", ascending=False
    )

    st.caption(
        "Products with at least 20 reviews, ranked by predicted "
        "negative-review percentage. Investigate these results "
        "before taking action."
    )
    st.dataframe(
        product_summary.head(20),
        use_container_width=True,
        hide_index=True
    )

# Review explorer
st.subheader("Explore individual reviews")

search = st.text_input("Search review text")

review_view = filtered.copy()
if search:
    review_view = review_view[
        review_view["review_text"].str.contains(
            search, case=False, na=False, regex=False
        )
    ]

display_columns = [
    c for c in [
        "parent_asin", "rating", "review_text",
        "predicted_sentiment", "prediction_confidence",
        "sentiment_disagreement"
    ]
    if c in review_view.columns
]

st.dataframe(
    review_view[display_columns].head(500),
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download filtered reviews",
    data=review_view.to_csv(index=False).encode("utf-8"),
    file_name="filtered_review_insights.csv",
    mime="text/csv"
)

# Live prediction
st.subheader("Analyze a new review")

new_review = st.text_area(
    "Enter a product review",
    placeholder="e.g. The toy is fun, but one of the pieces broke..."
)

if st.button("Analyze review"):
    if not new_review.strip():
        st.warning("Please enter a review first.")
    else:
        prediction = model.predict([new_review])[0]
        probabilities = model.predict_proba([new_review])[0]
        classes = model.named_steps["classifier"].classes_

        st.write("**Predicted sentiment:**", prediction)
        st.write(
            "**Highest class probability:**",
            f"{max(probabilities):.1%}"
        )

        probability_df = pd.DataFrame({
            "Sentiment": classes,
            "Probability": probabilities
        })

        fig = px.bar(
            probability_df,
            x="Sentiment",
            y="Probability",
            range_y=[0, 1],
            title="Predicted probability by sentiment"
        )
        st.plotly_chart(fig, use_container_width=True)
