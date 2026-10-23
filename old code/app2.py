
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --------------------------------------------------
# 1. Page configuration and consistent colours
# --------------------------------------------------

st.set_page_config(
    page_title="Retail Review Intelligence",
    page_icon="🧸",
    layout="wide"
)

SENTIMENT_ORDER = ["positive", "neutral", "negative"]

SENTIMENT_COLORS = {
    "positive": "#16A34A",  # Green
    "neutral": "#2563EB",   # Blue
    "negative": "#DC2626",  # Red
}

st.title("🧸 Retail Review Intelligence")
st.caption(
    "Analyze customer sentiment, investigate product issues, "
    "and understand what customers say about specific aspects."
)

# --------------------------------------------------
# 2. Load prepared data and trained model
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("reviews_with_predictions.csv")


@st.cache_data
def load_aspects():
    return pd.read_csv("review_aspects.csv")


@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.joblib")


df = load_data()
aspects = load_aspects()
model = load_model()

# Clean up review text and sentiment labels
df["review_text"] = df["review_text"].fillna("").astype(str)
df["predicted_sentiment"] = (
    df["predicted_sentiment"].astype(str).str.lower()
)

if "aspect_sentiment" in aspects.columns:
    aspects["aspect_sentiment"] = (
        aspects["aspect_sentiment"].astype(str).str.lower()
    )

if "evidence_sentence" in aspects.columns:
    aspects["evidence_sentence"] = (
        aspects["evidence_sentence"].fillna("").astype(str)
    )

# --------------------------------------------------
# 3. Sidebar filters
# --------------------------------------------------

st.sidebar.header("Filter reviews")

available_sentiments = [
    s for s in SENTIMENT_ORDER
    if s in df["predicted_sentiment"].unique()
]

sentiments = st.sidebar.multiselect(
    "Predicted sentiment",
    available_sentiments,
    default=available_sentiments
)

filtered = df[
    df["predicted_sentiment"].isin(sentiments)
].copy()

if "rating" in filtered.columns:
    ratings = sorted(filtered["rating"].dropna().unique())

    selected_ratings = st.sidebar.multiselect(
        "Star rating",
        ratings,
        default=ratings
    )

    filtered = filtered[
        filtered["rating"].isin(selected_ratings)
    ]

# --------------------------------------------------
# 4. Customer feedback overview
# --------------------------------------------------

st.header("1. Customer Feedback Overview")

total = len(filtered)

negative_rate = (
    100 * (filtered["predicted_sentiment"] == "negative").mean()
    if total else 0
)

positive_rate = (
    100 * (filtered["predicted_sentiment"] == "positive").mean()
    if total else 0
)

disagreement_rate = (
    100 * filtered["sentiment_disagreement"].mean()
    if total and "sentiment_disagreement" in filtered.columns
    else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Reviews in view", f"{total:,}")
c2.metric("Predicted positive", f"{positive_rate:.1f}%")
c3.metric("Predicted negative", f"{negative_rate:.1f}%")
c4.metric("Rating-label disagreements", f"{disagreement_rate:.1f}%")

# --------------------------------------------------
# 5. Overall sentiment distribution
# --------------------------------------------------

st.subheader("Customer sentiment distribution")

if total:
    sentiment_counts = (
        filtered["predicted_sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .rename_axis("Sentiment")
        .reset_index(name="Reviews")
    )

    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Reviews",
        color="Sentiment",
        text="Reviews",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"Sentiment": SENTIMENT_ORDER},
        title="Predicted sentiment across selected reviews"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No reviews match the current sidebar filters.")

# --------------------------------------------------
# 6. Rating versus predicted sentiment
# --------------------------------------------------

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
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={
            "predicted_sentiment": SENTIMENT_ORDER
        },
        title="Predicted sentiment for each star rating"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 7. Product-level insights
# --------------------------------------------------

st.header("2. Product-Level Insights")

if "parent_asin" in filtered.columns:
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

    min_reviews = st.slider(
        "Minimum reviews required to rank a product",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )

    ranked_products = product_summary[
        product_summary["review_count"] >= min_reviews
    ].sort_values(
        ["negative_rate_pct", "review_count"],
        ascending=[False, False]
    )

    st.caption(
        "Products are ranked by the percentage of reviews predicted "
        "negative. Check the review count and average rating before "
        "drawing conclusions."
    )

    if ranked_products.empty:
        st.info("No products meet the minimum review threshold.")
    else:
        st.dataframe(
            ranked_products.head(20).round(2),
            use_container_width=True,
            hide_index=True
        )

# --------------------------------------------------
# 8. Aspect-level customer feedback
# --------------------------------------------------

st.header("3. Aspect-Level Customer Feedback")

st.write(
    "Explore what customers say about quality, durability, enjoyment, "
    "packaging, value for money, and other detected aspects."
)

required_aspect_columns = {
    "review_row_id",
    "aspect",
    "aspect_sentiment",
    "evidence_sentence"
}

if not required_aspect_columns.issubset(aspects.columns):
    missing = required_aspect_columns - set(aspects.columns)

    st.warning(
        "The aspect file is missing required columns: "
        + ", ".join(sorted(missing))
        + ". Please update the aspect-analysis script to include "
        "a stable review_row_id so repeated mentions can be counted "
        "as one review."
    )

elif aspects.empty:
    st.warning("No aspect mentions were detected.")

else:
    aspect_data = aspects.copy()

    # Remove rows without usable aspect labels
    aspect_data = aspect_data.dropna(
        subset=["review_row_id", "aspect"]
    )

    aspect_data["aspect"] = aspect_data["aspect"].astype(str)
    aspect_data["aspect_sentiment"] = (
        aspect_data["aspect_sentiment"].astype(str).str.lower()
    )

    aspect_data = aspect_data[
        aspect_data["aspect_sentiment"].isin(SENTIMENT_ORDER)
    ]

    # Build grouping columns that exist in the CSV
    review_aspect_keys = ["review_row_id", "aspect"]

    for col in ["parent_asin", "product_title"]:
        if col in aspect_data.columns:
            review_aspect_keys.append(col)

    # Count one result per review per aspect.
    # If a review contains both positive and negative sentences
    # about the same aspect, negative takes precedence.
    def summarise_review_sentiment(values):
        labels = set(values)

        if "negative" in labels:
            return "negative"
        if "positive" in labels:
            return "positive"
        return "neutral"

    aggregation = {
        "aspect_sentiment": (
            "aspect_sentiment",
            summarise_review_sentiment
        ),
        "evidence_sentence": (
            "evidence_sentence",
            "first"
        )
    }

    for col in ["rating", "overall_predicted_sentiment"]:
        if col in aspect_data.columns:
            aggregation[col] = (col, "first")

    review_aspect = (
        aspect_data.groupby(
            review_aspect_keys,
            dropna=False
        )
        .agg(**aggregation)
        .reset_index()
    )

    # Count unique reviews, not repeated aspect mentions
    aspect_summary = (
        review_aspect.groupby(["aspect", "aspect_sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SENTIMENT_ORDER, fill_value=0)
        .reset_index()
    )

    aspect_summary["total_reviews"] = (
        aspect_summary[SENTIMENT_ORDER].sum(axis=1)
    )

    # Interactive filters
    aspect_options = sorted(
        review_aspect["aspect"].dropna().unique()
    )

    selected_aspect = st.selectbox(
        "Choose an aspect",
        aspect_options
    )

    selected_aspect_sentiment = st.selectbox(
        "Filter aspect sentiment",
        ["All", "negative", "neutral", "positive"]
    )

    product_options = []

    if "product_title" in review_aspect.columns:
        product_options = sorted(
            review_aspect["product_title"]
            .dropna()
            .astype(str)
            .unique()
        )

    selected_product = "All products"

    if product_options:
        selected_product = st.selectbox(
            "Filter by product",
            ["All products"] + product_options
        )

    # Overall chart: all aspects, unique review counts
    st.subheader("Positive and negative reviews by aspect")

    aspect_chart = aspect_summary.melt(
        id_vars=["aspect", "total_reviews"],
        value_vars=SENTIMENT_ORDER,
        var_name="Sentiment",
        value_name="Reviews"
    )

    fig = px.bar(
        aspect_chart,
        x="aspect",
        y="Reviews",
        color="Sentiment",
        barmode="group",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"Sentiment": SENTIMENT_ORDER},
        title="Unique reviews mentioning each aspect"
    )

    fig.update_layout(xaxis_title="Aspect")
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics for selected aspect
    chosen_summary = aspect_summary[
        aspect_summary["aspect"] == selected_aspect
    ]

    if not chosen_summary.empty:
        row = chosen_summary.iloc[0]

        total_aspect_reviews = int(row["total_reviews"])
        neg_count = int(row["negative"])
        neu_count = int(row["neutral"])
        pos_count = int(row["positive"])

        negative_pct = (
            100 * neg_count / total_aspect_reviews
            if total_aspect_reviews else 0
        )

        a1, a2, a3, a4 = st.columns(4)

        a1.metric("Reviews mentioning aspect", total_aspect_reviews)
        a2.metric("Negative reviews", neg_count)
        a3.metric("Positive reviews", pos_count)
        a4.metric("Negative rate", f"{negative_pct:.1f}%")

    # Filter evidence rows
    aspect_view = review_aspect[
        review_aspect["aspect"] == selected_aspect
    ].copy()

    if selected_aspect_sentiment != "All":
        aspect_view = aspect_view[
            aspect_view["aspect_sentiment"]
            == selected_aspect_sentiment
        ]

    if (
        selected_product != "All products"
        and "product_title" in aspect_view.columns
    ):
        aspect_view = aspect_view[
            aspect_view["product_title"].astype(str)
            == selected_product
        ]

    st.subheader("Review evidence")

    evidence_columns = [
        col for col in [
            "product_title",
            "parent_asin",
            "rating",
            "aspect",
            "aspect_sentiment",
            "evidence_sentence",
            "review_row_id"
        ]
        if col in aspect_view.columns
    ]

    st.caption(
        "Each row represents one review for the selected aspect, "
        "not every repeated mention of that aspect."
    )

    st.dataframe(
        aspect_view[evidence_columns].head(500),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download aspect review evidence",
        data=aspect_view.to_csv(index=False).encode("utf-8"),
        file_name="aspect_review_evidence.csv",
        mime="text/csv"
    )

    # Product-aspect ranking
    if "parent_asin" in review_aspect.columns:
        st.subheader("Which products have the most aspect complaints?")

        product_aspect = (
            review_aspect.groupby(
                ["parent_asin", "aspect"]
                + (
                    ["product_title"]
                    if "product_title" in review_aspect.columns
                    else []
                )
            )
            .agg(
                total_reviews=("review_row_id", "nunique"),
                negative_reviews=(
                    "aspect_sentiment",
                    lambda s: (s == "negative").sum()
                ),
                positive_reviews=(
                    "aspect_sentiment",
                    lambda s: (s == "positive").sum()
                ),
                neutral_reviews=(
                    "aspect_sentiment",
                    lambda s: (s == "neutral").sum()
                )
            )
            .reset_index()
        )

        product_aspect["negative_rate_pct"] = (
            100 * product_aspect["negative_reviews"]
            / product_aspect["total_reviews"]
        )

        min_aspect_reviews = st.slider(
            "Minimum reviews mentioning this aspect",
            min_value=3,
            max_value=50,
            value=10,
            step=1
        )

        selected_product_aspect = product_aspect[
            (product_aspect["aspect"] == selected_aspect)
            & (
                product_aspect["total_reviews"]
                >= min_aspect_reviews
            )
        ].sort_values(
            ["negative_rate_pct", "negative_reviews"],
            ascending=[False, False]
        )

        st.caption(
            "This ranks products by negative sentiment among reviews "
            "that mention the selected aspect. It does not measure "
            "complaints as a percentage of all product reviews."
        )

        st.dataframe(
            selected_product_aspect.head(20).round(2),
            use_container_width=True,
            hide_index=True
        )

# --------------------------------------------------
# 9. Explore individual reviews
# --------------------------------------------------

st.header("4. Explore Individual Reviews")

search = st.text_input("Search review text")

review_view = filtered.copy()

if search:
    review_view = review_view[
        review_view["review_text"].str.contains(
            search,
            case=False,
            na=False,
            regex=False
        )
    ]

display_columns = [
    col for col in [
        "parent_asin",
        "rating",
        "review_text",
        "predicted_sentiment",
        "prediction_confidence",
        "sentiment_disagreement"
    ]
    if col in review_view.columns
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

# --------------------------------------------------
# 10. Analyze a new review
# --------------------------------------------------

st.header("5. Analyze a New Review")

new_review = st.text_area(
    "Enter a product review",
    placeholder=(
        "e.g. The toy is fun, but one of the pieces broke..."
    )
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
            color="Sentiment",
            range_y=[0, 1],
            color_discrete_map=SENTIMENT_COLORS,
            category_orders={"Sentiment": SENTIMENT_ORDER},
            title="Predicted probability by sentiment"
        )

        st.plotly_chart(fig, use_container_width=True)
