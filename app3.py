"""Retail Review Intelligence dashboard.
Run preparation first: py prepare_aspect_insights.py
Then launch: py -m streamlit run app.py
"""
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
REVIEWS_FILE = BASE / "reviews_with_predictions.csv"
MODEL_FILE = BASE / "sentiment_model.joblib"
UNIQUE_FILE = BASE / "review_aspects_unique.csv"
PRODUCT_ASPECT_FILE = BASE / "product_aspect_summary.csv"
OVERALL_ASPECT_FILE = BASE / "aspect_summary.csv"

SENTIMENTS = ["positive", "neutral", "negative"]
COLORS = {"positive": "#16A34A", "neutral": "#2563EB", "negative": "#DC2626"}

st.set_page_config(page_title="Retail Review Intelligence", page_icon="🧸", layout="wide")
st.title("🧸 Retail Review Intelligence")
st.caption("Analyze customer sentiment, investigate product issues, and identify aspects to improve.")

def check_file(path):
    if not path.exists():
        st.error(f"File not found: {path.name}. Put it in the same folder as app.py.")
        st.stop()

def mtime(path):
    return path.stat().st_mtime if path.exists() else 0

@st.cache_data(show_spinner="Loading reviews...")
def load_reviews(path, modified):
    return pd.read_csv(path, low_memory=False)

@st.cache_data(show_spinner="Loading prepared aspect data...")
def load_csv(path, modified):
    return pd.read_csv(path, low_memory=False)

@st.cache_resource(show_spinner="Loading sentiment model...")
def load_model(path, modified):
    return joblib.load(path)

check_file(REVIEWS_FILE)
check_file(MODEL_FILE)
df = load_reviews(str(REVIEWS_FILE), mtime(REVIEWS_FILE))
model = load_model(str(MODEL_FILE), mtime(MODEL_FILE))

if "review_text" not in df.columns or "predicted_sentiment" not in df.columns:
    st.error("reviews_with_predictions.csv needs review_text and predicted_sentiment columns.")
    st.stop()

df["review_text"] = df["review_text"].fillna("").astype(str)
df["predicted_sentiment"] = df["predicted_sentiment"].astype(str).str.lower()

# 1. Sidebar filters
st.sidebar.header("Filter reviews")
available = [s for s in SENTIMENTS if s in set(df["predicted_sentiment"].dropna())]
selected_sentiments = st.sidebar.multiselect("Predicted sentiment", available, default=available)
filtered = df[df["predicted_sentiment"].isin(selected_sentiments)].copy()

if "rating" in filtered.columns:
    rating_values = sorted(filtered["rating"].dropna().unique())
    selected_ratings = st.sidebar.multiselect("Star rating", rating_values, default=rating_values)
    filtered = filtered[filtered["rating"].isin(selected_ratings)]

# 2. Overview
st.header("1. Customer Feedback Overview")
n = len(filtered)
positive_pct = 100 * (filtered["predicted_sentiment"] == "positive").mean() if n else 0
negative_pct = 100 * (filtered["predicted_sentiment"] == "negative").mean() if n else 0
disagreement_pct = (
    100 * filtered["sentiment_disagreement"].fillna(False).astype(bool).mean()
    if n and "sentiment_disagreement" in filtered.columns else 0
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews in view", f"{n:,}")
c2.metric("Predicted positive", f"{positive_pct:.1f}%")
c3.metric("Predicted negative", f"{negative_pct:.1f}%")
c4.metric("Rating-label disagreements", f"{disagreement_pct:.1f}%")

left, right = st.columns(2)
with left:
    st.subheader("Customer sentiment distribution")
    if n:
        counts = filtered["predicted_sentiment"].value_counts().reindex(SENTIMENTS, fill_value=0).rename_axis("Sentiment").reset_index(name="Reviews")
        fig = px.bar(counts, x="Sentiment", y="Reviews", color="Sentiment", text="Reviews",
                     color_discrete_map=COLORS, category_orders={"Sentiment": SENTIMENTS})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No reviews match the filters.")
with right:
    st.subheader("Sentiment by star rating")
    if n and "rating" in filtered.columns:
        rating_counts = filtered.groupby(["rating", "predicted_sentiment"]).size().reset_index(name="Reviews")
        fig = px.bar(rating_counts, x="rating", y="Reviews", color="predicted_sentiment", barmode="group",
                     color_discrete_map=COLORS, category_orders={"predicted_sentiment": SENTIMENTS})
        st.plotly_chart(fig, use_container_width=True)

# 3. Overall product ranking
st.header("2. Products Requiring Investigation")
if "parent_asin" in filtered.columns:
    aggs = {
        "review_count": ("review_text", "size"),
        "negative_reviews": ("predicted_sentiment", lambda s: (s == "negative").sum()),
    }
    if "rating" in filtered.columns:
        aggs["average_rating"] = ("rating", "mean")
    if "product_title" in filtered.columns:
        aggs["product_title"] = ("product_title", "first")
    products = filtered.groupby("parent_asin", dropna=False).agg(**aggs).reset_index()
    products["negative_rate_pct"] = 100 * products["negative_reviews"] / products["review_count"].clip(lower=1)
    threshold = st.slider("Minimum reviews for product ranking", 5, 100, 20, 5)
    products = products[products["review_count"] >= threshold].sort_values(
        ["negative_rate_pct", "negative_reviews"], ascending=[False, False]
    )
    st.dataframe(products.head(30).round(2), use_container_width=True, hide_index=True)

# 4. Product-specific aspect analysis
st.header("3. Product Aspect Improvement")
prepared_files = [UNIQUE_FILE, PRODUCT_ASPECT_FILE, OVERALL_ASPECT_FILE]
if not all(path.exists() for path in prepared_files):
    st.warning("Prepared aspect files are missing. Run `py prepare_aspect_insights.py` first.")
else:
    unique = load_csv(str(UNIQUE_FILE), mtime(UNIQUE_FILE))
    product_aspects = load_csv(str(PRODUCT_ASPECT_FILE), mtime(PRODUCT_ASPECT_FILE))
    overall_aspects = load_csv(str(OVERALL_ASPECT_FILE), mtime(OVERALL_ASPECT_FILE))

    if unique.empty or product_aspects.empty:
        st.info("No aspect evidence is available.")
    else:
        product_aspects["parent_asin"] = product_aspects["parent_asin"].astype(str)
        unique["parent_asin"] = unique["parent_asin"].astype(str)
        unique["aspect"] = unique["aspect"].astype(str)
        unique["aspect_sentiment"] = unique["aspect_sentiment"].astype(str).str.lower()

        pcols = ["parent_asin"] + (["product_title"] if "product_title" in product_aspects.columns else [])
        product_choices = product_aspects[pcols].drop_duplicates("parent_asin").fillna("")
        product_choices["parent_asin"] = product_choices["parent_asin"].astype(str)
        has_title = "product_title" in product_choices.columns
        options = list(product_choices.itertuples(index=False, name=None))

        def product_label(option):
            asin = str(option[0])
            title = str(option[1]) if has_title else ""
            return f"{title} ({asin})" if title.strip() else asin

        selected = st.selectbox("Choose a product", options, format_func=product_label)
        asin = str(selected[0])
        summary = product_aspects[product_aspects["parent_asin"] == asin].copy()

        min_aspect_reviews = st.slider("Minimum reviews mentioning an aspect", 1, 50, 5)
        ranked = summary[summary["total_aspect_reviews"] >= min_aspect_reviews].sort_values(
            ["negative_rate_pct", "negative_reviews"], ascending=[False, False]
        )

        if ranked.empty:
            st.info("No aspects meet this threshold. Try lowering it.")
        else:
            worst = ranked.iloc[0]
            a, b, c = st.columns(3)
            a.metric("Aspects with enough evidence", len(ranked))
            b.metric("Most negative aspect", str(worst["aspect"]))
            c.metric("Negative rate among aspect reviews", f"{worst['negative_rate_pct']:.1f}%")

            st.subheader("Aspects with the highest negative rates")
            chart = ranked.head(10).sort_values("negative_rate_pct", ascending=True)
            fig = px.bar(
                chart, x="negative_rate_pct", y="aspect", orientation="h",
                text=chart["negative_rate_pct"].map(lambda v: f"{v:.1f}%"),
                hover_data=["negative_reviews", "total_aspect_reviews", "total_product_reviews",
                            "negative_pct_all_product_reviews"],
                color_discrete_sequence=[COLORS["negative"]],
            )
            fig.update_layout(
                xaxis_title="Negative reviews among reviews mentioning aspect (%)",
                yaxis_title="Aspect", showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Negative rate among aspect reviews = negative reviews mentioning the aspect / "
                "all reviews mentioning it. Negative % of all product reviews = negative reviews "
                "mentioning the aspect / all reviews for the product."
            )
            columns = [
                "aspect", "total_aspect_reviews", "negative_reviews", "neutral_reviews",
                "positive_reviews", "negative_rate_pct", "total_product_reviews",
                "negative_pct_all_product_reviews"
            ]
            st.dataframe(ranked[columns].round(2), use_container_width=True, hide_index=True)

            aspect_choice = st.selectbox("Inspect evidence for aspect", ranked["aspect"].tolist())
            sentiment_choice = st.selectbox("Evidence sentiment", ["negative", "neutral", "positive", "All"])
            evidence = unique[(unique["parent_asin"] == asin) & (unique["aspect"] == aspect_choice)].copy()
            if sentiment_choice != "All":
                evidence = evidence[evidence["aspect_sentiment"] == sentiment_choice]
            evidence_cols = [c for c in [
                "product_title", "parent_asin", "rating", "aspect", "aspect_sentiment",
                "evidence_sentence", "review_row_id"
            ] if c in evidence.columns]
            st.dataframe(evidence[evidence_cols].head(200), use_container_width=True, hide_index=True)
            st.download_button(
                "Download selected aspect evidence",
                data=evidence.to_csv(index=False).encode("utf-8"),
                file_name="selected_product_aspect_evidence.csv",
                mime="text/csv",
            )

            st.subheader("Overall aspect sentiment")
            sentiment_cols = [s for s in SENTIMENTS if s in overall_aspects.columns]
            if sentiment_cols:
                melted = overall_aspects.melt(id_vars=["aspect"], value_vars=sentiment_cols,
                                               var_name="Sentiment", value_name="Reviews")
                fig = px.bar(melted, x="aspect", y="Reviews", color="Sentiment", barmode="group",
                             color_discrete_map=COLORS, category_orders={"Sentiment": SENTIMENTS})
                st.plotly_chart(fig, use_container_width=True)

# 5. Review explorer
st.header("4. Explore Individual Reviews")
search = st.text_input("Search review text")
review_view = filtered
if search:
    review_view = review_view[review_view["review_text"].str.contains(search, case=False, na=False, regex=False)]
display_cols = [c for c in [
    "parent_asin", "rating", "review_text", "predicted_sentiment",
    "prediction_confidence", "sentiment_disagreement"
] if c in review_view.columns]
st.dataframe(review_view[display_cols].head(500), use_container_width=True, hide_index=True)
st.download_button(
    "Download filtered reviews",
    data=review_view.to_csv(index=False).encode("utf-8"),
    file_name="filtered_review_insights.csv",
    mime="text/csv",
)

# 6. Live prediction (form prevents rerun on every keystroke)
st.header("5. Analyze a New Review")
with st.form("new_review_form"):
    new_review = st.text_area("Enter a product review",
                              placeholder="The toy is fun, but one of the pieces broke...",
                              height=140)
    submitted = st.form_submit_button("Analyze review")
if submitted:
    if not new_review.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            with st.spinner("Analyzing review..."):
                prediction = model.predict([new_review])[0]
                probabilities = model.predict_proba([new_review])[0]
                if hasattr(model, "named_steps") and "classifier" in model.named_steps:
                    classes = model.named_steps["classifier"].classes_
                else:
                    classes = getattr(model, "classes_", [str(i) for i in range(len(probabilities))])
            st.write("**Predicted sentiment:**", prediction)
            st.write("**Highest class probability:**", f"{max(probabilities):.1%}")
            probability_df = pd.DataFrame({"Sentiment": classes, "Probability": probabilities})
            fig = px.bar(probability_df, x="Sentiment", y="Probability", color="Sentiment",
                         range_y=[0, 1], color_discrete_map=COLORS,
                         category_orders={"Sentiment": SENTIMENTS},
                         title="Predicted probability by sentiment")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not analyze this review: {exc}")
