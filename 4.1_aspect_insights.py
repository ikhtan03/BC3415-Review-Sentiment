"""Precompute unique review/aspect counts and negative percentages.
Run: py prepare_aspect_insights.py
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
ASPECTS_FILE = BASE / "review_aspects.csv"
REVIEWS_FILE = BASE / "reviews_with_predictions.csv"

def sentiment_for_review(labels):
    labels = {str(x).strip().lower() for x in labels if pd.notna(x)}
    # Conservative rule for mixed evidence: negative takes precedence.
    if "negative" in labels:
        return "negative"
    if "positive" in labels:
        return "positive"
    return "neutral"

def find_id_column(df):
    for col in ["review_row_id", "review_id", "reviewId", "id"]:
        if col in df.columns:
            return col
    return None

def main():
    for path in [ASPECTS_FILE, REVIEWS_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Put it in the same folder as this script."
            )

    print("Loading aspect evidence...")
    aspects = pd.read_csv(ASPECTS_FILE, low_memory=False)
    required = {"parent_asin", "aspect", "aspect_sentiment", "evidence_sentence"}
    missing = required - set(aspects.columns)
    if missing:
        raise ValueError(f"review_aspects.csv is missing columns: {sorted(missing)}")

    id_col = find_id_column(aspects)
    if id_col is None:
        raise ValueError(
            "review_aspects.csv must contain a stable review_row_id column. "
            "Add this when generating aspect evidence."
        )
    if id_col != "review_row_id":
        aspects = aspects.rename(columns={id_col: "review_row_id"})

    aspects["review_row_id"] = aspects["review_row_id"].astype("string")
    aspects["parent_asin"] = aspects["parent_asin"].astype("string")
    aspects["aspect"] = aspects["aspect"].astype("string").str.strip()
    aspects["aspect_sentiment"] = (
        aspects["aspect_sentiment"].astype("string").str.strip().str.lower()
    )
    aspects["evidence_sentence"] = aspects["evidence_sentence"].fillna("").astype(str)
    aspects = aspects[
        aspects["aspect_sentiment"].isin(["positive", "neutral", "negative"])
    ].dropna(subset=["review_row_id", "parent_asin", "aspect"])

    keys = ["review_row_id", "parent_asin", "aspect"]
    for col in ["product_title", "rating", "overall_predicted_sentiment"]:
        if col in aspects.columns:
            keys.append(col)

    print("Counting each review once per aspect...")
    unique = (
        aspects.groupby(keys, dropna=False)
        .agg(
            aspect_sentiment=("aspect_sentiment", sentiment_for_review),
            evidence_sentence=("evidence_sentence", "first"),
        )
        .reset_index()
    )
    unique.to_csv(BASE / "review_aspects_unique.csv", index=False)
    print("Saved review_aspects_unique.csv:", unique.shape)

    print("Calculating total reviews per product...")
    reviews = pd.read_csv(REVIEWS_FILE, low_memory=False)
    if "parent_asin" not in reviews.columns:
        raise ValueError("reviews_with_predictions.csv must contain parent_asin.")
    reviews["parent_asin"] = reviews["parent_asin"].astype("string")
    reviews = reviews.dropna(subset=["parent_asin"])

    review_id = find_id_column(reviews)
    if review_id:
        totals = (
            reviews.groupby("parent_asin")[review_id].nunique()
            .rename("total_product_reviews").reset_index()
        )
    else:
        # Assumes one row in reviews_with_predictions.csv represents one review.
        totals = (
            reviews.groupby("parent_asin").size()
            .rename("total_product_reviews").reset_index()
        )

    title_source = None
    if "product_title" in reviews.columns:
        title_source = reviews[["parent_asin", "product_title"]].dropna()
    elif "product_title" in unique.columns:
        title_source = unique[["parent_asin", "product_title"]].dropna()
    if title_source is not None and not title_source.empty:
        title_source = title_source.drop_duplicates("parent_asin")
        totals = totals.merge(title_source, on="parent_asin", how="left")

    group_keys = ["parent_asin", "aspect"]
    if "product_title" in unique.columns:
        group_keys.append("product_title")

    summary = (
        unique.groupby(group_keys, dropna=False)
        .agg(
            total_aspect_reviews=("review_row_id", "nunique"),
            negative_reviews=("aspect_sentiment", lambda s: (s == "negative").sum()),
            neutral_reviews=("aspect_sentiment", lambda s: (s == "neutral").sum()),
            positive_reviews=("aspect_sentiment", lambda s: (s == "positive").sum()),
        )
        .reset_index()
    )
    summary["negative_rate_pct"] = (
        100 * summary["negative_reviews"] / summary["total_aspect_reviews"].clip(lower=1)
    )

    summary = summary.merge(totals, on="parent_asin", how="left")
    summary["total_product_reviews"] = summary["total_product_reviews"].fillna(0)
    summary["negative_pct_all_product_reviews"] = (
        100 * summary["negative_reviews"]
        / summary["total_product_reviews"].clip(lower=1)
    )
    summary = summary.sort_values(
        ["negative_rate_pct", "negative_reviews"], ascending=[False, False]
    )
    summary.to_csv(BASE / "product_aspect_summary.csv", index=False)
    print("Saved product_aspect_summary.csv:", summary.shape)

    overall = (
        unique.groupby(["aspect", "aspect_sentiment"]).size()
        .unstack(fill_value=0)
        .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
        .reset_index()
    )
    overall["total_aspect_reviews"] = overall[["positive", "neutral", "negative"]].sum(axis=1)
    overall["negative_rate_pct"] = (
        100 * overall["negative"] / overall["total_aspect_reviews"].clip(lower=1)
    )
    overall.to_csv(BASE / "aspect_summary.csv", index=False)
    print("Saved aspect_summary.csv:", overall.shape)
    print("\nDone. Next run: py -m streamlit run app.py")

if __name__ == "__main__":
    main()
