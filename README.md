# Retail Review Intelligence
## Amazon Toys & Games Review Sentiment and Product-Aspect Analysis

## Project overview

Retail Review Intelligence analyses Amazon Toys & Games reviews from the [Amazon Reviews'23 dataset](https://amazon-reviews-2023.github.io/). It combines three-class sentiment classification, product-level analysis, and aspect-level review evidence in a Streamlit dashboard. The aim is to help users investigate customer feedback, compare review sentiment with star ratings, and identify product aspects that may warrant further investigation.

**Important:** Sentiment labels derived from star ratings are proxy labels, not independently human-annotated sentiment. Model predictions and aspect summaries are estimates. Check the original review evidence before making decisions.

## Main capabilities

1. **Customer Feedback Overview** — view predicted sentiment distribution, compare sentiment with star ratings, and inspect disagreement between predicted sentiment and rating-derived labels.
2. **Products Requiring Investigation** — rank products by negative-review rate while applying a minimum review-count threshold.
3. **Product Aspect Improvement** — compare the negative rate for aspects discussed in a selected product's reviews.
4. **Evidence Inspection** — view example evidence sentences for an aspect and filter evidence by sentiment.
5. **Explore Individual Reviews** — search review text, inspect predictions and confidence, and download filtered reviews.
6. **Analyze a New Review** — submit new text to the saved classifier and inspect its predicted sentiment and class probabilities.

## Data and sentiment labels

The source is the Amazon Reviews'23 Toys & Games review dataset. The raw dataset is large and is not included in this repository; obtain it from the dataset provider and place it where the relevant notebook expects it.

The modelling labels are derived from the numerical rating:

| Rating | Proxy sentiment |
|---|---|
| 1–2 stars | Negative |
| 3 stars | Neutral |
| 4–5 stars | Positive |

This lets the project train a three-class classifier, but it also means evaluation measures agreement with rating-derived labels. A positive rating can contain critical wording, and a low rating can contain mixed feedback.

## Workflow and files

Run notebooks from top to bottom, then run the scripts in dependency order. 

### 1. `0_bal_data.ipynb` — balanced sampling

- Reads the source review data.
- Maps ratings to negative, neutral, and positive proxy labels.
- Samples up to 50,000 records per class for a target of up to 150,000 records before later cleaning.
- Saves `balanced_3class_sentiment.csv`.

The actual sample can be smaller if the source has fewer eligible examples. Because the sample is balanced, its sentiment proportions should not be presented as the prevalence of sentiment in the full Amazon review population.

### 2. `1_data_preprocessing.ipynb` — text cleaning and duplicate handling

- Loads `balanced_3class_sentiment.csv`.
- Selects relevant review fields, filters invalid ratings and empty review text, and combines title and body into `review_text`.
- Recreates the three sentiment labels from the rating rules above.
- Removes exact duplicate rows.
- Within each user and identical `review_text`, removes repeated reviews posted within 24 hours of the last retained review. This rule is based on the same user and same text; it does **not** remove all identical text written by different users.
- Retains records with missing timestamps rather than using them in the 24-hour comparison.
- Saves `reviews.csv` for modelling.

For this particular dataset, the final review count became 147,640.

### 3. `2_toys_reviews_models.ipynb` — model comparison

The notebook uses a stratified 80/20 train/test split and compares the approaches implemented in the notebook, including:

- TextBlob baseline.
- TF-IDF + Logistic Regression.
- TF-IDF + Random Forest.
- Sentence-transformer embeddings + Logistic Regression.
- A pretrained transformer sentiment model.

The reported evaluation includes accuracy, macro F1, weighted F1, class-level metrics, and confusion matrices. Use the latest notebook output for final values. Macro F1 is a useful selection metric when performance across all three classes matters because it weights each class equally.

The current workflow described for this project selects TF-IDF + Logistic Regression by macro F1 and saves the classifier as `sentiment_model.joblib`. The notebook also generates `reviews_with_predictions.csv`, including predicted sentiment, prediction confidence, rating-derived sentiment, and a disagreement indicator. Confirm those filenames and the selected model against your final run before submission.

### 4. `3_product_and_reviews.py` — product metadata join

Joins review predictions to product metadata using `parent_asin`, the parent ID of the product. The notebook checks for duplicate metadata IDs and uses a many-to-one merge validation so duplicate product metadata does not silently multiply review rows. The output is the product-linked review data used for product-level exploration, saved in `reviews_with_products.csv`. 

### 5. `4.1_aspect_insights.py` - aspect evidence generation 

The aspect evidence file ( `review_aspects.csv`) is an input to the preparation script below. It must contain these columns:

- `review_row_id` (or a supported alternative ID column such as `review_id`, `reviewId`, or `id`)
- `parent_asin`
- `aspect`
- `aspect_sentiment`
- `evidence_sentence`

Optional useful fields include `product_title`, `rating`, and `overall_predicted_sentiment`. The aspect-generation code should preserve a stable review identifier so multiple sentences from the same review can be counted once per aspect. Aspect names used in this project include:

- Quality and durability
- Broken or missing parts
- Enjoyment and play
- Ease of use and instructions
- Delivery and arrival
- Packaging
- Value for money
- Design and appearance
- Safety

The aspect-generation stage may use keyword rules to identify evidence and sentiment classification to label evidence. Inspect the actual aspect-generation code to confirm the exact method. Keyword rules can miss implicit mentions and can misinterpret context, negation, or words with multiple meanings.

### 6. `5.product_insights.py` — unique aspect counts

This script reads `review_aspects.csv` and `reviews_with_predictions.csv`, then creates:

- `review_aspects_unique.csv` — one record per review/product/aspect.
- `product_aspect_summary.csv` — product/aspect sentiment counts and negative percentages.
- `aspect_summary.csv` — overall sentiment counts and negative rate for each aspect.

When the same review has multiple sentiment labels for one aspect, the script applies a conservative rule: negative takes precedence, then positive, then neutral. It counts a review once per product/aspect, regardless of how many evidence sentences mention that aspect.

#### Negative percentage definitions

**Negative rate among reviews mentioning an aspect**

`negative_rate_pct = negative_reviews / total_aspect_reviews * 100`

This answers: among reviews that discuss the aspect, what percentage are negative for that aspect?

**Negative aspect reviews as a percentage of all product reviews**

`negative_pct_all_product_reviews = negative_reviews / total_product_reviews * 100`

This answers: what percentage of all reviews for the product contain negative feedback about the aspect?

These percentages have different denominators and should not be used interchangeably. A review that does not mention an aspect is excluded from the first denominator but remains in the second. Always interpret a high percentage alongside its review count; a high rate based on very few reviews is uncertain.

### 7. `app3.py` — Streamlit dashboard

The app loads `reviews_with_predictions.csv`, `sentiment_model.joblib`, and the three prepared aspect-summary CSVs. It provides the six capabilities listed above. It uses cached loading for data and the model, so filters do not need to rerun model training or aspect aggregation.

The script and its required files should be in the same folder unless you change the file paths in the code.

## Setup and running

Use the same Python environment for installation and execution. From the project folder, install the core app dependencies:

```bash
py -m pip install pandas scikit-learn joblib plotly streamlit
```

Install any additional packages required by the notebooks (for example, the NLP, transformer, or sentence-transformer libraries used by your model experiments).

Place the following files beside `5_product_insights.py` and `app3.py`:

```text
reviews_with_predictions.csv
sentiment_model.joblib
review_aspects.csv
prepare_aspect_insights.py
app.py
```

Then run the aspect-summary preparation and launch the dashboard:

```bash
py 5_product_insights.py
py -m streamlit run app3.py
```

If the browser tab opens but the dashboard does not load, check the terminal for the first error message. Common causes include missing CSV/model files, running Streamlit from the wrong Python environment, missing packages, or column names that do not match the expected inputs. The app loads the full `reviews_with_predictions.csv` into memory, so very large CSVs may need more RAM; use precomputed summaries and consider a smaller demo CSV if startup is slow.

## Limitations and responsible interpretation

1. **Proxy labels:** the model is trained to predict rating-derived labels, not an independently annotated emotional sentiment target.
2. **Balanced sample:** class frequencies in the balanced modelling dataset do not estimate real-world sentiment prevalence.
3. **Duplicate rule scope:** the preprocessing rule removes exact duplicate rows and repeated same-user/same-text reviews within 24 hours; it does not guarantee removal of near-duplicates across different users.
4. **Aspect detection:** keyword-based detection can miss implicit concerns, context, sarcasm, and negation.
5. **Overall vs aspect sentiment:** a review can be positive overall but negative about one aspect, such as durability.
6. **Confidence:** a classifier's maximum class probability is not necessarily a calibrated probability of correctness.
7. **Small denominators:** aspect rates based on few reviews can be unstable. Display counts and inspect the supporting sentences.
8. **Association is not causation:** complaints about delivery or packaging do not necessarily mean the product itself is defective.
