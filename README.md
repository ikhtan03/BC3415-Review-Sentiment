# BC3415-Review-Sentiment

## Dataset
Dataset is the Amazon Toys and Games Dataset from https://amazon-reviews-2023.github.io/. The file is very large so it isn't uploaded here.

## Preprocessing
We take the assumption that ratings correspond to sentiment, to carry out supervised ML models. Specifically, 1-2 is negative, 3 is neutral, 4-5 is positive. The reviews were sampled from the overall dataset such that the dataset is balanced between the 3 sentiments, with 50,000 observations per sentiment class. 

Under pre-processing (1.1_bal_data.ipynb), duplicate reviews were removed. In addition, if reviews were identical for different parent_asin and were posted within 24 hours of each other, the duplicated were also removed. This prevented the reviews analysed from potentially being both in training and test sets, and hence cause overfitting. This left a total review count of 147,640.

Review title and review text were combined into a single review text, which is used as the main predictor. Data was also split into 80% training and 20% test split. 

## Models
Several models were used to carry out the analysis, and the best model was chosen based on Macro F1. The models included were
- Textblob
- TF-IDF (Term Frequency-Inverse Document Frequency) with Logistic Regression
- TF-IDF (Term Frequency-Inverse Document Frequency) with Random Forest
- BERT with Logistic Regression
- Transformer

In this case, the model with the highest Macro F1 on the test set was TF-IDF with Logistic Regression, so this model was used for further review analyses. 

## Product Insights
Product reviews were further analysed to determine sentiments regarding aspects of the product, based on specific keywords corresponding to those aspects. The specific aspects considered were:
- Quality and durability
- Broken or missing parts
- Enjoyment and play
- Ease of use and instruction
- Delivery and arrival
- Packaging
- Value for money
- Design and appearance
- Safety

These were then analysed to see how product-specific aspects were perceived, based on the frequency of key phrases corresponding to each aspect. The phrases were analysed using the TF-IDF model to determine whether they have positive, neutral or negative sentiment attached to them. Positive and negative rates were also computed for each product and aspect.

## App and Web Interface
The app...
