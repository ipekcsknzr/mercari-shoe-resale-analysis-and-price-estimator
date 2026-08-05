# Mercari Women’s Shoe Resale Analysis & Price Estimator

## Project Overview

This project uses women’s shoe listings from the Mercari Price Suggestion Challenge dataset to study resale-price patterns and build a deployable machine-learning pricing tool.

The project has two main components:

1. **Condition-retention analysis**  
   Measures how each brand’s median listing price changes as item condition worsens within the same shoe type.

2. **Machine-learning price-range estimator**  
   Uses brand, shoe type, condition, listing text, and engineered features to generate a suggested Mercari listing-price range.

The analysis covers seven women’s shoe categories and uses **53,176 cleaned listings**.

The selected deployment model is a **Premium-Feature Ridge Regression model**. It was chosen because it offered a strong balance between simplicity, interpretability, proportional accuracy, and dollar accuracy.

Final test-set performance:

- **Mean Absolute Log Error (MALE):** 0.304
- **Median Absolute Percentage Error (MdAPE):** 23.7%
- **Mean Absolute Error (MAE):** $12.16
- **Root Mean Squared Error (RMSE):** $25.46
- **R²:** 0.604

A Streamlit application loads the trained model and returns a practical suggested listing range of **±15% around the model prediction**.

The displayed range is intended as pricing guidance. It is not a formal confidence interval, guaranteed sale price, or prediction of a completed transaction.

## Dataset

The original dataset is the Mercari Price Suggestion Challenge dataset. It contains approximately **1.48 million marketplace listings**.

After narrowing down the total marketplace listings to **77,654 women’s shoe listings**, the dataset was filtered and cleaned to keep out listings with missing/blank brand names and/or a price of $0, and the final dataset contained **53,176 women’s shoe listings**.

## Shoe Categories

The project analyzes **seven** shoe types:

- Athletic
- Boots
- Sandals
- Fashion Sneakers
- Pumps
- Flats
- Loafers & Slip-Ons

Each brand is analyzed separately within each shoe type.

## Data Cleaning

The cleaning process included:

- Filtering the original dataset to women’s shoes
- Removing missing or blank brand names
- Removing zero-price listings
- Standardizing shoe categories into seven shoe types
- Combining original condition levels 4 and 5
- Saving the cleaned dataset as a Parquet file

The final condition groups are Condition 1 (best), Condition 2, Condition 3, and Condition 4–5 (worst).

## Methodology

For every brand and shoe-type combination, the analysis calculates listing counts, median listing prices, and price retention relative to Condition 1.

Median price is used because it is less sensitive than the mean to unusually high or low marketplace listings.

### Price Retention

```text
Condition 3 Median Price
------------------------- × 100
Condition 1 Median Price
```

A retention value of **65%** means that the brand’s Condition 3 median listing price is 65% of its own Condition 1 median listing price within the same shoe type.

A value above 100% is labeled as a **non-monotonic price pattern**. It does not mean that worse condition increases value. These cases may reflect differences in models, styles, listing composition, or seller-reported condition.

## Minimum Sample Requirement

A brand and shoe-type combination must have at least 10 Condition 1 listings and 10 listings in a comparison condition.

## What Are “High-Data Brands”?

The visualizations display up to eight **high-data brands** for each shoe category.

“High-data brands” are the brands with the largest combined number of qualifying Condition 1 and Condition 3 listings within that shoe type.

```text
Combined sample size =
Condition 1 listing count + Condition 3 listing count
```

These brands are selected because they have more available data, not because they have the highest prices or the best condition retention.

The charts should therefore not be interpreted as rankings of the “best” brands. The high-data selection is used only to create clearer and more reliable comparisons.

## Example Visualization

The chart below shows Condition 3 price retention for high-data Athletic shoe brands. Each percentage compares a brand’s Condition 3 median listing price with that same brand’s Condition 1 median listing price.

The brands shown were selected based on sample size, not price or retention performance.

![Athletic Shoe Condition 3 Retention](outputs/charts/athletic_condition_3_retention.png)

## Visualizations

Two charts are created for every shoe category:

1. **Condition 1 to Condition 3 Price Change**  
   A slope chart showing each high-data brand’s Condition 1 and Condition 3 median listing prices.

2. **Condition 3 Price Retention**  
   A horizontal bar chart showing Condition 3 median price as a percentage of the brand’s own Condition 1 median price.

## Project Structure

```text
mercari-shoe-resale-analysis/
├── app.py
├── data/
│   ├── raw/
│   │   └── train.tsv
│   └── processed/
│       ├── clean_womens_shoes.parquet
│       ├── brand_shoe_condition_retention.parquet
│       └── brand_shoe_bootstrap_intervals.parquet
├── models/
│   └── mercari_shoe_price_estimator.joblib
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_shoe_data_cleaning.ipynb
│   ├── 03_exploratory_brand_tiers.ipynb
│   ├── 04_condition_retention_analysis.ipynb
│   ├── 05_visualizations.ipynb
│   ├── 06_bootstrap_confidence_intervals.ipynb
│   └── 07_shoe_price_estimator.ipynb
├── outputs/
│   ├── charts/
│   └── tables/
│       └── brand_shoe_condition_retention.csv
├── .gitignore
├── README.md
└── requirements.txt

```
## Notebook Workflow

### `01_data_exploration.ipynb`

Explores dataset dimensions, missing values, category structure, brand frequency, prices, and item conditions.

### `02_shoe_data_cleaning.ipynb`

Filters and cleans the women’s shoe data, creates broader shoe types and condition groups, and saves the processed dataset.

### `03_exploratory_brand_tiers.ipynb`

Contains an earlier exploratory brand-tier approach. It is retained to document the development process but is not part of the final methodology.

### `04_condition_retention_analysis.ipynb`

Calculates listing counts, median prices, condition-retention percentages, and expected versus non-monotonic patterns.

### `05_condition_visualizations.ipynb`

Creates the final slope charts and retention-percentage charts for all seven shoe categories.

### `06_bootstrap_confidence_intervals.ipynb` 

Estimates 95% confidence intervals using bootstrap resampling

### `07_shoe_price_estimator.ipynb` 

Builds, compares, tunes, and evaluates machine-learning price prediction models

## Findings

Approximately **92.6%** of the analyzed brand and shoe-type combinations showed an expected decline pattern, while **7.4%** showed a non-monotonic pattern.

These results should be interpreted cautiously because the dataset does not identify identical shoe models across condition groups.

## Price Estimation Model

The project includes a machine-learning model that estimates the expected listing price of a women’s shoe on Mercari.

The model uses:

- Brand
- Shoe type
- Item condition
- Listing title
- Item description
- Premium keyword indicators
- Title length
- Description length

The Streamlit application makes the listing title optional. When no title is entered, the app uses the selected brand and shoe type as fallback text.

Categorical features are transformed using one-hot encoding. Listing titles and descriptions are converted into numerical features using word-level TF-IDF.

Additional indicators identify terms such as:

- `limited`
- `rare`
- `vintage`
- `authentic`
- `designer`
- `luxury`
- `leather`
- `suede`
- `new`
- `nwt`
- `nwot`

Because listing prices are strongly right-skewed, the model predicts `log(price + 1)` and converts predictions back into dollars.

### Models Tested

The following approaches were evaluated:

- Median-price baseline
- Linear Regression
- Ridge Regression
- Tuned Ridge Regression
- Title-Enhanced Ridge Regression
- Description-Enhanced Ridge Regression
- Premium-Feature Ridge Regression
- Weighted Ridge Regression
- Character-Enhanced Weighted Ridge Regression
- XGBoost Regression

### Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Median Baseline | $21.75 | $41.95 | -0.073 |
| Linear Regression | $14.83 | $30.03 | 0.450 |
| Ridge Regression | $14.85 | $30.25 | 0.442 |
| Title-Enhanced Ridge | $12.65 | $26.53 | 0.571 |
| Description-Enhanced Ridge | $12.38 | $25.73 | 0.596 |
| Tuned Ridge | $12.22 | $25.69 | 0.597 |
| **Premium-Feature Ridge** | **$12.16** | **$25.46** | **0.604** |
| Balanced Weighted Ridge | $12.19 | $24.59 | 0.631 |
| XGBoost | $13.07 | $27.24 | 0.547 |
| Character-Enhanced Weighted Ridge | $12.15 | $24.28 | 0.640 |

The Character-Enhanced Weighted Ridge model achieved the strongest raw RMSE and R² results. However, the **Premium-Feature Ridge model** was selected for deployment because it remained nearly tied in dollar MAE while requiring less preprocessing and no special sample weighting.

### Selected Deployment Model

The deployed model combines:

- One-hot encoded brand and shoe type
- Item condition
- Word-level TF-IDF features from listing titles
- Word-level TF-IDF features from item descriptions
- Premium keyword indicators
- Title and description length
- Ridge Regression with `alpha = 2.0`

Final selected-model performance:

| Metric | Result |
|---|---:|
| Mean Absolute Log Error | 0.304 |
| Median Absolute Percentage Error | 23.7% |
| Mean Absolute Error | $12.16 |
| Root Mean Squared Error | $25.46 |
| R² | 0.604 |

MALE is used as the primary proportional-error metric because a fixed dollar error has a different practical meaning for inexpensive and expensive shoes.

MdAPE shows that half of the model’s predictions had an absolute percentage error of approximately **23.7% or less**.

MAE is retained because it gives an intuitive dollar-based interpretation of model error.

### Suggested Listing Range

The Streamlit application does not display one exact price as a guaranteed value. Instead, it shows a suggested listing range calculated as:

```text
Lower bound = Predicted price × 0.85
Upper bound = Predicted price × 1.15

```

### Performance by Price Range

| Actual Listing Price | Number of Test Listings | MAE |
|---|---:|---:|
| $0–$25 | 4,341 | $6.80 |
| $25–$50 | 3,833 | $9.03 |
| $50–$100 | 1,880 | $17.26 |
| $100+ | 582 | $56.06 |

The estimator performs best for typical Mercari listings under $50. Rare and expensive listings remain more difficult because they appear less frequently and often depend on details not fully captured in the dataset, such as exact model rarity, authenticity, release year, or original retail price.

## Limitations

- The dataset contains listing prices rather than confirmed transaction prices.
- The model estimates an appropriate listing range, not a guaranteed resale value.
- The ±15% displayed range is a practical recommendation and not a formal confidence interval.
- Testing showed that a ±15% interval contained approximately 33.2% of test-set actual prices.
- Rare and expensive listings remain more difficult to predict.
- The dataset does not include original retail price, release year, authenticity verification, exact shoe model, shoe size, seller reputation, or completed-sale status.
- Item condition is self-reported by sellers and may be inconsistent.
- Condition 5 contains relatively few listings, so predictions for that condition may be less reliable.
- The model was trained on historical Mercari data and may not capture current marketplace trends.

## Next Steps

- Deploy the Streamlit application publicly
- Add a screenshot or demonstration GIF to the README
- Evaluate the model on newer or external resale-market data
- Improve prediction quality for rare and premium listings
- Add reliability warnings based on brand and shoe-type sample size
- Explore more formal prediction-interval methods separately from the practical ±15% listing range
- Improve the interface for unknown or manually entered brands
- Add automated testing for feature engineering and model inference

## Tools Used

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Matplotlib
- Jupyter Notebook
- Streamlit
- joblib
- Parquet
- VS Code
- Git
- GitHub

