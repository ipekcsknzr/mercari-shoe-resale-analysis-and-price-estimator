# Mercari Women’s Shoe Condition & Price Resilience Analysis

## Project Overview

This project analyzes women’s shoe listings from the Mercari Price Suggestion Challenge dataset to understand how listing prices change across brands, shoe types, and item conditions.

The project has two main parts:

1. **Condition-retention analysis**  
   Measures how each brand’s own median listing price changes as shoe condition worsens within the same shoe type.

2. **Machine-learning price estimator**  
   Predicts an expected Mercari listing price using brand, shoe type, item condition, listing title, item description, and engineered text features.

The analysis focuses on seven women’s shoe categories and uses more than 53,000 cleaned listings.

The current best prediction model is a character-enhanced, balanced weighted Ridge Regression model with:

- **Mean Absolute Error:** $12.15
- **Root Mean Squared Error:** $24.28
- **R²:** 0.640

The estimator performs best for typical listings under $50, while rare and premium listings remain more difficult to predict.

The long-term goal is to build a Kelley Blue Book–style website where users can enter shoe details and receive:

- An estimated Mercari listing price
- An adaptive price range
- A confidence or reliability indicator

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

Charts are saved in:

```text
outputs/charts/
```

## Project Structure

```text
mercari-shoe-resale-analysis/
├── data/
│   ├── raw/
│   │   └── train.tsv
│   └── processed/
│       ├── clean_womens_shoes.parquet
│       └── brand_shoe_condition_retention.parquet
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_shoe_data_cleaning.ipynb
│   ├── 03_exploratory_brand_tiers.ipynb
│   ├── 04_condition_retention_analysis.ipynb
│   └── 05_condition_visualizations.ipynb
├── outputs/
│   ├── charts/
│   └── tables/
│       └── brand_shoe_condition_retention.csv
├── .gitignore
├── requirements.txt
└── README.md
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

## Findings

Approximately **92.6%** of the analyzed brand and shoe-type combinations showed an expected decline pattern, while **7.4%** showed a non-monotonic pattern.

These results should be interpreted cautiously because the dataset does not identify identical shoe models across condition groups.

## Price Estimation Model

In addition to the condition-retention analysis, this project now includes a machine-learning model that estimates the expected Mercari listing price of a women’s shoe.

The estimator uses the following inputs:

- Brand
- Shoe type
- Item condition
- Listing title
- Item description
- Premium keyword indicators
- Title length
- Description length

Listing titles and descriptions are converted into numerical features using both word-level and character-level TF-IDF. Character-level features help the model recognize model numbers, abbreviations, misspellings, and variations such as `Air Max`, `AirMax`, and `air-max`.

Because listing prices are strongly right-skewed, the model predicts `log(price + 1)` and converts predictions back into dollar values.

### Models Tested

The following models were evaluated:

- Median-price baseline
- Linear Regression
- Ridge Regression
- Weighted Ridge Regression
- XGBoost Regression

The median baseline predicted the same median price for every listing. The trained models were compared using Mean Absolute Error, Root Mean Squared Error, and R².

### Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Median Baseline | $21.75 | $41.95 | -0.073 |
| Linear Regression | $14.83 | $30.03 | 0.450 |
| Ridge Regression | $14.85 | $30.25 | 0.442 |
| Title-Enhanced Ridge | $12.65 | $26.53 | 0.571 |
| Description-Enhanced Ridge | $12.38 | $25.73 | 0.596 |
| Tuned Ridge | $12.22 | $25.69 | 0.597 |
| Premium-Feature Ridge | $12.16 | $25.46 | 0.604 |
| Balanced Weighted Ridge | $12.19 | $24.59 | 0.631 |
| XGBoost | $13.07 | $27.24 | 0.547 |
| Character-Enhanced Weighted Ridge | **$12.15** | **$24.28** | **0.640** |

The current best-performing model is the **Character-Enhanced Balanced Weighted Ridge Regression model**.

### Current Best Model

The final model combines:

- One-hot encoded brand and shoe type
- Item condition
- Word-level TF-IDF features from listing titles
- Character-level TF-IDF features from listing titles
- TF-IDF features from item descriptions
- Premium keyword indicators
- Title and description length
- Balanced sample weighting for higher-priced listings
- Ridge Regression with `alpha = 2.0`

The model achieved:

- **Mean Absolute Error:** $12.15
- **Root Mean Squared Error:** $24.28
- **R²:** 0.640

This means the model explains approximately 64% of the variation in listing prices and misses the actual listing price by about $12.15 on average.

### Performance by Price Range

| Actual Listing Price | Number of Test Listings | MAE |
|---|---:|---:|
| $0–$25 | 4,341 | $6.80 |
| $25–$50 | 3,833 | $9.03 |
| $50–$100 | 1,880 | $17.26 |
| $100+ | 582 | $56.06 |

The estimator performs best for typical Mercari listings under $50. Rare and expensive listings remain more difficult because they appear less frequently and often depend on details not fully captured in the dataset, such as exact model rarity, authenticity, release year, or original retail price.

## Limitations

- The Mercari dataset contains listing prices rather than confirmed sale prices.
- Expensive and rare shoes are much harder to predict than typical listings.
- The dataset does not include original retail price, release year, authenticity verification, exact shoe model, size, or seller reputation.
- Item condition is reported by the seller and may be inconsistent.
- The model should be interpreted as a pricing estimate rather than a guaranteed resale value.

## Next Steps

- Improve detection of premium and rare listings
- Create adaptive prediction ranges based on model uncertainty
- Retrain the final model on the full cleaned dataset
- Save the trained preprocessing and prediction pipeline
- Build a Streamlit website where users can enter shoe information and receive an estimated listing price
- Add confidence or reliability warnings for high-priced and low-data listings

## Tools Used

Python, pandas, matplotlib, Jupyter Notebook, VS Code, Git, GitHub, and Parquet.
