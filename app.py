from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from textwrap import dedent


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cherry+Bomb+One&family=Nunito:wght@400;600;700;800&display=swap');
    /* Main page background */
    .stApp {
        background:
            linear-gradient(
                180deg,
                #eaf8ff 0%,
                #f7fcff 45%,
                #ffffff 100%
            );
    }

    /* Main content width and spacing */
    .block-container {
        max-width: 850px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    /* General text */
    html, body, [class*="css"] {
        color: #12345b;
    }

    /* Hero section */
    .hero-card {
        background: linear-gradient(
            135deg,
            #c9edff,
            #eef9ff
        );
        border: 1px solid #a9ddf7;
        border-radius: 24px;
        padding: 30px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(20, 70, 120, 0.10);
    }

   html, body, [class*="css"] {
    font-family: "Avenir Next", "Trebuchet MS", sans-serif;
    color: #12345b;
}

.hero-title {
        font-family: "Cherry Bomb One", cursive;
        color: #0b2d5c;
        font-size: 3rem;
        font-weight: 400;
        letter-spacing: 0.02em;
        line-height: 1.05;
        margin-bottom: 14px;
}

.hero-subtitle {
    font-family: "Nunito", sans-serif;
    color: #1d5685;
    font-size: 1.15rem;
    font-weight: 700;
    margin-top: 14px;
}

.hero-description {
    font-family: "Nunito", sans-serif;
    color: #527694;
    font-size: 0.95rem;
    font-weight: 500;
    line-height: 1.5;
    margin: 6px auto 0;
    max-width: 560px;
}

    /* Form card */
    div[data-testid="stForm"] {
        background-color: rgba(255, 255, 255, 0.94);
        border: 1px solid #b9e4f7;
        border-radius: 22px;
        padding: 25px;
        box-shadow: 0 8px 25px rgba(20, 70, 120, 0.08);
    }

    /* Field labels */
    label,
    div[data-testid="stWidgetLabel"] p {
        color: #0b2d5c !important;
        font-weight: 700 !important;
    }

    /* Text fields and dropdowns */
    input,
    textarea,
    div[data-baseweb="select"] > div {
        background-color: #f7fcff !important;
        border-color: #a9dcef !important;
        border-radius: 12px !important;
        color: #12345b !important;
    }

    input:focus,
    textarea:focus {
        border-color: #4fa6d8 !important;
        box-shadow: 0 0 0 2px rgba(79, 166, 216, 0.18) !important;
    }

    /* Estimate button */
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(
            135deg,
            #163f73,
            #275f9e
        );
        color: white;
        border: none;
        border-radius: 14px;
        min-height: 48px;
        font-size: 1rem;
        font-weight: 750;
        box-shadow: 0 6px 16px rgba(11, 45, 92, 0.22);
        transition: all 0.2s ease;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(
            135deg,
            #0b2d5c,
            #1b4f87
        );
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(11, 45, 92, 0.28);
    }

    /* Result metric */
    div[data-testid="stMetric"] {
        background: linear-gradient(
            135deg,
            #d9f2ff,
            #ffffff
        );
        border: 1px solid #a9dcef;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 6px 18px rgba(20, 70, 120, 0.08);
    }

    div[data-testid="stMetricLabel"] {
        color: #315b83;
    }

    div[data-testid="stMetricValue"] {
        color: #0b2d5c;
        font-weight: 800;
    }

    /* Alert boxes */
    div[data-testid="stAlert"] {
        border-radius: 16px;
    }

    /* Expander */
    details {
        background-color: rgba(255, 255, 255, 0.92);
        border: 1px solid #b9e4f7 !important;
        border-radius: 16px !important;
        padding: 4px 12px;
    }

    /* Divider */
    hr {
        border-color: #cceafa !important;
    }

    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }
    
    .range-card {
    background: linear-gradient(
        135deg,
        #d9f2ff 0%,
        #ffffff 100%
    );
    border: 2px solid #8fcfee;
    border-radius: 22px;
    padding: 30px 24px;
    margin-top: 12px;
    margin-bottom: 18px;
    text-align: center;
    box-shadow: 0 10px 28px rgba(21, 73, 120, 0.12);
}

.range-label {
    color: #315b83;
    font-family: "Nunito", sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.range-value {
    color: #0b2d5c;
    font-family: "Cherry Bomb One", cursive;
    font-size: 2.7rem;
    line-height: 1.15;
    margin: 0;
}

.range-note {
    color: #56738f;
    font-family: "Nunito", sans-serif;
    font-size: 0.9rem;
    margin-top: 10px;

}

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Load saved model
# ---------------------------------------------------------
PROJECT_DIRECTORY = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIRECTORY
    / "models"
    / "mercari_shoe_price_estimator.joblib"
)


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


bundle = load_model_bundle()

model = bundle["model"]
premium_keywords = bundle["premium_keywords"]
feature_columns = bundle["feature_columns"]
brand_options = bundle["brand_options"]
shoe_type_options = bundle["shoe_type_options"]
range_percentage = bundle["suggested_range_percentage"]


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------
def build_premium_features(data):
    data = data.copy()

    data["name"] = (
        data["name"]
        .fillna("")
        .astype(str)
    )

    data["item_description"] = (
        data["item_description"]
        .fillna("")
        .astype(str)
    )

    combined_text = (
        data["name"]
        + " "
        + data["item_description"]
    ).str.lower()

    for keyword in premium_keywords:
        pattern = rf"\b{re.escape(keyword.lower())}\b"

        data[f"has_{keyword}"] = (
            combined_text
            .str.contains(
                pattern,
                regex=True,
                na=False
            )
            .astype(int)
        )

    data["title_length"] = data["name"].str.len()

    data["description_length"] = (
        data["item_description"].str.len()
    )

    return data


# ---------------------------------------------------------
# App content
# ---------------------------------------------------------
st.markdown(
    '<div class="hero-card">'
    '<div class="hero-title">Mercari Shoe Price Estimator</div>'
    '<div class="hero-subtitle">Smart pricing guidance for women’s shoes on Mercari.</div>'
    '<div class="hero-description">Enter the shoe details below to receive a suggested listing range.</div>'
    '</div>',
    unsafe_allow_html=True
)

condition_options = {
    "1 — New": 1,
    "2 — Like New": 2,
    "3 — Good": 3,
    "4 — Fair": 4,
    "5 — Poor": 5
}


with st.form("shoe_estimator_form"):

    brand_name = st.selectbox(
        "Brand",
        options=brand_options
    )

    shoe_type = st.selectbox(
        "Shoe Type",
        options=shoe_type_options
    )

    selected_condition = st.selectbox(
        "Item Condition",
        options=list(condition_options.keys())
    )

    listing_name = st.text_input(
        "Listing Title (Optional)",
        placeholder="Example: Nike Air Max 270 Women's Shoes"
    )

    item_description = st.text_area(
        "Item Description",
        placeholder=(
            "Example: Gently used authentic Nike Air Max shoes "
            "in good condition."
        ),
        height=130
    )

    submitted = st.form_submit_button(
        "Estimate Price",
        use_container_width=True
    )

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    item_condition_id = condition_options[
        selected_condition
    ]

    effective_listing_name = listing_name.strip()

    if not effective_listing_name:
        effective_listing_name = (
            f"{brand_name} {shoe_type}"
        )

    listing = pd.DataFrame({
        "brand_name": [brand_name],
        "shoe_type": [shoe_type],
        "item_condition_id": [item_condition_id],
        "name": [effective_listing_name],
        "item_description": [item_description]
    })

    listing = build_premium_features(listing)

    predicted_log_price = model.predict(
        listing[feature_columns]
    )

    predicted_price = float(
        np.expm1(predicted_log_price)[0]
    )

    predicted_price = max(predicted_price, 0)

    lower_price = predicted_price * (
        1 - range_percentage
    )

    upper_price = predicted_price * (
        1 + range_percentage
    )

    st.divider()

    st.markdown(
        """
        <h2 style="
            color: #0b2d5c;
            text-align: center;
            font-family: 'Avenir Next', 'Trebuchet MS', sans-serif;
            margin-bottom: 1rem;
        ">
            Your Suggested Price Range
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
    dedent(
        f"""
        <div class="range-card">
            <div class="range-label">Suggested Listing Range</div>
            <div class="range-value">${lower_price:,.2f} – ${upper_price:,.2f}</div>
            <div class="range-note">Based on the shoe details provided</div>
        </div>
        """
    ).strip(),
    unsafe_allow_html=True
)

    if item_condition_id == 5:
        st.warning(
            "Limited Condition 5 data was available, "
            "so this estimate may be less reliable."
        )

    if not listing_name.strip():
        st.info(
            "No listing title was provided, so the estimate used "
            "the selected brand and shoe type instead."
        )

    st.caption(
        "The suggested range is set to ±15% around the model "
        "estimate for practical pricing guidance. It is not a "
        "formal confidence interval or guaranteed selling price."
    )