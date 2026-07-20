"""Streamlit UI for Ames Housing price prediction."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from src.config import ProjectPaths
from src.predict import load_model, predict_dataframe


st.set_page_config(
    page_title="Ames Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(234, 179, 8, 0.18), transparent 24%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.14), transparent 22%),
            linear-gradient(180deg, #e8efe8 0%, #eef2ea 38%, #f6efe4 100%);
        color: #122018;
    }

    .stApp header, .stApp footer {
        background: transparent;
    }

    .hero {
        padding: 1.6rem 1.7rem;
        border-radius: 24px;
        color: #f7f7f2;
        background: linear-gradient(135deg, rgba(25, 48, 38, 0.97), rgba(52, 78, 61, 0.92));
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 20px 55px rgba(25, 48, 38, 0.24);
    }

    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.75rem;
        color: #f2c94c;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .hero-title {
        font-size: 2.4rem;
        line-height: 1.05;
        margin-bottom: 0.5rem;
        font-weight: 800;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #d9e4da;
        max-width: 55rem;
        line-height: 1.55;
    }

    .metric-card {
        background: rgba(247, 243, 233, 0.82);
        border: 1px solid rgba(79, 111, 82, 0.16);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        box-shadow: 0 12px 30px rgba(25, 48, 38, 0.06);
    }

    .metric-label {
        color: #4f6f52;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
        font-weight: 700;
    }

    .metric-value {
        color: #122018;
        font-size: 1.4rem;
        font-weight: 800;
    }

    .metric-hint {
        color: #596a5c;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    .panel {
        background: rgba(247, 243, 233, 0.88);
        border: 1px solid rgba(79, 111, 82, 0.14);
        border-radius: 22px;
        padding: 1.2rem 1.25rem;
        box-shadow: 0 16px 36px rgba(25, 48, 38, 0.06);
    }

    .prediction-card {
        background: linear-gradient(135deg, #193026, #355846);
        color: #f7f7f2;
        border-radius: 22px;
        padding: 1.3rem 1.35rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 18px 45px rgba(25, 48, 38, 0.18);
    }

    .prediction-label {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #c9d8c6;
        font-size: 0.74rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .prediction-price {
        font-size: 2.4rem;
        line-height: 1;
        font-weight: 900;
        margin-bottom: 0.4rem;
    }

    .prediction-band {
        color: #d9e4da;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .tech-list li {
        margin-bottom: 0.35rem;
    }

    div[data-testid="stExpander"] {
        background: rgba(247, 243, 233, 0.86);
        border: 1px solid rgba(79, 111, 82, 0.16);
        border-radius: 18px;
    }

    [data-testid="stSidebar"] {
        background: rgba(232, 239, 232, 0.96);
        border-right: 1px solid rgba(79, 111, 82, 0.12);
    }

    .stButton > button {
        background: linear-gradient(135deg, #5f8d72, #3f6b57);
        color: #f7f7f2;
        border: none;
        border-radius: 14px;
        padding: 0.7rem 1rem;
        font-weight: 700;
        box-shadow: 0 10px 24px rgba(63, 107, 87, 0.22);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #6f9d81, #4b7c64);
        color: #ffffff;
    }
</style>
"""


SECTION_ORDER = [
    "Location and Site",
    "Quality and Structure",
    "Basement and Garage",
    "Interior and Living Area",
    "Exterior and Amenities",
    "Other",
]


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    """Load the Kaggle training set for defaults and schema inference."""

    return pd.read_csv(ProjectPaths.train_csv)


@st.cache_resource
def load_trained_bundle():
    """Load the fitted model bundle once per session."""

    return load_model(ProjectPaths.final_model_path)


def infer_section(column: str) -> str:
    """Group columns into human-readable input sections."""

    lower = column.lower()
    if any(
        token in lower
        for token in [
            "lot",
            "neighborhood",
            "mssubclass",
            "mszoning",
            "street",
            "alley",
            "land",
            "condition1",
            "condition2",
            "lotconfig",
            "bldgtype",
            "housestyle",
            "mosold",
            "yrsold",
            "saletype",
            "salecondition",
            "utility",
        ]
    ):
        return "Location and Site"
    if any(token in lower for token in ["year", "qual", "cond", "heating", "centralair", "electrical", "functional", "roof", "foundation", "exter", "kitchenqual"]):
        return "Quality and Structure"
    if any(token in lower for token in ["bsmt", "garage"]):
        return "Basement and Garage"
    if any(token in lower for token in ["flr", "bath", "bedroom", "kitchen", "fireplace", "grlivarea", "totrms"]):
        return "Interior and Living Area"
    if any(token in lower for token in ["porch", "deck", "pool", "fence", "misc", "masvnr", "paveddrive"]):
        return "Exterior and Amenities"
    return "Other"


def build_default_value(series: pd.Series):
    """Build a sensible default value from the training distribution."""

    if pd.api.types.is_numeric_dtype(series):
        value = series.dropna().median()
        if pd.isna(value):
            return 0
        if float(value).is_integer():
            return int(value)
        return float(value)

    mode_series = series.dropna().astype(str).mode()
    if not mode_series.empty:
        return mode_series.iloc[0]
    return "None"


def render_numeric_input(column: str, default_value, min_value=None, max_value=None):
    """Render a numeric input with a compact default."""

    if isinstance(default_value, int):
        return st.number_input(
            column,
            value=int(default_value),
            step=1,
            min_value=min_value,
            max_value=max_value,
        )

    return st.number_input(
        column,
        value=float(default_value),
        step=0.1,
        min_value=min_value,
        max_value=max_value,
        format="%.2f",
    )


def render_categorical_input(column: str, series: pd.Series):
    """Render a selectbox for categorical values."""

    options = sorted({str(value) for value in series.dropna().astype(str).unique().tolist()})
    if "None" not in options:
        options = ["None"] + options
    default_value = build_default_value(series)
    index = options.index(default_value) if default_value in options else 0
    return st.selectbox(column, options=options, index=index)


def build_section_map(feature_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Map raw columns to UI sections."""

    sections: Dict[str, List[str]] = {}
    for column in feature_df.columns:
        sections.setdefault(infer_section(column), []).append(column)
    return sections


def render_header(bundle) -> None:
    """Render the hero section and top-level summary cards."""

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Ames Housing • Machine Learning Demo</div>
            <div class="hero-title">Predict a home price with a polished, production-style experience.</div>
            <div class="hero-subtitle">
                This interface uses a CPU-friendly tree-based model trained on the Kaggle Ames Housing dataset.
                The preprocessing pipeline handles missingness, rare categories, and feature engineering behind the scenes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Training rows", f"{len(load_reference_data()):,}", "Kaggle Ames housing records"),
        ("Best model", bundle.best_model_name, "Selected by cross-validation"),
        ("CV RMSE (log)", f"{bundle.cv_rmse_log:.4f}", "Lower is better"),
        ("Feature set", str(len(bundle.feature_names) or "Auto"), "Engineered + encoded features"),
    ]
    for container, (label, value, hint) in zip([c1, c2, c3, c4], cards):
        with container:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-hint">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def collect_user_inputs(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Collect a full single-row Ames feature frame from the main form."""

    feature_df = reference_df.drop(columns=["SalePrice"], errors="ignore").drop(columns=["Id"], errors="ignore")
    sections = build_section_map(feature_df)
    values: Dict[str, object] = {}

    with st.form("prediction_form", clear_on_submit=False):
        st.markdown(
            """
            <div class="panel">
                <strong>Describe the property</strong><br/>
                Fill out the sections below. The defaults are seeded from the training data so the form starts from a realistic home.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        for section_name in SECTION_ORDER:
            columns = sections.get(section_name, [])
            if not columns:
                continue

            with st.expander(section_name, expanded=(section_name in {"Location and Site", "Quality and Structure"})):
                st.caption("Grouped together to keep the form readable and fast to scan.")
                for idx in range(0, len(columns), 2):
                    left_col, right_col = st.columns(2)
                    for offset, target_col in enumerate(columns[idx : idx + 2]):
                        container = left_col if offset == 0 else right_col
                        with container:
                            series = reference_df[target_col]
                            label = target_col.replace("_", " ")
                            if pd.api.types.is_numeric_dtype(series):
                                default_value = build_default_value(series)
                                values[target_col] = render_numeric_input(label, default_value)
                            else:
                                values[target_col] = render_categorical_input(label, series)

        submitted = st.form_submit_button("Predict Sale Price", use_container_width=True, type="primary")

    if submitted:
        return pd.DataFrame([values])
    return pd.DataFrame()


def render_prediction(bundle, user_df: pd.DataFrame) -> None:
    """Render the prediction result card and an approximate uncertainty band."""

    prediction = float(predict_dataframe(bundle, user_df)[0])
    spread = float(np.exp(bundle.cv_rmse_log))
    lower = prediction / spread
    upper = prediction * spread

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">Estimated sale price</div>
            <div class="prediction-price">${prediction:,.0f}</div>
            <div class="prediction-band">
                Approximate CV-based range: <strong>${lower:,.0f}</strong> to <strong>${upper:,.0f}</strong><br/>
                This band is derived from the model's log-scale cross-validation error, so treat it as directional rather than exact.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Point estimate", f"${prediction:,.0f}")
    with c2:
        st.metric("Lower band", f"${lower:,.0f}")
    with c3:
        st.metric("Upper band", f"${upper:,.0f}")


def render_sidebar(bundle) -> None:
    """Render supportive context in the sidebar."""

    with st.sidebar:
        st.markdown("## Model Notes")
        st.write(f"Best model: `{bundle.best_model_name}`")
        st.write("Target was trained on `log1p(SalePrice)` and converted back to dollars at inference.")
        st.write("Missing data is handled using Ames-specific rules, not one-size-fits-all imputation.")

        st.markdown("## Why this UI works")
        st.write("The form is organized into domain groups so it feels like a guided estimate rather than a giant spreadsheet.")
        st.write("Defaults are pulled from the training data to keep inputs realistic.")

        st.markdown("## Techniques Used")
        st.markdown(
            """
            <ul class="tech-list">
                <li>Domain-aware missing value handling</li>
                <li>Rare category grouping</li>
                <li>Feature engineering for area, age, and quality</li>
                <li>XGBoost / LightGBM cross-validation</li>
                <li>Joblib serialization for the trained pipeline</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    """Render the interactive prediction app."""

    st.markdown(APP_CSS, unsafe_allow_html=True)
    reference_df = load_reference_data()
    bundle = load_trained_bundle()
    render_sidebar(bundle)
    render_header(bundle)

    st.write("")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    user_df = collect_user_inputs(reference_df)
    st.markdown("</div>", unsafe_allow_html=True)

    if not user_df.empty:
        st.write("")
        st.markdown("### Prediction Result")
        render_prediction(bundle, user_df)
        st.info("This prediction is a model estimate, not a professional appraisal.")


if __name__ == "__main__":
    main()
