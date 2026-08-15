"""PulseIQ dashboard.

Consumes the FastAPI service over HTTP. It holds no model and no database
connection -- if this file starts importing from `pulseiq.training`, the
separation has been lost.

Run:
    uvicorn pulseiq.api.main:app --reload      # terminal 1
    streamlit run app/streamlit_app.py         # terminal 2
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import APIClient  # noqa: E402

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="PulseIQ", page_icon="📈", layout="wide")


@st.cache_resource
def get_client() -> APIClient:
    return APIClient(API_BASE_URL)


client = get_client()


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.title("PulseIQ")
    st.caption("Competitor pricing & sentiment")

    health, error = client.health()
    if error or health is None:
        st.error("API unreachable")
        st.code(error or "No response from API", language=None)
        st.stop()

    status = health["status"]
    st.success("All systems OK") if status == "ok" else st.warning(f"Status: {status}")

    with st.expander("Components", expanded=(status != "ok")):
        for component in health["components"]:
            icon = "✅" if component["status"] == "ok" else "⚠️"
            st.write(
                f"{icon} **{component['name']}** — {component['detail'] or component['status']}"
            )

    page = st.radio(
        "View", ["Forecast", "Sentiment", "Recommendation"], label_visibility="collapsed"
    )


# --- forecast ---------------------------------------------------------------

if page == "Forecast":
    st.header("Price forecast")

    products, error = client.list_products()
    if error or products is None:
        st.error(error or "No response from API")
        st.stop()
    if not products:
        st.info(
            "No products with enough history yet. Ingest some:\n\n"
            "`python -m pulseiq.ingestion.run_ingest --open-prices --max-series 200`"
        )
        st.stop()

    models, _ = client.list_models()
    models = models or {"naive_last": ""}

    left, middle, right = st.columns([3, 1, 1])
    with left:
        options = {
            f"{p['product_name']}  ({p['n_observations']} obs)": p["product_name"] for p in products
        }
        selected = st.selectbox("Product", list(options))
        product_name = options[selected]
    with middle:
        horizon = st.slider("Months ahead", 1, 12, 3)
    with right:
        model = st.selectbox("Model", list(models))

    st.caption(models.get(model, ""))

    result, error = client.forecast(product_name, horizon=horizon, model=model)
    if error or result is None:
        st.error(error or "No forecast returned")
        st.stop()

    a, b, c, d = st.columns(4)
    a.metric("Last observed", f"{result['last_observed_price']:.2f}")
    b.metric("Observations", result["n_observations"])
    predicted = result["forecast"][-1]["predicted_price"]
    delta = predicted - result["last_observed_price"]
    c.metric(f"Forecast (+{horizon}m)", f"{predicted:.2f}", f"{delta:+.2f}")
    d.metric("Served from cache", "yes" if result["cached"] else "no")

    # Plot the forecast against the series it extends. A prediction shown on its
    # own cannot be judged plausible or not -- the earlier version drew a single
    # anchor point, which made every model look identical.
    history, history_error = client.price_history(product_name, limit=12)

    observed = (
        {-len(history) + 1 + i: price for i, price in enumerate(history)}
        if history
        else {0: result["last_observed_price"]}
    )
    forecast_points = {p["period"]: p["predicted_price"] for p in result["forecast"]}
    # Anchor the forecast at period 0 so the two lines join rather than float apart.
    forecast_points[0] = result["last_observed_price"]

    chart_data = pd.DataFrame({"observed": observed, "forecast": forecast_points}).sort_index()
    st.line_chart(chart_data)
    st.caption("Period 0 is the last observed month; positive periods are forecast.")

    if history_error:
        st.caption(f"(History unavailable: {history_error})")

    # The finding this project measured, shown where a user would otherwise
    # over-trust the number above.
    st.warning(result["baseline_note"])


# --- sentiment --------------------------------------------------------------

elif page == "Sentiment":
    st.header("Review sentiment")
    st.caption("DistilBERT fine-tuned with LoRA — 94.1% accuracy on a held-out test set.")

    default = (
        "Battery died within a week, complete waste of money.\n"
        "Sound quality is incredible and the noise cancellation is superb.\n"
        "It works fine I suppose, nothing special about it."
    )
    raw = st.text_area("One review per line", value=default, height=160)
    texts = [line.strip() for line in raw.splitlines() if line.strip()]

    if st.button("Classify", type="primary") and texts:
        result, error = client.sentiment(texts)
        if error or result is None:
            st.error(error or "No response from API")
            if error and "unavailable" in error.lower():
                st.info(
                    "The fine-tuned adapter is missing. Run "
                    "`notebooks/finetune_sentiment.ipynb` and unzip the result into "
                    "`models/sentiment_lora/`."
                )
            st.stop()

        for prediction in result["predictions"]:
            confidence = prediction["confidence"]
            icon = "🟢" if prediction["label"] == "positive" else "🔴"
            if confidence < 0.7:
                icon = "🟡"
            with st.container(border=True):
                st.write(f"{icon} **{prediction['label']}** — confidence {confidence:.1%}")
                st.caption(prediction["text"])
                st.progress(prediction["positive"], text=f"positive {prediction['positive']:.1%}")

        st.info(
            "Confidence near 50% is meaningful rather than a defect: 3-star reviews "
            "were excluded from training, so genuinely mixed text has no class to "
            "belong to and the model signals that honestly."
        )


# --- recommendation ---------------------------------------------------------

else:
    st.header("Pricing recommendation")

    products, error = client.list_products()
    if error or products is None:
        st.error(error or "No response from API")
        st.stop()
    if not products:
        st.info("No products available. Ingest price data first.")
        st.stop()

    product_name = st.selectbox("Product", [p["product_name"] for p in products])
    if product_name is None:
        st.stop()

    context = st.text_area(
        "Additional context (optional)", placeholder="e.g. competitor launched a rival model"
    )

    if st.button("Generate", type="primary"):
        with st.spinner("Asking the model..."):
            result, error = client.recommend(product_name, context=context or None)
        if error or result is None:
            st.error(error or "No response from API")
            if error and ("503" in error or "provider" in error.lower()):
                st.info("Set GROQ_API_KEY in `.env` to enable recommendations.")
            st.stop()

        st.markdown(result["recommendation"])
        st.caption(f"Generated by {result['provider']}" + (" (cached)" if result["cached"] else ""))
