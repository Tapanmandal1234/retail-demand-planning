import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Retail Demand Planning | Tapan Mandal",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN / CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #061523;
        color: #eee9df;
    }

    [data-testid="stSidebar"] {
        background-color: #091b2b;
        border-right: 1px solid rgba(255,255,255,0.12);
    }

    h1, h2, h3 {
        font-family: Georgia, serif !important;
        color: #f5f2ea !important;
    }

    p, div, span, label {
        color: #eee9df;
    }

    .eyebrow {
        font-size: 11px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #9ba9b4;
        margin-bottom: 8px;
    }

    .intro {
        font-size: 18px;
        line-height: 1.6;
        max-width: 900px;
        color: #c9d0d5;
        margin-bottom: 35px;
    }

    .section-number {
        font-size: 10px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #8295a5;
        margin-top: 35px;
        margin-bottom: 5px;
    }

    .insight-box {
        border-left: 2px solid #d6a08d;
        background-color: rgba(255,255,255,0.035);
        padding: 18px 22px;
        margin-top: 15px;
        margin-bottom: 25px;
    }

    [data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.12);
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    hr {
        border-color: rgba(255,255,255,0.12);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="eyebrow">01 / Retail Strategy & Forecasting</div>',
    unsafe_allow_html=True,
)

st.title("Retail Demand Planning")

st.markdown(
    """
    <div class="intro">
    <b>What should we plan for next?</b><br>
    Using historical product and store performance to understand demand,
    identify momentum and risk, forecast future demand, and translate
    the analysis into commercial planning decisions.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(file_source):
    df = pd.read_csv(file_source)

    required_columns = {"date", "store", "item", "sales"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "The dataset must contain: date, store, item, sales"
        )

    df["date"] = pd.to_datetime(df["date"])

    df["store"] = df["store"].astype(int)
    df["item"] = df["item"].astype(int)
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    df = df.dropna(subset=["date", "store", "item", "sales"])

    return df


# Try GitHub-hosted train.csv first.
# If it is not present, allow manual upload.

DATA_PATH = "train.csv"

if os.path.exists(DATA_PATH):
    source = DATA_PATH
else:
    st.warning(
        "I can't find train.csv in the repository. "
        "Upload the Kaggle train.csv file below."
    )

    source = st.file_uploader(
        "Upload train.csv",
        type=["csv"],
    )

    if source is None:
        st.stop()


with st.spinner("Loading the retail demand dataset..."):
    df = load_data(source)


# ============================================================
# BASIC DATA INFO
# ============================================================

min_date = df["date"].min()
max_date = df["date"].max()

stores = sorted(df["store"].unique())
items = sorted(df["item"].unique())


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## Planning Controls")

selected_store = st.sidebar.selectbox(
    "Store",
    ["All Stores"] + stores,
)

selected_item = st.sidebar.selectbox(
    "Item for detailed analysis",
    items,
)

st.sidebar.markdown("---")

st.sidebar.caption(
    f"Dataset period: "
    f"{min_date.strftime('%b %Y')} – "
    f"{max_date.strftime('%b %Y')}"
)

st.sidebar.caption(
    f"{len(stores)} stores · {len(items)} items"
)


# ============================================================
# FILTER DATA
# ============================================================

if selected_store == "All Stores":
    analysis_df = df.copy()
else:
    analysis_df = df[df["store"] == selected_store].copy()


# ============================================================
# OVERALL METRICS
# ============================================================

st.markdown(
    '<div class="section-number">01 / Demand Landscape</div>',
    unsafe_allow_html=True,
)

st.header("Five years of demand")

total_sales = analysis_df["sales"].sum()
avg_daily_sales = (
    analysis_df.groupby("date")["sales"].sum().mean()
)

recent_90_start = max_date - pd.Timedelta(days=89)
previous_90_start = recent_90_start - pd.Timedelta(days=90)

recent_sales = analysis_df.loc[
    analysis_df["date"] >= recent_90_start,
    "sales",
].sum()

previous_sales = analysis_df.loc[
    (analysis_df["date"] >= previous_90_start)
    & (analysis_df["date"] < recent_90_start),
    "sales",
].sum()

momentum = (
    (recent_sales - previous_sales) / previous_sales
    if previous_sales > 0
    else np.nan
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Historical Demand",
    f"{total_sales:,.0f}",
)

m2.metric(
    "Average Daily Demand",
    f"{avg_daily_sales:,.0f}",
)

m3.metric(
    "Items",
    f"{analysis_df['item'].nunique():,}",
)

m4.metric(
    "Recent 90-Day Momentum",
    "N/A" if pd.isna(momentum) else f"{momentum:+.1%}",
)


# ============================================================
# MONTHLY DEMAND TREND
# ============================================================

monthly = (
    analysis_df
    .set_index("date")
    .resample("MS")["sales"]
    .sum()
    .reset_index()
)

fig_monthly = px.line(
    monthly,
    x="date",
    y="sales",
    labels={
        "date": "",
        "sales": "Units sold",
    },
)

fig_monthly.update_traces(
    line=dict(width=2),
)

fig_monthly.update_layout(
    template="plotly_dark",
    paper_bgcolor="#061523",
    plot_bgcolor="#061523",
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
    showlegend=False,
)

st.plotly_chart(fig_monthly, use_container_width=True)


# ============================================================
# SEASONALITY
# ============================================================

st.markdown(
    '<div class="section-number">02 / Seasonality</div>',
    unsafe_allow_html=True,
)

st.header("When does demand peak?")

seasonality = (
    analysis_df
    .assign(month=analysis_df["date"].dt.month)
    .groupby("month")["sales"]
    .mean()
    .reset_index()
)

month_names = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

seasonality["Month"] = seasonality["month"].map(month_names)

fig_season = px.bar(
    seasonality,
    x="Month",
    y="sales",
    labels={
        "sales": "Average daily item-store demand",
    },
)

fig_season.update_layout(
    template="plotly_dark",
    paper_bgcolor="#061523",
    plot_bgcolor="#061523",
    height=380,
    margin=dict(l=10, r=10, t=30, b=10),
    showlegend=False,
)

st.plotly_chart(fig_season, use_container_width=True)

peak_month = seasonality.loc[
    seasonality["sales"].idxmax(),
    "Month",
]

low_month = seasonality.loc[
    seasonality["sales"].idxmin(),
    "Month",
]

st.markdown(
    f"""
    <div class="insight-box">
    <b>Demand signal:</b> {peak_month} has the highest average demand
    in the historical dataset, while {low_month} has the lowest.
    This seasonality matters when interpreting recent momentum and
    future planning requirements.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PRODUCT PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-number">03 / Product Performance</div>',
    unsafe_allow_html=True,
)

st.header("Which items drive demand?")


@st.cache_data(show_spinner=False)
def build_item_summary(data):

    max_d = data["date"].max()

    recent_start = max_d - pd.Timedelta(days=89)
    previous_start = recent_start - pd.Timedelta(days=90)

    total = (
        data.groupby("item")["sales"]
        .sum()
        .rename("Total Demand")
    )

    daily_item = (
        data.groupby(["date", "item"])["sales"]
        .sum()
        .reset_index()
    )

    volatility = (
        daily_item.groupby("item")["sales"]
        .agg(["mean", "std"])
    )

    volatility["Volatility"] = (
        volatility["std"]
        / volatility["mean"].replace(0, np.nan)
    )

    recent = (
        data[data["date"] >= recent_start]
        .groupby("item")["sales"]
        .sum()
        .rename("Recent Demand")
    )

    previous = (
        data[
            (data["date"] >= previous_start)
            & (data["date"] < recent_start)
        ]
        .groupby("item")["sales"]
        .sum()
        .rename("Previous Demand")
    )

    summary = pd.concat(
        [
            total,
            recent,
            previous,
            volatility["Volatility"],
        ],
        axis=1,
    ).fillna(0)

    summary["Growth Rate"] = np.where(
        summary["Previous Demand"] > 0,
        (
            summary["Recent Demand"]
            - summary["Previous Demand"]
        )
        / summary["Previous Demand"],
        np.nan,
    )

    summary["Demand Share"] = (
        summary["Total Demand"]
        / summary["Total Demand"].sum()
    )

    summary = summary.reset_index()

    return summary


item_summary = build_item_summary(analysis_df)

top_items = (
    item_summary
    .nlargest(10, "Total Demand")
    .sort_values("Total Demand")
)

fig_products = px.bar(
    top_items,
    x="Total Demand",
    y=top_items["item"].astype(str),
    orientation="h",
    labels={
        "y": "Item",
    },
)

fig_products.update_layout(
    template="plotly_dark",
    paper_bgcolor="#061523",
    plot_bgcolor="#061523",
    height=450,
    margin=dict(l=10, r=10, t=30, b=10),
    showlegend=False,
)

st.plotly_chart(fig_products, use_container_width=True)


# ============================================================
# STORE × ITEM HEATMAP
# ============================================================

st.markdown(
    '<div class="section-number">04 / Store Differences</div>',
    unsafe_allow_html=True,
)

st.header("Does the same assortment perform everywhere?")

store_item = (
    df.groupby(["store", "item"])["sales"]
    .sum()
    .reset_index()
)

top_15_items = (
    df.groupby("item")["sales"]
    .sum()
    .nlargest(15)
    .index
)

heatmap_data = store_item[
    store_item["item"].isin(top_15_items)
]

heatmap_pivot = heatmap_data.pivot(
    index="store",
    columns="item",
    values="sales",
)

fig_heatmap = px.imshow(
    heatmap_pivot,
    aspect="auto",
    labels=dict(
        x="Item",
        y="Store",
        color="Historical demand",
    ),
)

fig_heatmap.update_layout(
    template="plotly_dark",
    paper_bgcolor="#061523",
    plot_bgcolor="#061523",
    height=430,
    margin=dict(l=10, r=10, t=30, b=10),
)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown(
    """
    <div class="insight-box">
    <b>Why this matters:</b> differences across stores indicate whether
    a single planning approach is appropriate or whether demand patterns
    vary enough to justify store-level decisions.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FORECASTING
# ============================================================

st.markdown(
    '<div class="section-number">05 / Forecasting</div>',
    unsafe_allow_html=True,
)

st.header(f"Can we forecast Item {selected_item}?")


@st.cache_data(show_spinner=False)
def prepare_weekly_series(data, item):

    item_data = data[data["item"] == item].copy()

    weekly = (
        item_data
        .set_index("date")
        .resample("W")["sales"]
        .sum()
        .asfreq("W", fill_value=0)
    )

    return weekly


weekly = prepare_weekly_series(
    analysis_df,
    selected_item,
)


def calculate_mape(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(
        np.abs(
            (actual[mask] - predicted[mask])
            / actual[mask]
        )
    ) * 100


def calculate_mae(actual, predicted):
    return np.mean(
        np.abs(
            np.asarray(actual)
            - np.asarray(predicted)
        )
    )


TEST_WEEKS = 13

if len(weekly) < 120:
    st.warning(
        "There is not enough weekly history for reliable "
        "seasonal forecasting for this selection."
    )

else:

    train = weekly.iloc[:-TEST_WEEKS]
    test = weekly.iloc[-TEST_WEEKS:]

    # --------------------------------------------------------
    # BASELINE MODEL
    # Seasonal naive:
    # same weeks from previous year
    # --------------------------------------------------------

    baseline = weekly.shift(52).loc[test.index]

    if baseline.isna().any():
        baseline = pd.Series(
            [train.iloc[-1]] * len(test),
            index=test.index,
        )

    baseline_mae = calculate_mae(
        test.values,
        baseline.values,
    )

    baseline_mape = calculate_mape(
        test.values,
        baseline.values,
    )


    # --------------------------------------------------------
    # STRONGER MODEL
    # Holt-Winters Exponential Smoothing
    # --------------------------------------------------------

    try:

        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=52,
            initialization_method="estimated",
        ).fit(
            optimized=True,
            use_brute=False,
        )

        model_prediction = model.forecast(TEST_WEEKS)

        model_mae = calculate_mae(
            test.values,
            model_prediction.values,
        )

        model_mape = calculate_mape(
            test.values,
            model_prediction.values,
        )

        model_success = True

    except Exception:

        model_prediction = baseline.copy()
        model_mae = baseline_mae
        model_mape = baseline_mape
        model_success = False


    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Baseline MAE",
        f"{baseline_mae:,.0f}",
    )

    c2.metric(
        "Baseline MAPE",
        f"{baseline_mape:.1f}%",
    )

    c3.metric(
        "Holt-Winters MAE",
        f"{model_mae:,.0f}",
    )

    c4.metric(
        "Holt-Winters MAPE",
        f"{model_mape:.1f}%",
    )


    # --------------------------------------------------------
    # ACTUAL VS FORECAST
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        {
            "Date": test.index,
            "Actual": test.values,
            "Seasonal Naive": baseline.values,
            "Holt-Winters": model_prediction.values,
        }
    )

    fig_test = go.Figure()

    fig_test.add_trace(
        go.Scatter(
            x=comparison["Date"],
            y=comparison["Actual"],
            mode="lines+markers",
            name="Actual",
        )
    )

    fig_test.add_trace(
        go.Scatter(
            x=comparison["Date"],
            y=comparison["Seasonal Naive"],
            mode="lines",
            name="Seasonal Naive",
        )
    )

    fig_test.add_trace(
        go.Scatter(
            x=comparison["Date"],
            y=comparison["Holt-Winters"],
            mode="lines",
            name="Holt-Winters",
        )
    )

    fig_test.update_layout(
        template="plotly_dark",
        paper_bgcolor="#061523",
        plot_bgcolor="#061523",
        height=430,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Weekly demand",
        xaxis_title="",
    )

    st.plotly_chart(
        fig_test,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # CHOOSE BEST MODEL
    # --------------------------------------------------------

    if model_mae < baseline_mae:
        best_model_name = "Holt-Winters"
    else:
        best_model_name = "Seasonal Naive"

    st.markdown(
        f"""
        <div class="insight-box">
        <b>Validation result:</b> {best_model_name} produced the lower
        holdout MAE for Item {selected_item}. The comparison uses the
        final {TEST_WEEKS} weeks as an unseen chronological test period,
        rather than randomly splitting time-series observations.
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # FUTURE 13-WEEK FORECAST
    # --------------------------------------------------------

    FORECAST_WEEKS = 13

    if best_model_name == "Holt-Winters" and model_success:

        final_model = ExponentialSmoothing(
            weekly,
            trend="add",
            seasonal="add",
            seasonal_periods=52,
            initialization_method="estimated",
        ).fit(
            optimized=True,
            use_brute=False,
        )

        future_forecast = final_model.forecast(
            FORECAST_WEEKS
        )

    else:

        future_dates = pd.date_range(
            start=weekly.index[-1]
            + pd.Timedelta(weeks=1),
            periods=FORECAST_WEEKS,
            freq="W",
        )

        last_year = weekly.iloc[-52:]

        future_forecast = pd.Series(
            last_year.iloc[:FORECAST_WEEKS].values,
            index=future_dates,
        )

    future_forecast = future_forecast.clip(lower=0)

    history_to_show = weekly.iloc[-52:]

    fig_future = go.Figure()

    fig_future.add_trace(
        go.Scatter(
            x=history_to_show.index,
            y=history_to_show.values,
            mode="lines",
            name="Historical demand",
        )
    )

    fig_future.add_trace(
        go.Scatter(
            x=future_forecast.index,
            y=future_forecast.values,
            mode="lines+markers",
            name="13-week forecast",
        )
    )

    fig_future.update_layout(
        template="plotly_dark",
        paper_bgcolor="#061523",
        plot_bgcolor="#061523",
        height=430,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Weekly demand",
        xaxis_title="",
    )

    st.subheader("Next 13 Weeks")

    st.plotly_chart(
        fig_future,
        use_container_width=True,
    )


# ============================================================
# PLANNING MATRIX
# ============================================================

st.markdown(
    '<div class="section-number">06 / Commercial Planning</div>',
    unsafe_allow_html=True,
)

st.header("Demand × Momentum × Risk")


def assign_decision(row, demand_high, demand_low, volatility_high):

    growth = row["Growth Rate"]
    demand = row["Total Demand"]
    volatility = row["Volatility"]

    if (
        pd.notna(growth)
        and growth > 0.05
        and demand >= demand_high
        and volatility < volatility_high
    ):
        return "INCREASE"

    elif (
        pd.notna(growth)
        and growth < -0.05
        and demand <= demand_low
    ):
        return "REDUCE"

    elif volatility >= volatility_high:
        return "WATCH"

    else:
        return "MAINTAIN"


demand_high = item_summary["Total Demand"].quantile(0.70)
demand_low = item_summary["Total Demand"].quantile(0.30)
volatility_high = item_summary["Volatility"].quantile(0.70)

item_summary["Planning Decision"] = item_summary.apply(
    assign_decision,
    axis=1,
    args=(
        demand_high,
        demand_low,
        volatility_high,
    ),
)


fig_matrix = px.scatter(
    item_summary,
    x="Total Demand",
    y="Growth Rate",
    size="Demand Share",
    color="Planning Decision",
    hover_name="item",
    hover_data={
        "Volatility": ":.2f",
        "Demand Share": ":.2%",
        "Growth Rate": ":.1%",
    },
    labels={
        "Total Demand": "Historical demand",
        "Growth Rate": "Recent 90-day momentum",
    },
)

fig_matrix.add_hline(
    y=0,
    line_dash="dash",
    opacity=0.4,
)

fig_matrix.add_vline(
    x=item_summary["Total Demand"].median(),
    line_dash="dash",
    opacity=0.4,
)

fig_matrix.update_layout(
    template="plotly_dark",
    paper_bgcolor="#061523",
    plot_bgcolor="#061523",
    height=550,
    margin=dict(l=10, r=10, t=30, b=10),
)

st.plotly_chart(
    fig_matrix,
    use_container_width=True,
)


# ============================================================
# DECISION TABLE
# ============================================================

st.markdown(
    '<div class="section-number">07 / Decision Output</div>',
    unsafe_allow_html=True,
)

st.header("What should we do next?")

decision_table = item_summary[
    [
        "item",
        "Total Demand",
        "Demand Share",
        "Growth Rate",
        "Volatility",
        "Planning Decision",
    ]
].copy()

decision_table.columns = [
    "Item",
    "Historical Demand",
    "Demand Share",
    "Recent Momentum",
    "Volatility",
    "Planning Decision",
]

decision_table = decision_table.sort_values(
    ["Planning Decision", "Historical Demand"],
    ascending=[True, False],
)

st.dataframe(
    decision_table.style.format(
        {
            "Historical Demand": "{:,.0f}",
            "Demand Share": "{:.1%}",
            "Recent Momentum": "{:+.1%}",
            "Volatility": "{:.2f}",
        }
    ),
    use_container_width=True,
    height=500,
)


# ============================================================
# METHODOLOGY / LIMITATIONS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-number">Method / Limitations</div>',
    unsafe_allow_html=True,
)

with st.expander("How the analysis works"):

    st.markdown(
        """
        **Demand**
        is calculated from historical unit sales.

        **Recent momentum**
        compares the most recent 90 days with the preceding 90 days.

        **Volatility**
        is measured using the coefficient of variation of daily demand.

        **Baseline forecast**
        uses a seasonal-naive approach: comparable weeks from the
        previous year.

        **Stronger forecast**
        uses Holt-Winters exponential smoothing with additive trend
        and 52-week seasonality.

        **Validation**
        uses the final 13 weeks as a chronological holdout period.
        No random train/test splitting is used.

        **Planning decisions**
        combine demand level, recent momentum, and volatility.
        Thresholds are relative to the observed item distribution,
        rather than invented inventory constraints.

        **Important limitation**
        This dataset contains historical sales demand, not inventory,
        stock-outs, margins, lead times, purchase orders, product
        categories, or wholesale-account information. Therefore,
        recommendations should be interpreted as demand-planning
        signals rather than inventory optimization instructions.
        """
    )


st.caption(
    "Portfolio project · Retail Demand Planning · "
    "Python / Pandas / Plotly / Statsmodels"
)
