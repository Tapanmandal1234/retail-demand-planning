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
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Retail Demand Planning | Tapan Mandal",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. DESIGN SYSTEM
# ============================================================

BG = "#061523"
PANEL = "#0d1e2e"
CREAM = "#F2EEE5"
MUTED = "#94A3AF"
GRID = "rgba(238,233,223,0.18)"

HISTORICAL = "#8296A8"
POSITIVE = "#72B7A8"
NEGATIVE = "#D9826B"
WATCH = "#D6A15F"
MAINTAIN = "#A7B5C1"
ACCENT = "#D8A28D"


st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {BG};
        color: {CREAM};
    }}

    .block-container {{
        max-width: 1180px;
        padding-top: 3.5rem;
        padding-bottom: 5rem;
    }}

    [data-testid="stSidebar"] {{
        background-color: #071b2a;
        border-right: 1px solid rgba(255,255,255,0.10);
    }}

    h1, h2, h3 {{
        font-family: Georgia, serif !important;
        color: {CREAM} !important;
        letter-spacing: -0.025em;
    }}

    h1 {{
        font-size: 3.15rem !important;
    }}

    p, div, span, label {{
        color: {CREAM};
    }}

    .eyebrow {{
        font-size: 10px;
        letter-spacing: 0.20em;
        text-transform: uppercase;
        color: {MUTED};
        margin-bottom: 14px;
    }}

    .intro {{
        font-size: 18px;
        line-height: 1.65;
        max-width: 930px;
        color: #CBD3D8;
        margin-bottom: 45px;
    }}

    .section-number {{
        font-size: 10px;
        letter-spacing: 0.20em;
        text-transform: uppercase;
        color: {MUTED};
        margin-top: 45px;
        margin-bottom: 8px;
    }}

    .insight-box {{
        border-left: 2px solid {ACCENT};
        background-color: rgba(255,255,255,0.035);
        padding: 18px 22px;
        margin-top: 15px;
        margin-bottom: 30px;
        line-height: 1.6;
    }}

    .method-box {{
        background-color: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.10);
        padding: 20px 22px;
        line-height: 1.6;
    }}

    [data-testid="stMetric"] {{
        background-color: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.12);
        padding: 18px;
        min-height: 118px;
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    hr {{
        border-color: rgba(255,255,255,0.12);
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. PLOTLY THEME
# ============================================================

def style_figure(fig, height=430, legend=True):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=MUTED),
        height=height,
        margin=dict(l=15, r=15, t=30, b=20),
        showlegend=legend,
        hoverlabel=dict(
            bgcolor=PANEL,
            font_color=CREAM,
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
    )

    return fig


# ============================================================
# 4. HEADER
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
    Using five years of product and store demand to identify seasonal
    patterns, distinguish genuine momentum from seasonality, test
    alternative forecasting approaches, and translate the evidence
    into commercial planning signals.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. LOAD REAL DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(source):

    df = pd.read_csv(source)

    required = {"date", "store", "item", "sales"}

    if not required.issubset(df.columns):
        raise ValueError(
            "Dataset must contain date, store, item and sales."
        )

    df["date"] = pd.to_datetime(df["date"])
    df["store"] = pd.to_numeric(df["store"], errors="coerce")
    df["item"] = pd.to_numeric(df["item"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    df = df.dropna(
        subset=["date", "store", "item", "sales"]
    )

    df["store"] = df["store"].astype(int)
    df["item"] = df["item"].astype(int)

    return df.sort_values("date")


DATA_PATH = "train.csv"

if os.path.exists(DATA_PATH):
    source = DATA_PATH

else:

    st.warning(
        "train.csv was not found in the repository. "
        "Upload the Kaggle training file below."
    )

    source = st.file_uploader(
        "Upload train.csv",
        type=["csv"],
    )

    if source is None:
        st.stop()


with st.spinner("Loading retail demand data..."):
    df = load_data(source)


min_date = df["date"].min()
max_date = df["date"].max()

stores = sorted(df["store"].unique())
items = sorted(df["item"].unique())


# ============================================================
# 6. SIDEBAR
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

st.sidebar.caption(
    "Momentum = latest 90 days vs "
    "same period one year earlier."
)


# ============================================================
# 7. FILTER DATA
# ============================================================

if selected_store == "All Stores":
    analysis_df = df.copy()
    scope_label = "all stores"
else:
    analysis_df = df[
        df["store"] == selected_store
    ].copy()

    scope_label = f"Store {selected_store}"


# ============================================================
# 8. YOY MOMENTUM FUNCTION
# ============================================================

def calculate_yoy_periods(data):

    end = data["date"].max()

    current_start = end - pd.Timedelta(days=89)

    prior_end = end - pd.DateOffset(years=1)
    prior_start = current_start - pd.DateOffset(years=1)

    return (
        current_start,
        end,
        prior_start,
        prior_end,
    )


(
    current_start,
    current_end,
    prior_start,
    prior_end,
) = calculate_yoy_periods(analysis_df)


current_90 = analysis_df.loc[
    analysis_df["date"].between(
        current_start,
        current_end,
    ),
    "sales",
].sum()


prior_90 = analysis_df.loc[
    analysis_df["date"].between(
        prior_start,
        prior_end,
    ),
    "sales",
].sum()


overall_yoy_momentum = (
    (current_90 - prior_90) / prior_90
    if prior_90 > 0
    else np.nan
)


# ============================================================
# 9. DEMAND LANDSCAPE
# ============================================================

st.markdown(
    '<div class="section-number">01 / Demand Landscape</div>',
    unsafe_allow_html=True,
)

st.header("Five years of demand")


daily_total = (
    analysis_df.groupby("date")["sales"]
    .sum()
)


total_sales = analysis_df["sales"].sum()
average_daily = daily_total.mean()


m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Historical Demand",
    f"{total_sales:,.0f}",
)

m2.metric(
    "Average Daily Demand",
    f"{average_daily:,.0f}",
)

m3.metric(
    "Items",
    f"{analysis_df['item'].nunique()}",
)

m4.metric(
    "Recent YoY Momentum",
    (
        "N/A"
        if pd.isna(overall_yoy_momentum)
        else f"{overall_yoy_momentum:+.1%}"
    ),
)


monthly = (
    analysis_df
    .set_index("date")
    .resample("MS")["sales"]
    .sum()
    .reset_index()
)


fig_monthly = go.Figure()

fig_monthly.add_trace(
    go.Scatter(
        x=monthly["date"],
        y=monthly["sales"],
        mode="lines",
        line=dict(
            color=CREAM,
            width=2.4,
        ),
        hovertemplate=(
            "%{x|%b %Y}<br>"
            "%{y:,.0f} units"
            "<extra></extra>"
        ),
    )
)

style_figure(
    fig_monthly,
    height=430,
    legend=False,
)

fig_monthly.update_yaxes(
    title="Monthly demand"
)

st.plotly_chart(
    fig_monthly,
    use_container_width=True,
)


# ============================================================
# 10. SEASONALITY
# ============================================================

st.markdown(
    '<div class="section-number">02 / Seasonality</div>',
    unsafe_allow_html=True,
)

st.header("When does demand peak?")


seasonality_df = analysis_df.copy()

seasonality_df["month"] = (
    seasonality_df["date"].dt.month
)


seasonality = (
    seasonality_df
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


seasonality["Month"] = (
    seasonality["month"].map(month_names)
)


peak_row = seasonality.loc[
    seasonality["sales"].idxmax()
]

low_row = seasonality.loc[
    seasonality["sales"].idxmin()
]


peak_month = peak_row["Month"]
low_month = low_row["Month"]

peak_value = peak_row["sales"]
low_value = low_row["sales"]


seasonal_gap = (
    (peak_value / low_value) - 1
    if low_value > 0
    else np.nan
)


bar_colors = [
    ACCENT if m == peak_month else HISTORICAL
    for m in seasonality["Month"]
]


fig_season = go.Figure()

fig_season.add_trace(
    go.Bar(
        x=seasonality["Month"],
        y=seasonality["sales"],
        marker_color=bar_colors,
        hovertemplate=(
            "%{x}<br>"
            "%{y:.1f} average demand"
            "<extra></extra>"
        ),
    )
)


style_figure(
    fig_season,
    height=390,
    legend=False,
)

fig_season.update_yaxes(
    title="Average daily item-store demand"
)


st.plotly_chart(
    fig_season,
    use_container_width=True,
)


if not pd.isna(seasonal_gap):

    st.markdown(
        f"""
        <div class="insight-box">
        <b>Demand signal:</b> {peak_month} has the strongest
        average demand and {low_month} the weakest.
        Average demand in {peak_month} is approximately
        <b>{seasonal_gap:.0%} higher</b> than in {low_month}.
        This pronounced seasonal pattern is why recent performance
        should be compared with the same period one year earlier,
        rather than simply with the preceding quarter.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 11. ITEM SUMMARY WITH CORRECTED YOY MOMENTUM
# ============================================================

@st.cache_data(show_spinner=False)
def build_item_summary(data):

    end = data["date"].max()

    current_start = end - pd.Timedelta(days=89)

    prior_end = end - pd.DateOffset(years=1)
    prior_start = current_start - pd.DateOffset(years=1)


    total = (
        data.groupby("item")["sales"]
        .sum()
        .rename("Historical Demand")
    )


    daily_item = (
        data.groupby(
            ["date", "item"]
        )["sales"]
        .sum()
        .reset_index()
    )


    volatility_stats = (
        daily_item
        .groupby("item")["sales"]
        .agg(["mean", "std"])
    )


    volatility_stats["Volatility"] = (
        volatility_stats["std"]
        / volatility_stats["mean"]
        .replace(0, np.nan)
    )


    current = (
        data[
            data["date"].between(
                current_start,
                end,
            )
        ]
        .groupby("item")["sales"]
        .sum()
        .rename("Current 90D")
    )


    prior = (
        data[
            data["date"].between(
                prior_start,
                prior_end,
            )
        ]
        .groupby("item")["sales"]
        .sum()
        .rename("Prior Year 90D")
    )


    summary = pd.concat(
        [
            total,
            current,
            prior,
            volatility_stats["Volatility"],
        ],
        axis=1,
    )


    summary["YoY Momentum"] = np.where(
        summary["Prior Year 90D"] > 0,
        (
            summary["Current 90D"]
            - summary["Prior Year 90D"]
        )
        / summary["Prior Year 90D"],
        np.nan,
    )


    summary["Demand Share"] = (
        summary["Historical Demand"]
        / summary["Historical Demand"].sum()
    )


    return summary.reset_index()


item_summary = build_item_summary(
    analysis_df
)


# ============================================================
# 12. PRODUCT PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-number">03 / Product Performance</div>',
    unsafe_allow_html=True,
)

st.header("Which items drive demand?")


top_items = (
    item_summary
    .nlargest(
        10,
        "Historical Demand",
    )
    .sort_values(
        "Historical Demand"
    )
)


fig_products = go.Figure()

fig_products.add_trace(
    go.Bar(
        x=top_items["Historical Demand"],
        y=[
            f"Item {x}"
            for x in top_items["item"]
        ],
        orientation="h",
        marker_color=HISTORICAL,
        hovertemplate=(
            "%{y}<br>"
            "%{x:,.0f} units"
            "<extra></extra>"
        ),
    )
)


style_figure(
    fig_products,
    height=450,
    legend=False,
)

fig_products.update_xaxes(
    title="Historical demand"
)


st.plotly_chart(
    fig_products,
    use_container_width=True,
)


top_item = item_summary.loc[
    item_summary["Historical Demand"].idxmax()
]


st.markdown(
    f"""
    <div class="insight-box">
    <b>Product concentration:</b>
    Item {int(top_item['item'])} is the highest-demand product,
    accounting for approximately
    <b>{top_item['Demand Share']:.1%}</b>
    of historical demand in {scope_label}.
    The ranking provides a view of demand scale; momentum and
    forecasting below determine whether high historical demand is
    continuing.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 13. STORE × ITEM ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-number">04 / Store Differences</div>',
    unsafe_allow_html=True,
)

st.header(
    "Does the same assortment perform everywhere?"
)


store_totals = (
    df.groupby("store")["sales"]
    .sum()
    .sort_values(
        ascending=False
    )
)


strongest_store = store_totals.index[0]
weakest_store = store_totals.index[-1]

store_ratio = (
    store_totals.iloc[0]
    / store_totals.iloc[-1]
)


top_15_items = (
    df.groupby("item")["sales"]
    .sum()
    .nlargest(15)
    .index
)


heatmap_source = (
    df[
        df["item"].isin(
            top_15_items
        )
    ]
    .groupby(
        ["store", "item"]
    )["sales"]
    .sum()
    .reset_index()
)


heatmap_pivot = (
    heatmap_source
    .pivot(
        index="store",
        columns="item",
        values="sales",
    )
)


fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=heatmap_pivot.values,
        x=[
            f"Item {x}"
            for x in heatmap_pivot.columns
        ],
        y=[
            f"Store {x}"
            for x in heatmap_pivot.index
        ],
        colorscale=[
            [0.0, "#172B3A"],
            [0.5, "#627D91"],
            [1.0, "#D8A28D"],
        ],
        colorbar=dict(
            title="Demand",
        ),
        hovertemplate=(
            "%{y}<br>"
            "%{x}<br>"
            "%{z:,.0f} units"
            "<extra></extra>"
        ),
    )
)


style_figure(
    fig_heatmap,
    height=450,
    legend=False,
)


st.plotly_chart(
    fig_heatmap,
    use_container_width=True,
)


st.markdown(
    f"""
    <div class="insight-box">
    <b>Store signal:</b>
    Store {strongest_store} generated the highest historical demand,
    while Store {weakest_store} generated the lowest.
    Total demand at the strongest location is approximately
    <b>{store_ratio:.1f}×</b> that of the weakest.
    This suggests that store-level volume differences should be
    considered rather than assuming every location requires the same
    absolute demand plan.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 14. FORECAST HELPERS
# ============================================================

def calculate_mae(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return np.mean(
        np.abs(
            actual - predicted
        )
    )


def calculate_mape(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    if mask.sum() == 0:
        return np.nan

    return (
        np.mean(
            np.abs(
                (
                    actual[mask]
                    - predicted[mask]
                )
                / actual[mask]
            )
        )
        * 100
    )


@st.cache_data(show_spinner=False)
def prepare_weekly_series(
    data,
    item,
):

    item_data = data[
        data["item"] == item
    ].copy()

    weekly = (
        item_data
        .set_index("date")
        .resample("W")["sales"]
        .sum()
        .asfreq(
            "W",
            fill_value=0,
        )
    )

    return weekly


# ============================================================
# 15. FORECASTING
# ============================================================

st.markdown(
    '<div class="section-number">05 / Forecasting</div>',
    unsafe_allow_html=True,
)

st.header(
    f"Can we forecast Item {selected_item}?"
)


weekly = prepare_weekly_series(
    analysis_df,
    selected_item,
)


TEST_WEEKS = 13
FORECAST_WEEKS = 13


if len(weekly) < 120:

    st.warning(
        "There is not enough weekly history "
        "for reliable seasonal forecasting."
    )

else:

    train = weekly.iloc[:-TEST_WEEKS]
    test = weekly.iloc[-TEST_WEEKS:]


    # --------------------------------------------------------
    # BASELINE: SEASONAL NAIVE
    # --------------------------------------------------------

    baseline = (
        weekly
        .shift(52)
        .loc[test.index]
    )


    if baseline.isna().any():

        baseline = pd.Series(
            [train.iloc[-1]]
            * len(test),
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
    # HOLT-WINTERS
    # --------------------------------------------------------

    model_success = False


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


        hw_prediction = model.forecast(
            TEST_WEEKS
        )


        hw_mae = calculate_mae(
            test.values,
            hw_prediction.values,
        )


        hw_mape = calculate_mape(
            test.values,
            hw_prediction.values,
        )


        model_success = True


    except Exception:

        hw_prediction = baseline.copy()
        hw_mae = baseline_mae
        hw_mape = baseline_mape


    # --------------------------------------------------------
    # MODEL METRICS
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
        f"{hw_mae:,.0f}",
    )

    c4.metric(
        "Holt-Winters MAPE",
        f"{hw_mape:.1f}%",
    )


    # --------------------------------------------------------
    # HOLDOUT GRAPH
    # --------------------------------------------------------

    fig_test = go.Figure()


    fig_test.add_trace(
        go.Scatter(
            x=test.index,
            y=test.values,
            mode="lines+markers",
            name="Actual",
            line=dict(
                color=CREAM,
                width=2.5,
            ),
        )
    )


    fig_test.add_trace(
        go.Scatter(
            x=baseline.index,
            y=baseline.values,
            mode="lines",
            name="Seasonal Naive",
            line=dict(
                color=ACCENT,
                width=2,
            ),
        )
    )


    fig_test.add_trace(
        go.Scatter(
            x=hw_prediction.index,
            y=hw_prediction.values,
            mode="lines",
            name="Holt-Winters",
            line=dict(
                color=POSITIVE,
                width=2,
            ),
        )
    )


    style_figure(
        fig_test,
        height=440,
        legend=True,
    )


    fig_test.update_yaxes(
        title="Weekly demand"
    )


    st.plotly_chart(
        fig_test,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if (
        model_success
        and hw_mae < baseline_mae
    ):

        best_model_name = (
            "Holt-Winters"
        )

        winning_mae = hw_mae
        winning_mape = hw_mape

    else:

        best_model_name = (
            "Seasonal Naive"
        )

        winning_mae = baseline_mae
        winning_mape = baseline_mape


    improvement = (
        abs(
            baseline_mae
            - hw_mae
        )
        / max(
            baseline_mae,
            hw_mae,
        )
    )


    st.markdown(
        f"""
        <div class="insight-box">
        <b>Validation result:</b>
        {best_model_name} produced the lower error for Item
        {selected_item}, with a holdout MAE of
        <b>{winning_mae:,.0f}</b> units and MAPE of
        <b>{winning_mape:.1f}%</b>.
        The models were evaluated on the final {TEST_WEEKS}
        weeks as an unseen chronological test period rather than
        using a random split.
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------------

    if (
        best_model_name
        == "Holt-Winters"
        and model_success
    ):

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


        future_forecast = (
            final_model.forecast(
                FORECAST_WEEKS
            )
        )


    else:

        future_dates = pd.date_range(
            start=(
                weekly.index[-1]
                + pd.Timedelta(
                    weeks=1
                )
            ),
            periods=FORECAST_WEEKS,
            freq="W",
        )


        prior_year_values = (
            weekly.iloc[-52:]
            .iloc[:FORECAST_WEEKS]
            .values
        )


        future_forecast = pd.Series(
            prior_year_values,
            index=future_dates,
        )


    future_forecast = (
        future_forecast.clip(
            lower=0
        )
    )


    # --------------------------------------------------------
    # FORECAST VS RECENT COMPARISON
    # --------------------------------------------------------

    forecast_total = (
        future_forecast.sum()
    )


    recent_13_total = (
        weekly.iloc[-13:].sum()
    )


    forecast_change = (
        (
            forecast_total
            - recent_13_total
        )
        / recent_13_total
        if recent_13_total > 0
        else np.nan
    )


    st.subheader(
        "Next 13 Weeks"
    )


    history_to_show = (
        weekly.iloc[-52:]
    )


    # Add final actual point to forecast
    # so visual transition is connected.

    forecast_x = [
        history_to_show.index[-1]
    ] + list(
        future_forecast.index
    )


    forecast_y = [
        history_to_show.iloc[-1]
    ] + list(
        future_forecast.values
    )


    fig_future = go.Figure()


    fig_future.add_trace(
        go.Scatter(
            x=history_to_show.index,
            y=history_to_show.values,
            mode="lines",
            name="Historical demand",
            line=dict(
                color=HISTORICAL,
                width=2,
            ),
        )
    )


    fig_future.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode="lines+markers",
            name="13-week forecast",
            line=dict(
                color=ACCENT,
                width=2.5,
            ),
        )
    )


    forecast_start = (
        future_forecast.index[0]
    )


    fig_future.add_vline(
        x=forecast_start.timestamp()
        * 1000,
        line_dash="dash",
        line_color=MUTED,
        opacity=0.7,
    )


    fig_future.add_annotation(
        x=forecast_start,
        y=1,
        yref="paper",
        text="FORECAST START",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(
            size=9,
            color=MUTED,
        ),
    )


    style_figure(
        fig_future,
        height=440,
        legend=True,
    )


    fig_future.update_yaxes(
        title="Weekly demand"
    )


    st.plotly_chart(
        fig_future,
        use_container_width=True,
    )


    direction_word = (
        "higher"
        if forecast_change >= 0
        else "lower"
    )


    st.markdown(
        f"""
        <div class="insight-box">
        <b>Forward signal:</b>
        the selected model forecasts approximately
        <b>{forecast_total:,.0f} units</b>
        over the next 13 weeks for Item {selected_item}.
        That is
        <b>{abs(forecast_change):.1%} {direction_word}</b>
        than demand in the most recent 13 observed weeks.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 16. COMMERCIAL PLANNING MATRIX
# ============================================================

st.markdown(
    '<div class="section-number">06 / Commercial Planning</div>',
    unsafe_allow_html=True,
)

st.header(
    "Demand × Momentum × Risk"
)


# Relative thresholds
demand_high = (
    item_summary[
        "Historical Demand"
    ].quantile(0.70)
)

demand_low = (
    item_summary[
        "Historical Demand"
    ].quantile(0.30)
)

volatility_high = (
    item_summary[
        "Volatility"
    ].quantile(0.70)
)


def assign_decision(row):

    growth = row["YoY Momentum"]
    demand = row["Historical Demand"]
    volatility = row["Volatility"]


    if pd.isna(growth):

        return "WATCH"


    # Strong scale + positive YoY momentum
    if (
        growth >= 0.05
        and demand >= demand_high
        and volatility < volatility_high
    ):

        return "INCREASE"


    # Weak scale + material decline
    if (
        growth <= -0.05
        and demand <= demand_low
    ):

        return "REDUCE"


    # High uncertainty / instability
    if volatility >= volatility_high:

        return "WATCH"


    return "MAINTAIN"


item_summary[
    "Planning Decision"
] = item_summary.apply(
    assign_decision,
    axis=1,
)


decision_colors = {
    "INCREASE": POSITIVE,
    "MAINTAIN": MAINTAIN,
    "WATCH": WATCH,
    "REDUCE": NEGATIVE,
}


fig_matrix = px.scatter(
    item_summary,
    x="Historical Demand",
    y="YoY Momentum",
    size="Demand Share",
    color="Planning Decision",
    color_discrete_map=decision_colors,
    hover_name="item",
    hover_data={
        "Demand Share": ":.2%",
        "YoY Momentum": ":+.1%",
        "Volatility": ":.2f",
        "Historical Demand": ":,.0f",
    },
    labels={
        "Historical Demand":
            "Historical demand",
        "YoY Momentum":
            "Recent YoY momentum",
    },
)


fig_matrix.add_hline(
    y=0,
    line_dash="dash",
    line_color=MUTED,
    opacity=0.65,
)


median_demand = (
    item_summary[
        "Historical Demand"
    ].median()
)


fig_matrix.add_vline(
    x=median_demand,
    line_dash="dash",
    line_color=MUTED,
    opacity=0.65,
)


# Quadrant annotations
x_min = (
    item_summary[
        "Historical Demand"
    ].min()
)

x_max = (
    item_summary[
        "Historical Demand"
    ].max()
)

y_min = (
    item_summary[
        "YoY Momentum"
    ].min()
)

y_max = (
    item_summary[
        "YoY Momentum"
    ].max()
)


if (
    pd.notna(y_min)
    and pd.notna(y_max)
):

    fig_matrix.add_annotation(
        x=x_max,
        y=y_max,
        text="HIGH DEMAND / GROWING",
        showarrow=False,
        xanchor="right",
        font=dict(
            size=9,
            color=MUTED,
        ),
    )

    fig_matrix.add_annotation(
        x=x_min,
        y=y_max,
        text="EMERGING",
        showarrow=False,
        xanchor="left",
        font=dict(
            size=9,
            color=MUTED,
        ),
    )

    fig_matrix.add_annotation(
        x=x_max,
        y=y_min,
        text="HIGH DEMAND / SOFTENING",
        showarrow=False,
        xanchor="right",
        font=dict(
            size=9,
            color=MUTED,
        ),
    )

    fig_matrix.add_annotation(
        x=x_min,
        y=y_min,
        text="LOW DEMAND / DECLINING",
        showarrow=False,
        xanchor="left",
        font=dict(
            size=9,
            color=MUTED,
        ),
    )


style_figure(
    fig_matrix,
    height=560,
    legend=True,
)


fig_matrix.update_yaxes(
    tickformat=".0%",
)


st.plotly_chart(
    fig_matrix,
    use_container_width=True,
)


increase_count = (
    item_summary[
        "Planning Decision"
    ]
    .eq("INCREASE")
    .sum()
)

maintain_count = (
    item_summary[
        "Planning Decision"
    ]
    .eq("MAINTAIN")
    .sum()
)

watch_count = (
    item_summary[
        "Planning Decision"
    ]
    .eq("WATCH")
    .sum()
)

reduce_count = (
    item_summary[
        "Planning Decision"
    ]
    .eq("REDUCE")
    .sum()
)


st.markdown(
    f"""
    <div class="insight-box">
    <b>Planning signal:</b>
    based on relative demand scale, same-period YoY momentum and
    volatility, the framework currently identifies
    <b>{increase_count} increase</b>,
    <b>{maintain_count} maintain</b>,
    <b>{watch_count} watch</b>, and
    <b>{reduce_count} reduce</b> signals.
    These are demand-planning indicators rather than purchase-order
    recommendations because the source data does not contain inventory,
    margins or lead times.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 17. SELECTED ITEM DECISION
# ============================================================

selected_summary = (
    item_summary[
        item_summary["item"]
        == selected_item
    ]
    .iloc[0]
)


st.markdown(
    '<div class="section-number">07 / Decision Output</div>',
    unsafe_allow_html=True,
)

st.header(
    "What should we do next?"
)


s1, s2, s3, s4 = st.columns(4)


s1.metric(
    f"Item {selected_item}",
    selected_summary[
        "Planning Decision"
    ],
)


s2.metric(
    "Historical Demand",
    f"""
    {selected_summary[
        'Historical Demand'
    ]:,.0f}
    """.strip(),
)


s3.metric(
    "Recent YoY Momentum",
    (
        "N/A"
        if pd.isna(
            selected_summary[
                "YoY Momentum"
            ]
        )
        else
        f"""
        {selected_summary[
            'YoY Momentum'
        ]:+.1%}
        """.strip()
    ),
)


s4.metric(
    "Demand Share",
    f"""
    {selected_summary[
        'Demand Share'
    ]:.1%}
    """.strip(),
)


# ============================================================
# 18. FINAL DECISION TABLE
# ============================================================

decision_table = item_summary[
    [
        "item",
        "Historical Demand",
        "Demand Share",
        "YoY Momentum",
        "Volatility",
        "Planning Decision",
    ]
].copy()


decision_table.columns = [
    "Item",
    "Historical Demand",
    "Demand Share",
    "Recent YoY Momentum",
    "Volatility",
    "Planning Decision",
]


decision_order = {
    "INCREASE": 1,
    "WATCH": 2,
    "MAINTAIN": 3,
    "REDUCE": 4,
}


decision_table["_order"] = (
    decision_table[
        "Planning Decision"
    ].map(decision_order)
)


decision_table = (
    decision_table
    .sort_values(
        [
            "_order",
            "Historical Demand",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .drop(
        columns="_order"
    )
)


st.dataframe(
    decision_table.style.format(
        {
            "Historical Demand":
                "{:,.0f}",
            "Demand Share":
                "{:.1%}",
            "Recent YoY Momentum":
                "{:+.1%}",
            "Volatility":
                "{:.2f}",
        }
    ),
    use_container_width=True,
    height=500,
)


# ============================================================
# 19. METHODOLOGY / LIMITATIONS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-number">Method / Limitations</div>',
    unsafe_allow_html=True,
)


with st.expander(
    "How the analysis works"
):

    st.markdown(
        """
### Demand
Historical demand is calculated from observed unit sales.

### Recent momentum
Momentum compares the **latest 90 days with the same 90-day
period one year earlier**. This avoids treating predictable
seasonality as underlying growth or decline.

### Volatility
Volatility is measured using the coefficient of variation of
daily demand.

### Baseline forecast
The baseline is a **seasonal-naive forecast**, using comparable
weeks from the previous year.

### Alternative forecast
The second model uses **Holt-Winters exponential smoothing**
with additive trend and 52-week seasonality.

### Validation
The final 13 observed weeks are held out as an unseen
chronological test period. The models are compared using
**MAE and MAPE**. No random train/test split is used.

### Commercial planning
Planning signals combine:

- historical demand scale,
- same-period year-over-year momentum,
- demand volatility.

The thresholds are relative to the observed item distribution
rather than invented inventory assumptions.

### Limitation
The dataset contains historical sales demand. It does **not**
contain inventory levels, stock-outs, gross margin, lead times,
purchase orders, product categories, markdowns, or wholesale
account information.

Therefore, **INCREASE / MAINTAIN / WATCH / REDUCE are
demand-planning signals, not literal inventory purchase
instructions.**
        """
    )


st.caption(
    "Portfolio project · Retail Demand Planning · "
    "Python / Pandas / Plotly / Statsmodels"
)
