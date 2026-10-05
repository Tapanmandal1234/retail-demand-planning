import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Page Configuration (Fixed the typo here!)
st.set_page_config(
    page_title="Retail Demand Planning | Tapan Mandal",
    layout="wide",
)

# Header Section
st.markdown("### 01 / RETAIL STRATEGY")
st.title("Fashion Demand Planning")
st.markdown("Using historical product and store performance to forecast demand, calculate a Growth × Demand × Volatility index, and translate it into commercial planning decisions.")

st.markdown("---")

# Generate Mock Data (We will replace this with real Kaggle data later!)
@st.cache_data
def load_data():
    np.random.seed(42)
    items = [f"ITEM_{i:03d}" for i in range(1, 31)]
    data = []
    for item in items:
        demand = np.random.randint(500, 5000)
        growth = np.random.uniform(-0.15, 0.25)
        volatility = np.random.uniform(0.1, 0.8)
        score = growth * demand / (volatility + 0.5)

        if growth > 0.05 and demand > 2500:
            decision = "INCREASE"
        elif growth < -0.05 and demand < 1500:
            decision = "REDUCE"
        elif volatility > 0.6:
            decision = "WATCH"
        else:
            decision = "MAINTAIN"

        data.append([item, demand, growth, volatility, score, decision])
        
    return pd.DataFrame(
        data,
        columns=[
            "Item",
            "Total Demand",
            "Growth Rate",
            "Volatility",
            "Strategic Score",
            "Planning Decision",
        ],
    )

df = load_data()

# Sidebar Controls
st.sidebar.markdown("### PLANNING CONTROLS")
selected_decision = st.sidebar.multiselect(
    "Filter Decision Type",
    options=["INCREASE", "MAINTAIN", "WATCH", "REDUCE"],
    default=["INCREASE", "MAINTAIN", "WATCH", "REDUCE"],
)

filtered_df = df[df["Planning Decision"].isin(selected_decision)]

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Items Tracked", len(df))
col2.metric("Recommended Increases", len(df[df["Planning Decision"] == "INCREASE"]))
col3.metric("Watchlist Items", len(df[df["Planning Decision"] == "WATCH"]))
col4.metric("Reduction Targets", len(df[df["Planning Decision"] == "REDUCE"]))

st.markdown("---")

# Strategic Matrix Chart
st.subheader("Growth × Demand Strategic Matrix")

fig = px.scatter(
    filtered_df,
    x="Total Demand",
    y="Growth Rate",
    size="Strategic Score",
    color="Planning Decision",
    hover_name="Item",
    color_discrete_map={
        "INCREASE": "#4a9bc0",
        "MAINTAIN": "#eee9df",
        "WATCH": "#d6a08d",
        "REDUCE": "#6e1d1d",
    },
    template="plotly_dark",
)

fig.update_layout(
    plot_bgcolor="#061523",
    paper_bgcolor="#061523",
    font=dict(family="Arial", color="#eee9df", size=10),
    margin=dict(t=20, l=20, r=20, b=20),
)

st.plotly_chart(fig, use_container_width=True)

# Decision Summary Table
st.subheader("Commercial Decision Summary")
st.dataframe(
    filtered_df.sort_values(by="Total Demand", ascending=False),
    use_container_width=True,
    height=350,
)
