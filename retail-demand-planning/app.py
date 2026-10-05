import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Retail Demand Planning",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Aggressive Custom CSS for Futuristic / Editorial UI
st.markdown(
    """
    
    """,
    unsafe_allow_html=True,
)

# 3. Chapter 1: The Context & Story
st.markdown('

```

01 / RETAIL STRATEGY & FORECASTING

', unsafe_allow_html=True)
st.title("Fashion Demand Planning")
st.markdown(
"""

Historical sales data only tells you what happened, not what to do next. This engine processes product and store performance to forecast demand, calculates a custom **Growth × Demand × Volatility** index, and translates the math into explicit commercial planning decisions.

```
""",
unsafe_allow_html=True,

```

)

# Generate Data

@st.cache_data
def load_data():
np.random.seed(42)
items = [f"ITEM_{i:03d}" for i in range(1, 41)]
data = []
for item in items:
demand = np.random.randint(500, 5000)
growth = np.random.uniform(-0.25, 0.35)
volatility = np.random.uniform(0.1, 0.9)

```
    score = abs(growth) * demand / (volatility + 0.5)

    if growth > 0.08 and demand > 2500:
        decision = "INCREASE"
    elif growth < -0.05 and demand < 1500:
        decision = "REDUCE"
    elif volatility > 0.65:
        decision = "WATCH"
    else:
        decision = "MAINTAIN"

    data.append([item, demand, round(growth, 3), round(volatility, 3), round(score, 1), decision])
    
return pd.DataFrame(data, columns=["Item", "Total Demand", "Growth Rate", "Volatility", "Strategic Score", "Planning Decision"])

```

df = load_data()

# Sidebar Setup

st.sidebar.markdown('

PLANNING PARAMETERS

', unsafe_allow_html=True)
selected_decision = st.sidebar.multiselect(
"Active Decision Tags",
options=["INCREASE", "MAINTAIN", "WATCH", "REDUCE"],
default=["INCREASE", "MAINTAIN", "WATCH", "REDUCE"],
)
filtered_df = df[df["Planning Decision"].isin(selected_decision)]

# 4. Chapter 2: The Macro View (Metrics)

st.markdown('

I. THE MACRO VIEW

', unsafe_allow_html=True)
st.markdown("### Portfolio Signals")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Assortment", len(filtered_df))
col2.metric("Growth Drivers", len(filtered_df[filtered_df["Planning Decision"] == "INCREASE"]))
col3.metric("High Volatility (Watch)", len(filtered_df[filtered_df["Planning Decision"] == "WATCH"]))
col4.metric("Markdown Targets", len(filtered_df[filtered_df["Planning Decision"] == "REDUCE"]))

st.markdown("





", unsafe_allow_html=True)

# 5. Chapter 3: The Execution (Visuals & Data)

st.markdown('

II. STRATEGIC POSITIONING

', unsafe_allow_html=True)
st.markdown("### Growth × Demand Matrix")
st.markdown(
'

Mapping items by momentum and total volume. Bubble size represents the composite Strategic Score. Color denotes the system-recommended action.

',
unsafe_allow_html=True
)

# Futuristic Color Palette for the Tags

color_map = {
"INCREASE": "#00e5ff",  # Cyan (High Tech)
"MAINTAIN": "#eee9df",  # Cream (Stable)
"WATCH": "#ffaa00",     # Amber (Warning)
"REDUCE": "#ff2a2a",    # Red (Danger/Cut)
}

fig = px.scatter(
filtered_df,
x="Total Demand",
y="Growth Rate",
size="Strategic Score",
color="Planning Decision",
hover_name="Item",
color_discrete_map=color_map,
template="plotly_dark",
)

# Remove gridlines for a cleaner, editorial look

fig.update_layout(
plot_bgcolor="rgba(0,0,0,0)",
paper_bgcolor="rgba(0,0,0,0)",
font=dict(family="Helvetica Neue", color="#a7a49c", size=11),
margin=dict(t=10, l=10, r=10, b=10),
xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
yaxis=dict(showgrid=False, zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("



", unsafe_allow_html=True)

# 6. Chapter 4: Output & Tags

st.markdown('

III. EXECUTION TABLE

', unsafe_allow_html=True)
st.markdown("### Commercial Decision Output")

# Function to physically color-code the dataframe tags

def style_tags(val):
colors = {
'INCREASE': 'color: #00e5ff; font-weight: bold;',
'MAINTAIN': 'color: #eee9df; font-weight: bold;',
'WATCH': 'color: #ffaa00; font-weight: bold;',
'REDUCE': 'color: #ff2a2a; font-weight: bold;'
}
return colors.get(val, '')

# Apply styling to the dataframe before rendering

styled_df = filtered_df.sort_values(by="Total Demand", ascending=False).style.map(
style_tags, subset=['Planning Decision']
)

st.dataframe(
styled_df,
use_container_width=True,
height=400,
)

```

```
