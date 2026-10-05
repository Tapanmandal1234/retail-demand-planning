import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Retail Demand Planning ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Aggressive Custom CSS for Futuristic / Editorial UI
custom_css = """

"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. Chapter 1: The Context
st.markdown(""" 01 / RETAIL STRATEGY & FORECASTING

""", unsafe_allow_html=True)
st.title("Fashion Demand Planning")
st.markdown(
"""
Historical sales data only tells you what happened, not what to do next. This engine processes historical item performance to forecast demand, calculates a custom Growth × Demand × Volatility index, and translates the math into explicit commercial planning decisions.""",
unsafe_allow_html=True,
)

# 4. THE REAL DATA ENGINE
@st.cache_data
def load_and_process_real_data():
try:
# Read the real Kaggle CSV file
df_raw = pd.read_csv('train.csv', parse_dates=['date'])# Separate recent 90 days vs previous 90 days to find actual Growth Momentum
    max_date = df_raw['date'].max()
    recent_cutoff = max_date - pd.Timedelta(days=90)
    older_cutoff = recent_cutoff - pd.Timedelta(days=90)
    
    recent_sales = df_raw[df_raw['date'] > recent_cutoff].groupby('item')['sales'].sum()
    older_sales = df_raw[(df_raw['date'] > older_cutoff) & (df_raw['date'] <= recent_cutoff)].groupby('item')['sales'].sum()
    
    # Calculate Total Demand and Volatility
    stats = df_raw.groupby('item').agg(
        total_demand=('sales', 'sum'),
        std_demand=('sales', 'std'),
        mean_demand=('sales', 'mean')
    )
    
    # Build the Final Output Table
    summary = pd.DataFrame()
    summary['Item'] = 'Product ' + stats.index.astype(str)
    summary['Total Demand'] = stats['total_demand'].values
    summary['Volatility'] = (stats['std_demand'] / stats['mean_demand']).values
    summary['Growth Rate'] = ((recent_sales.values - older_sales.values) / (older_sales.values + 1e-5))
    
    # Composite Strategic Score
    summary['Strategic Score'] = abs(summary['Growth Rate']) * summary['Total Demand'] / (summary['Volatility'] + 0.5)
    
    # Apply Commercial Decision Rules
    demand_70th = summary['Total Demand'].quantile(0.7)
    demand_30th = summary['Total Demand'].quantile(0.3)
    volatility_70th = summary['Volatility'].quantile(0.7)
    
    def make_decision(row):
        if row['Growth Rate'] > 0.05 and row['Total Demand'] > demand_70th:
            return 'INCREASE'
        elif row['Growth Rate'] < -0.05 and row['Total Demand'] < demand_30th:
            return 'REDUCE'
        elif row['Volatility'] > volatility_70th:
            return 'WATCH'
        else:
            return 'MAINTAIN'
            
    summary['Planning Decision'] = summary.apply(make_decision, axis=1)
    
    # Clean up the numbers
    summary['Growth Rate'] = summary['Growth Rate'].round(3)
    summary['Volatility'] = summary['Volatility'].round(3)
    summary['Strategic Score'] = summary['Strategic Score'].round(1)
    
    return summary
    
except FileNotFoundError:
    st.error("🚨 Missing Data: Please upload 'train.csv' to your GitHub repository to see the real analysis!")
    st.stop()

df = load_and_process_real_data()
# Sidebar Setup
st.sidebar.markdown("""

PLANNING PARAMETERS

""", unsafe_allow_html=True)
selected_decision = st.sidebar.multiselect(
"Active Decision Tags",
options=["INCREASE", "MAINTAIN", "WATCH", "REDUCE"],
default=["INCREASE", "MAINTAIN", "WATCH", "REDUCE"],
)
filtered_df = df[df["Planning Decision"].isin(selected_decision)]

4. Chapter 2: The Macro View
st.markdown("""

I. THE MACRO VIEW

""", unsafe_allow_html=True)
st.markdown("### Portfolio Signals")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Assortment", len(filtered_df))
col2.metric("Growth Drivers", len(filtered_df[filtered_df["Planning Decision"] == "INCREASE"]))
col3.metric("High Volatility (Watch)", len(filtered_df[filtered_df["Planning Decision"] == "WATCH"]))
col4.metric("Markdown Targets", len(filtered_df[filtered_df["Planning Decision"] == "REDUCE"]))

st.markdown("""



""", unsafe_allow_html=True)

5. Chapter 3: The Execution Matrix
st.markdown("""

II. STRATEGIC POSITIONING

""", unsafe_allow_html=True)
st.markdown("### Growth × Demand Matrix")
st.markdown(
"""

Mapping items by momentum and total volume. Bubble size represents the composite Strategic Score. Color denotes the system-recommended action.

""",
unsafe_allow_html=True
)

color_map = {
"INCREASE": "#00e5ff",
"MAINTAIN": "#eee9df",
"WATCH": "#ffaa00",
"REDUCE": "#ff2a2a",
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

fig.update_layout(
plot_bgcolor="rgba(0,0,0,0)",
paper_bgcolor="rgba(0,0,0,0)",
font=dict(family="Helvetica Neue", color="#a7a49c", size=11),
margin=dict(t=10, l=10, r=10, b=10),
xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
yaxis=dict(showgrid=False, zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""


""", unsafe_allow_html=True)

6. Chapter 4: Output Table
st.markdown("""

III. EXECUTION TABLE

""", unsafe_allow_html=True)
st.markdown("### Commercial Decision Output")

def style_tags(val):
colors = {
'INCREASE': 'color: #00e5ff; font-weight: bold;',
'MAINTAIN': 'color: #eee9df; font-weight: bold;',
'WATCH': 'color: #ffaa00; font-weight: bold;',
'REDUCE': 'color: #ff2a2a; font-weight: bold;'
}
return colors.get(val, '')

styled_df = filtered_df.sort_values(by="Total Demand", ascending=False).style.map(
style_tags, subset=['Planning Decision']
)

st.dataframe(
styled_df,
use_container_width=True,
height=400,
)
