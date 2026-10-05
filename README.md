# Retail Demand Planning & Forecasting

An interactive demand-planning application that turns historical retail sales data into forward-looking inventory and planning insights.

**Live Application:**  
https://retail-demand-planning-ttcwzam3p4or4fumngwsoc.streamlit.app/

---

## Overview

Retail demand planning is not just about predicting a number. Decision-makers need to understand:

- What is demand doing now?
- Is recent performance stronger or weaker than last year?
- How seasonal is the product?
- What should we expect next?
- How much confidence should we place in the forecast?

This project builds an interactive demand-planning workflow around those questions.

Rather than presenting a static forecasting notebook, the application allows users to explore demand patterns across stores and products, evaluate recent year-over-year momentum, and generate forecasts using historical seasonality.

---

## Business Question

**How can historical store-item demand be translated into a practical planning signal for future inventory and demand decisions?**

The application focuses on three components:

1. **Historical Demand** — understanding long-term volume and seasonality
2. **Recent Momentum** — identifying whether demand is strengthening or weakening versus the comparable period last year
3. **Forward Forecasting** — estimating expected future demand while validating the forecasting approach against historical data

---

## What the Application Does

### Historical Demand Analysis

Users can select individual stores and items to explore:

- historical sales volume
- long-term demand patterns
- seasonality
- recent performance

This creates the context necessary before interpreting a forecast.

### Recent YoY Momentum

Recent demand is compared with the equivalent period from the prior year.

The momentum signal uses the latest 90 days of available demand and compares it with the same calendar period one year earlier.

This helps distinguish:

- accelerating demand
- relatively stable demand
- weakening demand

Using the comparable prior-year period prevents normal seasonal changes from being mistaken for genuine momentum.

### Forecasting

The application evaluates forecasting approaches using historical data before producing the forward-looking forecast.

The forecasting workflow incorporates recurring seasonal demand patterns rather than simply extending the latest trend.

This allows the model to account for the fact that retail demand may naturally rise and fall at different points of the year.

### Forecast Validation

Forecasting performance is evaluated against historical observations.

The application compares seasonal forecasting approaches to determine whether the selected model improves on a straightforward seasonal baseline.

This is important because a more complex model should only be preferred when it provides meaningful forecasting value.

---

## Dataset

The project uses the **Store Item Demand Forecasting Challenge** dataset originally published on Kaggle.

The dataset contains daily sales observations from:

- **10 stores**
- **50 items**
- **2013–2017**

This provides multiple years of daily observations for analyzing seasonality, demand patterns, and forecasting performance.

---

## Analytical Approach

The project follows a practical demand-planning workflow:

**Historical Sales**

↓

**Demand & Seasonality Analysis**

↓

**Recent Year-over-Year Momentum**

↓

**Forecast Validation**

↓

**Forward Demand Forecast**

The goal is not simply to generate a prediction, but to provide enough context for someone to understand what is driving the planning signal.

---

## Tools & Technologies

- **Python**
- **Pandas**
- **NumPy**
- **Statsmodels**
- **Plotly**
- **Streamlit**
- Time-series forecasting
- Forecast validation
- Interactive data visualization

---

## Why I Built This

I wanted to build a project that went beyond exploratory data analysis and demonstrated how analytics can support an actual operating decision.

Demand forecasting becomes more useful when the prediction is connected to:

- historical context
- seasonality
- recent momentum
- model validation

The project therefore focuses on turning time-series analysis into a decision-support tool rather than presenting a forecasting model in isolation.

---

## Limitations

This is a portfolio analytics project and not a production inventory optimization system.

The dataset does not include several variables that would typically influence real-world retail demand, including:

- promotions
- pricing changes
- inventory availability
- stockouts
- holidays and events
- competitor activity
- product lifecycle changes
- supplier lead times

A production demand-planning system would incorporate these factors alongside historical sales.

---

## Live Demo

Explore the application here:

https://retail-demand-planning-ttcwzam3p4or4fumngwsoc.streamlit.app/

---

## Author

**Tapan Mandal**

Built as a portfolio project demonstrating demand forecasting, business analytics, and decision-oriented data visualization.
