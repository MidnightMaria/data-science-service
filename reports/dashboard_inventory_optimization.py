import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# ==========================================
# 🌅 STREAMLIT CONFIG
# ==========================================
st.set_page_config(
    page_title="Inventory Optimization Dashboard 🌅",
    page_icon="📦",
    layout="wide"
)

# ==========================================
# 🎨 CUSTOM WARM SUNSET THEME (CSS)
# ==========================================
st.markdown("""
    <style>
    /* App background and text color */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #2C241F 0%, #1F1B16 90%);
        color: #FFF7E6;
    }

    /* Sidebar style */
    [data-testid="stSidebar"] {
        background: #2C241F;
        color: #FFF7E6;
    }

    /* Header bar */
    [data-testid="stHeader"] {
        background: #1F1B16;
        color: #FFF7E6;
    }

    /* All text */
    h1, h2, h3, h4, h5, p, label, span, div {
        color: #FFF7E6 !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #FF8A65;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #F06292;
        color: #FFF7E6;
    }

    /* Dataframe box */
    .stDataFrame {
        background-color: #2C241F !important;
        border-radius: 10px;
    }

    /* Titles and section headers */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 📂 LOAD DATA
# ==========================================
data_path = "reports/optimization/inventory_optimization_report.csv"
if not os.path.exists(data_path):
    st.error("❌ Optimization report not found. Please run the optimization script first.")
    st.stop()

df = pd.read_csv(data_path)

# ==========================================
# 🧭 SIDEBAR FILTERS
# ==========================================
st.sidebar.title("🎨 Dashboard Controls")
store = st.sidebar.selectbox("Select Store", sorted(df["store"].unique()))
item = st.sidebar.selectbox("Select Item", sorted(df["item"].unique()))
filtered = df[(df["store"] == store) & (df["item"] == item)]

# ==========================================
# 🏷️ HEADER
# ==========================================
st.markdown("<h1 style='text-align:center;'>Inventory Optimization</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Data-driven approach for smarter stock control</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #FFB74D; opacity:0.4;'>", unsafe_allow_html=True)

# ==========================================
# 🧮 SUMMARY CARDS
# ==========================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="🧮 Safety Stock", value=f"{filtered['safety_stock'].values[0]:,.2f}")
with col2:
    st.metric(label="📉 Reorder Point", value=f"{filtered['reorder_point'].values[0]:,.2f}")
with col3:
    st.metric(label="📦 Optimal Stock", value=f"{filtered['optimal_stock_level'].values[0]:,.2f}")

# ==========================================
# 📊 TABLE PREVIEW
# ==========================================
st.markdown("### 📋 Optimization Report")
st.dataframe(filtered)

# ==========================================
# 📈 VISUALIZATION
# ==========================================
st.markdown("### 📊 Inventory Stock Levels")

fig = go.Figure()
fig.add_bar(x=filtered["item"], y=filtered["safety_stock"],
            name="Safety Stock", marker_color="#FFB74D")
fig.add_bar(x=filtered["item"], y=filtered["reorder_point"],
            name="Reorder Point", marker_color="#FF8A65")
fig.add_bar(x=filtered["item"], y=filtered["optimal_stock_level"],
            name="Optimal Stock", marker_color="#F06292")

fig.update_layout(
    title=f"Inventory Optimization Overview — Store {store}",
    xaxis_title="Item",
    yaxis_title="Stock Level",
    barmode="group",
    template="plotly_white",
    font=dict(color="#FFF7E6"),
    paper_bgcolor="#1F1B16",
    plot_bgcolor="#1F1B16",
    legend=dict(bgcolor="#1F1B16", bordercolor="#FFB74D", borderwidth=1)
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 🧾 FOOTER
# ==========================================
st.markdown("<hr style='border: 1px solid #FFB74D; opacity:0.3;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#FFB74D; opacity:0.8;'>"
    "<b>Agnes Maria</b> — Data Science Service Dashboard 🌇</p>",
    unsafe_allow_html=True
)
