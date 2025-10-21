import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
from geopy.geocoders import Nominatim
import time

# --- CONFIG ---
st.set_page_config(page_title="🏠 Apartment Price Prediction", page_icon="🏙️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #2c3e50, #4ca1af);
        color: white;
    }
    .main-title {
        text-align: center;
        color: #00FFFF;
        font-size: 38px !important;
        text-shadow: 2px 2px 10px rgba(0,255,255,0.5);
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background: rgba(0,0,0,0.8);
        color: white;
        text-align: center;
        padding: 8px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.markdown("<h1 class='main-title'>🏠 Apartment Price Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>An interactive machine learning app that predicts apartment prices in Moscow using Linear Regression and metro-based geolocation insights.</p>", unsafe_allow_html=True)
st.divider()

# --- LOAD DATA ---
data = pd.read_csv("data\data.csv")
data.dropna(inplace=True)

# --- SIDEBAR INPUTS ---
st.sidebar.header("🧩 Input Apartment Details")
area = st.sidebar.number_input("Area (m²)", min_value=10.0, max_value=500.0, value=50.0)
metro = st.sidebar.selectbox("Metro Station", data["Metro station"].unique())
building_type = st.sidebar.selectbox("Building Type", ("Secondary", "New building"))
predict_btn = st.sidebar.button("🚀 Run Prediction")

# --- FILTER DATA ---
filtered = data.copy()
if metro:
    filtered = filtered[filtered["Metro station"].str.contains(metro, case=False, na=False)]
if building_type:
    filtered = filtered[filtered["Apartment type"].str.contains(building_type, case=False, na=False)]

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["💰 Prediction", "📊 Data Insights", "📈 Visualizations", "ℹ️ About Project"])

# --- TAB 1: PREDICTION ---
with tab1:
    st.subheader("Prediction Results")

    if not filtered.empty and predict_btn:
        total = len(filtered)
        split_point = int(round(total / 1.5))

        train_input = np.array(filtered["Area(m²)"][:split_point])
        train_output = np.array(filtered["Price(₽)"][:split_point])

        # --- Normalize ---
        X_mean, X_std = np.mean(train_input), np.std(train_input)
        Y_mean, Y_std = np.mean(train_output), np.std(train_output)
        train_input = (train_input - X_mean) / X_std
        train_output = (train_output - Y_mean) / Y_std

        # --- Train Model ---
        lin_reg = LinearRegression()
        parameters, losses = lin_reg.train(train_input, train_output, lr=0.001, iters=2000)

        # --- Prediction ---
        predicted = lin_reg.predict_price(area, X_mean, X_std, Y_mean, Y_std)

        # --- Display ---
        st.success(f"🏢 Estimated Apartment Price: **{predicted:,.2f} ₽**")
        st.metric("📏 Area (m²)", area)
        st.metric("📍 Metro Station", metro)
        st.metric("🏗️ Building Type", building_type)

        # --- Loading Animation ---
        progress_text = st.empty()
        bar = st.progress(0)
        for i in range(100):
            bar.progress(i + 1)
            time.sleep(0.02)
            progress_text.text(f"Analyzing market trends... {i+1}%")

        st.divider()

        # --- Loss Chart ---
        st.subheader("📉 Training Loss Curve")
        st.line_chart(losses)

        # --- Regression Fit ---
        st.subheader("📈 Regression Line Fit")
        fig, ax = plt.subplots()
        ax.scatter(train_input, train_output, color='blue', label="Data (normalized)")
        x_vals = np.linspace(min(train_input), max(train_input), 100)
        ax.plot(x_vals, lin_reg.forward_propagation(x_vals), color='red', label="Regression Line")
        ax.set_xlabel("Area (normalized)")
        ax.set_ylabel("Price (normalized)")
        ax.legend()
        st.pyplot(fig)

        # --- Download Report ---
        report = {
            "Area (m²)": area,
            "Metro Station": metro,
            "Building Type": building_type,
            "Predicted Price (₽)": predicted
        }
        report_df = pd.DataFrame([report])
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Price Report",
            data=csv,
            file_name="apartment_price_prediction.csv",
            mime="text/csv"
        )

    else:
        st.warning("Please input apartment details and click **Run Prediction**.")

# --- TAB 2: DATA INSIGHTS ---
with tab2:
    st.subheader("Filtered Data Preview")
    st.dataframe(filtered.head(10))

    st.info(f"📊 Showing data for **{metro}**, building type **{building_type}**, total {len(filtered)} entries.")
    st.bar_chart(filtered["Price(₽)"].head(20))

# --- TAB 3: VISUALIZATION ---
with tab3:
    st.subheader(f"🗺️ Location of Apartments near {metro}")

    try:
        geolocator = Nominatim(user_agent="Apartment Price Predictor")
        location = geolocator.geocode(metro)
        if location:
            map_data = pd.DataFrame(
                np.random.randn(len(filtered), 2) / [50, 50] + [location.latitude, location.longitude],
                columns=['lat', 'lon']
            )
            st.map(map_data)
        else:
            st.error("Could not find location data for this metro station.")
    except Exception:
        st.warning("⚠️ Map data could not be loaded (check internet connection).")

# --- TAB 4: ABOUT PROJECT ---
with tab4:
    st.markdown("""
        ### 🧠 About This Project
        **Project Title:** Apartment Price Prediction  
        **Developer:** *Muktar Sanusi*  
        **Institution:** Moscow Aviation Institute (MAI)  

        **Description:**  
        This app uses a **custom Linear Regression algorithm** to estimate apartment prices based on area, metro proximity, and building type in Moscow.  
        It is designed to demonstrate **data preprocessing**, **machine learning training**, and **geospatial visualization** in a real-estate context.

        **Key Features:**
        - Real-time apartment price prediction  
        - Interactive map of selected metro area  
        - Data visualization and training loss plots  
        - Downloadable prediction reports  
    """)

# --- FOOTER ---
st.markdown("""
    <div class="footer">
        © 2025 Muktar Sanusi | Data Science & Machine Learning Engineer | Moscow Aviation Institute (MAI)
    </div>
""", unsafe_allow_html=True)
