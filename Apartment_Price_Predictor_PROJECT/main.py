import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
from geopy.geocoders import Nominatim
import time

# Load data
data = pd.read_csv("./Data/data.csv")
data.dropna(inplace=True)

#Web app title and description
st.set_page_config(page_title="Apartment Price Prediction", layout="wide")

st.title("🏠 Apartment Price Predictor (Data Science Focused)")
st.write("This app predicts apartment prices using **Linear Regression** and "
         "also explains how the model works, including visualizations and metrics in moscow by using the respective metro stations,area and building type")

# Sidebar inputs
st.sidebar.header("User Input")
area = st.sidebar.number_input("Area (sq meters)", min_value=10.0, max_value=500.0, value=50.0)
metro = st.sidebar.selectbox("Metro station",(data["Metro station"].unique()))
building_type = st.sidebar.selectbox("Building type",("Secondary","New building"))
predict_btn = st.sidebar.button("Run Prediction")

# Filter dataset
filtered = data.copy()
if metro:
    filtered = filtered[filtered["Metro station"].str.contains(metro, case=False, na=False)]
if building_type:
    filtered = filtered[filtered["Apartment type"].str.contains(building_type, case=False, na=False)]
# Show a dataframe/table of the filtered data
st.subheader("Filtered Data Preview")
st.dataframe(filtered.head())

# A map of the metro station with the apartment available of the metro inputed
geolocator=Nominatim(user_agent="Apartment Price Predictor")
location = geolocator.geocode(metro)
map_data=pd.DataFrame(np.random.randn(len(filtered),2)/[50,50] + [location.latitude, location.longitude],
                      columns=['lat','lon'])

st.map(map_data)


# Train data using linear regression algorithm
if not filtered.empty and predict_btn:
    total = len(filtered)
    split_point = int(round(total / 1.5))

    train_input = np.array(filtered["Area(m²)"][:split_point])
    train_output = np.array(filtered["Price(₽)"][:split_point])

    # Normalization
    X_mean, X_std = np.mean(train_input), np.std(train_input)
    Y_mean, Y_std = np.mean(train_output), np.std(train_output)

    train_input = (train_input - X_mean) / X_std
    train_output = (train_output - Y_mean) / Y_std

    # Train model using linear regression algorithm
    lin_reg = LinearRegression()
    parameters, losses = lin_reg.train(train_input, train_output, lr=0.001, iters=2000)

    # Predict the price of the apartment given the area
    predicted = lin_reg.predict_price(area, X_mean, X_std, Y_mean, Y_std)
    st.subheader("💰 Predicted Price")
    st.success(f"Estimated price: {predicted:,.2f} RUB")


    # Show a pictorial representation of a loading data
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        bar.progress((i + 1))
        time.sleep(0.05)
        latest_iteration.text(f"Loading graphs {i + 1}")



    # Show loss chart using the loss function of the Linear regression algorithm
    st.subheader("Training Loss")
    st.line_chart(losses)

    # Show regression plot
    st.subheader("Regression Fit")
    fig, ax = plt.subplots()
    ax.scatter(train_input, train_output, color='blue', label="Data (normalized)")
    x_vals = np.linspace(min(train_input), max(train_input), 100)
    ax.plot(x_vals, lin_reg.forward_propagation(x_vals), color='red', label="Regression line")
    ax.set_xlabel("Area (normalized)")
    ax.set_ylabel("Price (normalized)")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Please enter inputs and click **Run Prediction** to see results.")


