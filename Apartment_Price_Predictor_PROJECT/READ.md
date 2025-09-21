# 🏠 Apartment Price Predictor (Streamlit + Custom Linear Regression)

Welcome to the **Apartment Price Prediction App**!  
This project combines **data science** 📊, **machine learning** 🤖, and an **interactive Streamlit web app** 🌐 to predict **apartment prices in Moscow** based on **area, metro station, and building type**.

Unlike typical ML projects that rely on libraries like scikit-learn, this app uses a **custom-built Linear Regression model implemented from scratch with NumPy** 🔢. This makes it not just a prediction tool, but also a **learning project** for anyone interested in understanding **how machine learning algorithms work internally**.

---

## 📖 Background & Motivation

Moscow is one of the most dynamic real estate markets in the world. Apartment prices vary greatly depending on:
- 🚇 **Metro station proximity**
- 📐 **Apartment area**
- 🏢 **Building type** (new vs. secondary)

This app helps visualize and predict these price patterns. Beyond practical use, it also serves as an **educational resource** for:
- Students learning **Linear Regression**
- Developers practicing **Streamlit app development**
- Data scientists exploring **data preprocessing and visualization**

---

## ✨ Features
- 📂 Load and filter real estate data by **metro station** and **building type**
- 🏙️ Interactive **Moscow map** with sample apartment points
- 🔢 **Custom Linear Regression** implementation (no external ML libraries)
- 📉 Visualization of training **loss curve**
- 📊 Regression **fit line** plotting
- 🎛️ User-friendly **Streamlit sidebar inputs**
- 💰 Real-time **price predictions in RUB**

---

## 🛠️ Technical Stack

- **Python 3.8+**
- **Streamlit** – Interactive web app framework
- **NumPy** – Numerical computations
- **Pandas** – Data handling
- **Matplotlib** – Visualizations
- **Custom Linear Regression Class** – Implemented manually

---

## 📂 Project Structure

+ 📦 Apartment Price Predictor
-    ├── Data/
+    │ └── data.csv # Apartment dataset
 *   ├── LinearRegression.py # Custom linear regression class
  +  ├── main.py # Streamlit app
   - ├── requirements.txt # Dependencies
  *  └── README.md # Project documentation

**In the sidebar:**

+ Enter apartment area (m²)

- Select metro station

* Choose building type (Secondary / New building)

- Click Run Prediction

**View results:**

Filtered dataset preview

+ 🗺️ Moscow map with points

- 📉 Loss curve showing model training

* 📊 Regression line vs. data points

- 💰 Predicted apartment price

**🔬 How Linear Regression Works**

**Linear Regression** is a simple and widely used supervised _machine learning algorithm_. It predicts a continuous output variable (y) from an input variable (x).

+ ## 1. Intuition

_The app predicts apartment prices based on area, metro station, and building type_

Linear Regression assumes a linear relationship between input and output:

**Price = m * Area + c**


**m** is the slope, showing how much the price increases per square meter

**c** is the intercept, representing the baseline price when area is zero

## 2. Mathematical Formulation

### Hypothesis function:

**ŷ = m * x + c**


### Cost function (Mean Squared Error, MSE):

J(m,c) = (1/n) * Σ(ŷ_i - y_i)<sup>2</sup>


_Measures how far predictions are from actual values_

**Goal:** minimize this cost

### Gradients for Gradient Descent:

+ ∂J/∂m = (2/n) * Σ((ŷ_i - y_i) * x_i)
- ∂J/∂c = (2/n) * Σ(ŷ_i - y_i)

## 3. Gradient Descent

Iteratively update parameters **m** and **c**:

+ m = m - α * ∂J/∂m
- c = c - α * ∂J/∂c


+ α = learning rate (step size)

_Repeat until the loss converges to a minimum_

## 4. Normalization

Inputs (area) and outputs (price) are scaled using mean and standard deviation:

**x_scaled = (x - mean) / std**
**y_scaled = (y - mean) / std**


This improves gradient descent convergence and training stability

## 5. Prediction

Once trained, the model predicts new apartment prices:

**Predicted Price = m * Scaled Area + c**


The scaled prediction is converted back to the original price scale to get the final apartment price in RUB

📊 Example Run

User input:

**Area:** 70 m²

**Metro station:** Belorusskaya

**Building type:** Secondary

**Model output:**

Estimated price: 12,345,678.90 RUB

🖼️ Screenshots (Placeholder)

Add screenshots here after running the app 🚀

✅ App homepage

✅ Sidebar inputs

✅ Prediction results

✅ Loss curve and regression plot

# 🚀 Future Improvements

✅ **Add additional features (floor, renovation type, minutes to metro)**

**✅ Extend to multiple regression instead of simple regression**

**✅ Improve Moscow map visualization with actual geo-data**

**✅ Deploy online via Streamlit Cloud / Heroku**

**✅ Add model evaluation metrics (R², MAE, RMSE)**

## **🎯 Educational Value**

### This project is ideal for:

**🧑‍🎓 Students learning ML basics**

**🧑‍💻 Developers practicing Streamlit apps**

**📊 Data scientists who want a quick reference on implementing models from scratch**

# 🤝 Contributing

+ Contributions are welcome!

- Fork the repo

* Create a new branch (feature/new-feature)

+ Commit your changes

- Open a pull request 🚀



# 👨‍💻 Made with ❤️ by Muktar


---

