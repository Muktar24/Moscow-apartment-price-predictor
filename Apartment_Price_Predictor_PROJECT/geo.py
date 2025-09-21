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
data["apartment price"] = data.groupby("Metro station")["Price(₽)"].transform("mean")

print(data["apartment price"].to_string())