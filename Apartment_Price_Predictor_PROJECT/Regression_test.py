import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

# read the file to memory
data = pd.read_csv(".\\Data\\data.csv")
data.dropna(inplace=True)

# # The user inputs details
# area = float(input("Area :"))
# metro = input("Metro :")
# building_type = input("Building type :")
#
# # Filter data
# filtered = data.copy()
# if metro:
#     filtered = filtered[filtered["Metro station"].str.contains(metro, case=False, na=False)]
# if building_type:
#     filtered = filtered[filtered["Apartment type"].str.contains(building_type, case=False, na=False)]
#
#
# total = len(filtered)
# split_point = int(round(total / 1.5))
#
# train_input = np.array(filtered.Area[:split_point])
# train_output = np.array(filtered.Price[:split_point])
#
# # Normalization
# X_mean, X_std = np.mean(train_input), np.std(train_input)
# Y_mean, Y_std = np.mean(train_output), np.std(train_output)
#
# train_input = (train_input - X_mean) / X_std
# train_output = (train_output - Y_mean) / Y_std
#
#
# class LinearRegression:
#     def __init__(self):
#         self.parameters = {"m": 0, "c": 0}
#         self.loss = []
#
#     def forward_propagation(self, x):
#         return self.parameters["m"] * x + self.parameters["c"]
#
#     def cost_function(self, prediction, y):
#         return np.mean((y - prediction) ** 2)
#
#     def backward_propagation(self, x, y, prediction):
#         df = prediction - y
#         dm = 2 * np.mean(df * x)
#         dc = 2 * np.mean(df)
#         return {"dm": dm, "dc": dc}
#
#     def update_parameters(self, derivatives, lr):
#         self.parameters["m"] -= lr * derivatives["dm"]
#         self.parameters["c"] -= lr * derivatives["dc"]
#
#     def train(self, x, y, lr=0.001, iters=1000):
#         for i in range(iters):
#             prediction = self.forward_propagation(x)
#             cost = self.cost_function(prediction, y)
#             grads = self.backward_propagation(x, y, prediction)
#             self.update_parameters(grads, lr)
#             self.loss.append(cost)
#             if i % 100 == 0:
#                 print(f"Iteration {i}, Loss = {cost:.5f}")
#         return self.parameters, self.loss
#
#     def predict_price(self, val, X_mean, X_std, Y_mean, Y_std):
#         scaled_area = (val - X_mean) / X_std
#         scaled_pred = self.parameters["m"] * scaled_area + self.parameters["c"]
#         return scaled_pred * Y_std + Y_mean
#
#
# # Train model
# lin_reg = LinearRegression()
# parameters, losses = lin_reg.train(train_input, train_output, lr=0.001, iters=1000)
#
# # Prediction
# predicted = lin_reg.predict_price(area, X_mean, X_std, Y_mean, Y_std)
# print("Predicted price:", predicted)


# plot a graph of metro against their respective average price
data["Average price per metro"]=data.groupby(data["Metro station"])["Price(₽)"].mean()
print(data["Average price per metro"])



# plt.scatter(train_input, train_output, color='blue', label="Data (normalized)")
# x_vals = np.linspace(min(train_input), max(train_input), 100)
# plt.plot(x_vals, lin_reg.forward_propagation(x_vals), color='red', label="Regression line")
#
# plt.xlabel("Area (normalized)")
# plt.ylabel("Price (normalized)")
#
# plt.legend()
# plt.show()