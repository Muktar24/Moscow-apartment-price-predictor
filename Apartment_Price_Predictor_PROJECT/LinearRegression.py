import numpy as np

class LinearRegression:
    def __init__(self):
        self.parameters = {"m": 0, "c": 0}
        self.loss = []

    # A forward regression that uses the forward regression formula
    def forward_propagation(self, x):
        return self.parameters["m"] * x + self.parameters["c"]

    def cost_function(self, prediction, y):
        return np.mean((y - prediction) ** 2)

    def backward_propagation(self, x, y, prediction):
        df = prediction - y
        dm = 2 * np.mean(df * x)
        dc = 2 * np.mean(df)
        return {"dm": dm, "dc": dc}

    def update_parameters(self, derivatives, lr):
        self.parameters["m"] -= lr * derivatives["dm"]
        self.parameters["c"] -= lr * derivatives["dc"]

    def train(self, x, y, lr=0.001, iters=1000):
        for i in range(iters):
            prediction = self.forward_propagation(x)
            cost = self.cost_function(prediction, y)
            grads = self.backward_propagation(x, y, prediction)
            self.update_parameters(grads, lr)
            self.loss.append(cost)
            if i % 100 == 0:
                print(f"Iteration {i}, Loss = {cost:.5f}")
        return self.parameters, self.loss

    def predict_price(self, val, X_mean, X_std, Y_mean, Y_std):
        scaled_area = (val - X_mean) / X_std
        scaled_pred = self.parameters["m"] * scaled_area + self.parameters["c"]
        return scaled_pred * Y_std + Y_mean


