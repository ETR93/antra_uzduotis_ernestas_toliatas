import torch
from helpers import load_model
from ControlNN import ControlNN


class Main:
    def __init__(self):
        pass

    def predict_x_values(self):
        y_target_test = int(input("Input target value: "))
        n_steps = int(input("Input number of values to generate: "))

        model = load_model(ControlNN())

        y_prev2, y_prev1 = y_target_test, y_target_test
        x_prev = 0.0
        z_prev = 0.0

        x_values = []

        for _ in range(n_steps):
            nn_input = torch.tensor([[y_prev2, y_prev1, z_prev, x_prev, y_target_test]], dtype=torch.float32)
            x_next = model(nn_input).item()
            x_values.append(x_next)

            z_prev = z_prev + 2 * x_next + 0.11

            y_prev2, y_prev1 = y_prev1, y_target_test
            x_prev = x_next

        for i in x_values:
            print(f"X value for y={y_target_test}: {i}")


if __name__ == "__main__":
    Main().predict_x_values()