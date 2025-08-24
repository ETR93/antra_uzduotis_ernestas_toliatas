import torch.nn as nn
import torch.optim as optim
from ControlNN import ControlNN
from helpers import assign_starting_values, generate_train_data, save_model

class ModelTrain:
    def __init__(self):
        pass

    def train_model(self):
        N_train = int(input("Input the number of training samples: "))
        y_target_train = int(input("Input the target value: "))
        print("------------------------------------------")
        print("Generating model... ")
        starting_values = assign_starting_values(N_train, y_target_train)

        train_data = generate_train_data(N_train, starting_values, y_target_train)

        model = ControlNN()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        print("------------------------------------------")
        print("Saving model.. ")
        save_model(train_data, optimizer, model, criterion)
        print("------------------------------------------")
        print("Model saved. ")


if __name__ == "__main__":
    ModelTrain().train_model()