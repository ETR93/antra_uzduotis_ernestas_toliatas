import numpy as np
import torch

def assign_starting_values(number_of_training_samples, target_train):

    x_data = np.zeros(number_of_training_samples)
    y_data = np.zeros(number_of_training_samples)
    z_data = np.zeros(number_of_training_samples)

    y_data[0], y_data[1] = target_train, target_train
    z_data[0] = 0.0
    x_data[0] = 0.0

    return y_data, z_data, x_data

def generate_train_data(training_samples, starting_values, y_target_train):
    # y starting_values[0]
    # z starting_values[1]
    # x starting_values[2]

    X_train = []
    Y_train = []

    for i in range(2, training_samples):
        x_next = (0.3 * starting_values[2][i - 2] - 0.01 * starting_values[0][i - 2] - 0.1 * starting_values[1][i - 1]) / 8
        starting_values[2][i - 1] = x_next
        starting_values[1][i] = starting_values[1][i - 1] + 2 * x_next + 0.11
        starting_values[0][i] = y_target_train
        X_train.append([starting_values[0][i - 2], starting_values[0][i - 1], starting_values[1][i - 1],
                        starting_values[2][i - 2], y_target_train])
        Y_train.append([x_next])

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    return X_train, Y_train

def save_model(train_data, optimizer, model, criterion):

    num_epochs = int(input("Input the number of epochs: "))

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(train_data[0])
        loss = criterion(y_pred, train_data[1])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "control_nn.pth")

def load_model(modelAI):
    model = modelAI
    model.load_state_dict(torch.load("control_nn.pth"))
    model.eval()

    return model