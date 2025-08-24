
import torch.nn as nn

class ControlNN(nn.Module):
    def __init__(self):
        super(ControlNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.fc(x)