import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


class NeuralNet(nn.Module):
    def __init__(self, input_size, hs1, hs2, hs3, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hs1)
        self.fc2 = nn.Linear(hs1, hs2)
        self.fc3 = nn.Linear(hs2, hs3)
        self.fc4 = nn.Linear(hs3, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.input_dropout = nn.Dropout(0.25)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        return x


def get_random_forest_classifier(seed):
    return RandomForestClassifier(random_state=seed)
