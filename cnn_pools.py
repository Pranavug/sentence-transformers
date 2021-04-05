import torch
from torch import nn
import torch.nn.functional as F
import os
import json


class CNNSmall(nn.Module):
    def __init__(self):
        super(CNNSmall, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size = (5, 1), stride = (3, 1), padding = (1, 0))
        # self.pool1 = torch.nn.MaxPool2d(kernel_size = (2, 1), stride = (2, 1), padding = 0)

        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size = (5, 1), stride = (3, 1), padding = (1, 0))
        # self.pool2 = torch.nn.MaxPool2d(kernel_size = (2, 1), stride = (2, 1), padding = 0)

        self.conv4 = torch.nn.Conv2d(32, 1, kernel_size = (5, 1), stride = (3, 1), padding = (1, 0))
        # self.pool4 = torch.nn.MaxPool2d(kernel_size = (4, 1), stride = (2, 1), padding = 0)

    def forward(self, inp):
        # print("CNN-small forward")
        x = inp['token_embeddings']
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        # x = self.pool1(x)

        x = F.relu(self.conv2(x))
        # x = self.pool2(x)

        # x = F.relu(self.conv3(x))
        # x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = x.squeeze()
        # x = self.pool4(x)

        inp.update({'sentence_embedding': x})
        return inp

    def save(self, output_path: str):
        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    @staticmethod
    def load(input_path: str):
        weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        model = CNNSmall()
        model.load_state_dict(weights)
        return model


class CNNLarge(nn.Module):
    def __init__(self):
        super(CNNLarge, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size = (1, 1), stride = 1, padding = (1, 0))
        self.pool1 = torch.nn.MaxPool2d(kernel_size = (2, 1), stride = (2, 1), padding = 0)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size = (5, 1), stride = 1, padding = (1, 0))
        self.pool2 = torch.nn.MaxPool2d(kernel_size = (2, 1), stride = (2, 1), padding = 0)

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size = (3, 1), stride = 1, padding = (1, 0))
        self.pool3 = torch.nn.MaxPool2d(kernel_size = (3, 1), stride = (2, 1), padding = 0)

        self.conv4 = torch.nn.Conv2d(128, 1, kernel_size = (3, 1), stride = 1, padding = (1, 0))
        self.pool4 = torch.nn.MaxPool2d(kernel_size = (4, 1), stride = (2, 1), padding = 0)

    def forward(self, x):
        print("CNN-large forward")
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        return x

    def save(self, output_path: str):
        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    @staticmethod
    def load(input_path: str):
        weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        model = CNNLarge()
        model.load_state_dict(weights)
        return model


if __name__ == '__main__':
    model = CNNLarge()
    sample_input = torch.zeros((16, 50, 768))
    sample_input = sample_input.unsqueeze(1)

    print(sample_input.shape)

    output = model(sample_input)

    print(output.shape)
    output = output.squeeze()

    print(output.shape)
