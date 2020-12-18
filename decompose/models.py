import torch
import torch.nn.functional as F
from os import path

class ConvModel(torch.nn.Module):
    def __init__(self, width, digit_models=None):
        super().__init__()

        self.width = width
        kernel_size=3

        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(1, width, kernel_size)
        self.conv2 = torch.nn.Conv2d(width, width, kernel_size)
        self.conv3 = torch.nn.Conv2d(width, width, kernel_size)
        self.classify = torch.nn.Linear(width, 11)

    def forward(self, x):
        l1 = self.relu(self.conv1(x))
        l2 = self.relu(self.conv2(l1))
        l3 = self.relu(self.conv3(l2))
        l4 = l3.mean(dim=[2,3])
        z = self.classify(l4)
        return z

class BigConvModel(torch.nn.Module):
    def __init__(self, width, digit_models):
        super().__init__()

        self.width = width
        kernel_size=3

        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(1, width, kernel_size)
        self.conv2 = torch.nn.Conv2d(width, width, kernel_size, groups=10)
        self.conv3 = torch.nn.Conv2d(width, width, kernel_size, groups=10)
        self.classify = torch.nn.Linear(width, 11)

        for i in range(10):
            digit_model = digit_models[i]
            each_width = int(width / 10)
            for j in range(each_width):
                index = (i * each_width) + j
                self.conv1.weight.data[index] = digit_model.conv1.weight.data[j]
                self.conv1.bias.data[index] = digit_model.conv1.bias.data[j]
                self.conv2.weight.data[index] = digit_model.conv2.weight.data[j]
                self.conv2.bias.data[index] = digit_model.conv2.bias.data[j]
                self.conv3.weight.data[index] = digit_model.conv3.weight.data[j]
                self.conv3.bias.data[index] = digit_model.conv3.bias.data[j]


    def forward(self, x):
        l1 = self.relu(self.conv1(x))
        l2 = self.relu(self.conv2(l1))
        l3 = self.relu(self.conv3(l2))
        l4 = l3.mean(dim=[2,3])
        z = self.classify(l4)
        return z

def save_model(model):
    model_name = 'model_weights/%s.th' % type(model).__name__
    model_path = path.join(path.dirname(path.abspath(__file__)), model_name)
    torch.save(model.state_dict(), model_path)


def load_model(model):
    model_name = 'model_weights/%s.th' % type(model).__name__
    model_path = path.join(path.dirname(path.abspath(__file__)), model_name)
    model.load_state_dict(torch.load(model_path))
