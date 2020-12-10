import torch
import torch.nn.functional as F
from os import path

class LinearModel(torch.nn.Module):
    def __init__(self, preset=None, top=None, k=None):
        super().__init__()

        self.num_neurons = 1000

        self.relu = torch.nn.ReLU()

        self.l1 = torch.nn.Linear(3*64*64, self.num_neurons)
        self.l2 = torch.nn.Linear(self.num_neurons, self.num_neurons)
        self.l3 = torch.nn.Linear(self.num_neurons, self.num_neurons)
        self.l4 = torch.nn.Linear(self.num_neurons, self.num_neurons)

        if preset is not None:
            for i in range(k):
                index = top[1][i].item()
                self.l1.weight.data[index] = preset.weight.data[index]
                self.l1.bias.data[index] = preset.bias.data[index]

        self.classify = torch.nn.Linear(self.num_neurons, 6)

    def forward(self, x):
        # flatten image
        z = x.view(x.size(0), -1)

        # pass through network
        z = self.relu(self.l1(z))
        z = self.relu(self.l2(z))
        z = self.relu(self.l3(z))
        z = self.relu(self.l4(z))
        z = self.classify(z)

        return z

class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        kernel_size = 5
        layer1 = 100

        self.conv1 = torch.nn.Conv2d(3, layer1, kernel_size)
        self.relu1 = torch.nn.ReLU()

        # self.conv2 = torch.nn.Conv2d(32, 64, kernel_size, stride=2, padding=1)
        # self.relu2 = torch.nn.ReLU()

        # self.conv3 = torch.nn.Conv2d(64, 128, kernel_size, stride=2, padding=2)
        # self.relu3 = torch.nn.ReLU()

        # self.conv4 = torch.nn.Conv2d(128, 256, kernel_size, stride=2, padding=2)
        # self.relu4 = torch.nn.ReLU()

        self.classify = torch.nn.Linear(layer1, 6)

    def forward(self, x):
        l1 = self.relu1(self.conv1(x))
        # l2 = self.relu2(self.conv2(l1))
        # l3 = self.relu3(self.conv3(l2))
        # l4 = self.relu4(self.conv4(l3))
        # l5 = self.classifier(l4.mean(dim=[2,3]))
        l2 = l1.mean(dim=[2,3])
        z = self.classify(l2)
        return z

class FCN(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, bias=False)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=6, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        #self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)
        self.classifier = torch.nn.Linear(c, n_output_channels)

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d'%i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d'%i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)

        z = z
        z = z.mean(dim=[2,3])
        return self.classifier(z)


def save_model(model):
    model_name = 'model_weights/%s.th' % type(model).__name__
    model_path = path.join(path.dirname(path.abspath(__file__)), model_name)
    torch.save(model.state_dict(), model_path)


def load_model(model):
    model_name = 'model_weights/%s.th' % type(model).__name__
    model_path = path.join(path.dirname(path.abspath(__file__)), model_name)
    model.load_state_dict(torch.load(model_path))
