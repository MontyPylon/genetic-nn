import torch
import torch.nn.functional as F

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = torch.nn.Linear(3*64*64, 100)
        self.relu = torch.nn.ReLU()
        self.classify = torch.nn.Linear(100, 6)

        # self.network = torch.nn.Sequential(
        #     torch.nn.Linear(3 * 64 * 64, 100),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(100, 6),
        # )

    def forward(self, x):
        # flatten image
        z = x.view(x.size(0), -1)

        # pass through network
        z = self.relu(self.l1(z))
        z = self.classify(z)

        return z


def save_model(model):
    from torch import save
    from os import path
    test = type(model).__name__
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % type(model).__name__))


def load_model(model):
    from torch import load
    from os import path
    r = type(model)
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % type(model).__name__), map_location='cpu'))
    return r
