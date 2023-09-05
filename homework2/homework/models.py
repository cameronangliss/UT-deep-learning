import torch


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 10, 7),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 12, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(12, 15, 3),
            torch.nn.MaxPool2d(3),
            torch.nn.Linear(18, 6),
        )

    def forward(self, x):
        return self.layers(x)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
