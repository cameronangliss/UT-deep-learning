import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        return F.cross_entropy(input, target)


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(64*64*3, 6)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        x = self.layer(x)
        return x


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(64*64*3, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        x = self.model(x)
        return x


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
