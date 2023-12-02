import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
            )
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(n_output)
            )

        def forward(self, x):
            return self.layers(x) + self.downsample(x)

    def __init__(self, layers=[16, 32, 64], n_input_channels=3):
        super().__init__()
        c = layers[0]
        L = [
            torch.nn.Conv2d(n_input_channels, c, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        for l in layers:
            L.append(self.Block(c, l, stride=2))
            c=l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, 1)

    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)


class FCN(torch.nn.Module):
    """
    Your code here.
    Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
    Hint: Use up-convolutions
    Hint: Use skip connections
    Hint: Use residual connections
    Hint: Always pad by kernel_size / 2, use an odd kernel_size
    """
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.n_input = n_input
            self.n_output = n_output
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
            )
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(n_output)
            )

        def forward(self, x):
            if x.size()[0] == x.size()[2] == x.size()[3] == 1:
                if self.n_output > self.n_input:
                    zeros = torch.zeros(1, self.n_output - self.n_input, 1, 1)
                    return torch.cat([zeros, x], dim=1)
                else:
                    return x[:, :self.n_output, :, :]
            else:
                return self.layers(x) + self.downsample(x)

    def __init__(self, layers=[32, 64, 128, 256], n_input_channels=3):
        super().__init__()
        self.down_blocks = []
        c = n_input_channels
        for l in layers:
            self.down_blocks.append(self.Block(c, l))
            c = l
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.up_convs = []
        for l in reversed(layers[1:]):
            self.up_convs.append(torch.nn.ConvTranspose2d(l, l, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.up_blocks = []
        rev_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            self.up_blocks.append(self.Block(rev_layers[i] + rev_layers[i + 1], rev_layers[i + 1]))
        self.final_conv = torch.nn.Conv2d(layers[0], 5, kernel_size=1)
        self.network_chain = torch.nn.Sequential(*self.down_blocks, *self.up_convs, *self.up_blocks)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """

        activations = []
        # print("start", x.size())
        for block in self.down_blocks[:-1]:
            x = block(x)
            # print("side", x.size())
            activations.append(x)
            x = self.pool(x)
            # print("down", x.size())
        x = self.down_blocks[-1](x)
        # print("side", x.size())
        rev_acts = list(reversed(activations))
        for i in range(len(self.up_blocks)):
            x = self.up_convs[i](x)
            # print("up", x.size())
            H = rev_acts[i].size()[2]
            W = rev_acts[i].size()[3]
            x = torch.cat([x[:, :, :H, :W], rev_acts[i]], dim=1)
            # print("cat", x.size())
            x = self.up_blocks[i](x)
            # print("side", x.size())
        x = self.final_conv(x)
        # print("final", x.size())
        return x


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
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
