import torch
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    mean = torch.mean(weights.view(-1))
    values, _ = torch.topk(weights.view(-1), k=10)
    largest_vals_mean = torch.mean(values)
    print("mean:", mean)
    print("largest mean:", largest_vals_mean)
    # indicating if the puck is not seen
    # if torch.max(weights) < 0.04:
    #     return None
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Detector(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.n_input = n_input
            self.n_output = n_output
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(inplace=True),
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
        """
        Your code here
        """

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
        self.final_conv = torch.nn.Conv2d(layers[0], 1, kernel_size=1)
        self.network_chain = torch.nn.Sequential(*self.down_blocks, *self.up_convs, *self.up_blocks)

    def forward(self, x):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
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
        x = self.final_conv(x)[:, 0, :, :]
        # print("final", x.size())
        return x

    def detect(self, x):
        x = self.forward(x[None])
        detections = spatial_argmax(x)
        if detections is None:
            return None
        else:
            return detections[0]


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Detector):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


# if __name__ == '__main__':
#     from .controller import control
#     from .utils import PyTux
#     from argparse import ArgumentParser


#     def test_planner(args):
#         # Load model
#         planner = load_model().eval()
#         pytux = PyTux()
#         for t in args.track:
#             steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
#             print(steps, how_far)
#         pytux.close()


#     parser = ArgumentParser("Test the planner")
#     parser.add_argument('track', nargs='+')
#     parser.add_argument('-v', '--verbose', action='store_true')
#     args = parser.parse_args()
#     test_planner(args)
