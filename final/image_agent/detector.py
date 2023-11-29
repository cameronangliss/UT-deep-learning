import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Detector(torch.nn.Module):
    def __init__(self, layers=[32, 64, 128, 256], n_input_channels=3):
        """
        Your code here
        """

        super().__init__()
        n_input_channels=3
        n_output_channels=3
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down4 = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            #nn.ConvTranspose2d(128,128//2,2,2)
            nn.Conv2d(256,128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            #nn.ConvTranspose2d(128,128//2,2,2)
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            #nn.ConvTranspose2d(128,128//2,2,2)
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(2)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.u1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, stride=2)
        )
        self.u2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2)
        )
        self.u3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, stride=2)
        )
        self.u4 = nn.Sequential(
            nn.ConvTranspose2d(32,32,2,stride=2)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
        )
        
        
    def forward(self, x):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """

        x = img
        b,a,h,w = x.shape
        xd1 = self.down1(x)
        xd2 = xd1
        b,a,h1,w1 = xd1.shape
        if (h1 > 1 and w1) > 1:
            xd2 = self.maxpool(xd2)
        xd2 = self.down2(xd2)
        xd3 = xd2
        b,a,h2,w2 = xd2.shape
        if (h2 > 1 and w2 > 1):
            xd3 = self.maxpool(xd3)
        xd3 = self.down3(xd3)
        xd4 = xd3
        b,a,h3,w3 = xd3.shape
        if (h3 > 1 and w3 > 1):
            xd4 = self.maxpool(xd4)
        xd4 = self.down4(xd4)
        b,a,h4,w4 = xd4.shape
        x = xd4
        x = self.up1(x)
        if (h3 > 1 and w3 > 1):
            x = self.u2(x)
        x = torch.cat([x,xd3], dim=1)  
        x = self.up2(x)
        if (h2 > 1 and w2 > 1):
            x = self.u3(x)
        x = torch.cat([x,xd2], dim=1)        
        x = self.up3(x)
        if (h1 > 1 and w1) > 1:
            x = self.u4(x)
        x = torch.cat([x,xd1], dim=1)
        x = self.up4(x)
        hm = x[:, 0, :, :]
        ret = spatial_argmax(hm)
        return ret


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
