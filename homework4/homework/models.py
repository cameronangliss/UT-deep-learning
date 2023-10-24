import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100) -> list[tuple[float, int, int]]:
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_pool = torch.nn.MaxPool2d(max_pool_ks, stride=1, padding=max_pool_ks//2).to(device)
    max_vals = max_pool(heatmap[None, None])
    dets = []
    for y in range(heatmap.size()[0]):
        for x in range(heatmap.size()[1]):
            if len(dets) == max_det:
                return dets
            if heatmap[y, x] == max_vals[0, 0, y, x] and heatmap[y, x] > min_score:
                dets += [(heatmap[y, x], x, y)]
    return dets


class Detector(torch.nn.Module):
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
        """
           Your code here.
           Setup your detection network
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
        self.final_conv = torch.nn.Conv2d(layers[0], 3, kernel_size=1)
        self.network_chain = torch.nn.Sequential(*self.down_blocks, *self.up_convs, *self.up_blocks)

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
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

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """

        heatmaps = self.forward(image)[0]
        kart_peaks = [(score, cx, cy, 0, 0) for score, cx, cy in extract_peak(heatmaps[0], max_det=30)]
        bomb_peaks = [(score, cx, cy, 0, 0) for score, cx, cy in extract_peak(heatmaps[1], max_det=30)]
        pickup_peaks = [(score, cx, cy, 0, 0) for score, cx, cy in extract_peak(heatmaps[2], max_det=30)]
        return [kart_peaks, bomb_peaks, pickup_peaks]


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
