import os
from .detector import Detector, save_model, CNNClassifier
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms
import matplotlib.pyplot as plt


def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    # create a model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Added fort Kyle's mac -> switch to mps if available
    if torch.backends.mps.is_available():
        print("swapped to mps")
        device = torch.device("mps")
    model = Detector().to(device)
    if os.path.exists("image_agent/det.th"):
        print("Loading saved model...")
        model.load_state_dict(torch.load("image_agent/det.th", map_location="cpu"))
        print("Done!")
    coord_loss = torch.nn.BCEWithLogitsLoss()
    bools_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # load the data: train and valid
    train_transform = dense_transforms.Compose(
        [
            dense_transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25
            ),
            dense_transforms.RandomHorizontalFlip(),
            dense_transforms.ToTensor(),
            dense_transforms.ToHeatmap(),
        ]
    )
    train_data = load_data("drive_data", transform=train_transform)
    # valid_data = load_data("drive_data")

    # Run SGD for several epochs
    global_step = 0

    for epoch in range(args.n_epochs):
        avg_error = 0
        i = 0
        for batch in train_data:
            images = batch[0].to(device)
            heatmaps = batch[1][:, 0, :, :].to(device)
            bools = batch[2].to(device)
            model_output = model.forward(images)
            hm_outputs = model_output[:,:1]
            bool_outputs = model_output[:,1:]
            hm_loss = coord_loss.forward(hm_outputs, heatmaps)
            bool_loss = bools_loss.forward(bool_outputs, bools)
            #train_logger.add_scalar("loss", train_error, global_step=global_step)
            optimizer.zero_grad()
            train_error  = 0.5*hm_loss + 0.5*bool_loss
            train_error.backward()
            optimizer.step()
            global_step += 1
            i += 1
            avg_error += (1 / i) * (train_error.item() - avg_error)
        # plt.imsave("image.png", images[0, 0, :, :].cpu(), cmap="gray")
        # plt.imsave("label.png", heatmaps[0].cpu(), cmap="gray")
        print(f"Epoch {epoch + 1} training error:", avg_error)

    save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--n_epochs', default=10, type=int)

    args = parser.parse_args()
    train(args)
