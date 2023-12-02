import os
from .detector import Detector, save_model, CNNClassifier
import torch
from torchvision import transforms
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data


def cnntrain(args):
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
    model = CNNClassifier().to(device)
    if os.path.exists("homework/cnndet.th"):
        print("Loading saved model...")
        model.load_state_dict(torch.load("homework/cnndet.th"))
        print("Done!")
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # load the data: train and valid
    train_transform = transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #dense_transforms.ToHeatmap(),
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
            #print(batch[2])
            labels = batch[2].to(device)
            model_output = model.forward(images)
            train_error = loss.forward(model_output, labels)
            train_logger.add_scalar("loss", train_error, global_step=global_step)
            optimizer.zero_grad()
            train_error.backward()
            optimizer.step()
            global_step += 1
            i += 1
            avg_error += (1 / i) * (train_error.item() - avg_error)
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
    cnntrain(args)
