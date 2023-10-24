import torch
import numpy as np

from .models import Detector, save_model
from .utils import PR, load_detection_data, point_close
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, "valid"), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """

    # create a model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Detector().to(device)
    # model.load_state_dict(torch.load("homework/det.th"))
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # load the data: train and valid
    transform = dense_transforms.Compose(
        [
            dense_transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25
            ),
            dense_transforms.RandomHorizontalFlip(),
            dense_transforms.ToTensor(),
            dense_transforms.ToHeatmap(),
        ]
    )
    train_data = load_detection_data("dense_data/train", transform=transform)
    valid_data = load_detection_data("dense_data/valid")

    # Run SGD for several epochs
    gs = 0
    while True:
        pr_box = [PR() for _ in range(3)]
        pr_dist = [PR(is_close=point_close) for _ in range(3)]
        for batch in train_data:
            image = batch[0].to(device)
            heatmaps = batch[1].to(device)
            detections = model.detect(image)
            for i in range(3):
                pr_box[i].add(detections[i], heatmaps[:, i, :, :].detach().cpu().numpy())
                pr_dist[i].add(detections[i], heatmaps[:, i, :, :].detach().cpu().numpy())
            model_output = model.forward(image)
            error = loss.forward(model_output, heatmaps)
            train_logger.add_scalar("loss", error, global_step=gs)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
            gs += 1
        train_logger.add_scalar("PiB kart", pr_box[0].average_prec, global_step=gs)
        train_logger.add_scalar("PC kart", pr_dist[0].average_prec, global_step=gs)
        train_logger.add_scalar("PiB bomb", pr_box[1].average_prec, global_step=gs)
        train_logger.add_scalar("PC bomb", pr_dist[1].average_prec, global_step=gs)
        train_logger.add_scalar("PiB pickup", pr_box[2].average_prec, global_step=gs)
        train_logger.add_scalar("PC pickup", pr_dist[2].average_prec, global_step=gs)
        pr_box = [PR() for _ in range(3)]
        pr_dist = [PR(is_close=point_close) for _ in range(3)]
        for batch in valid_data:
            image = batch[0].to(device)
            heatmaps = batch[1].to(device)
            detections = model.detect(image)
            for i in range(3):
                pr_box.add(detections[i], heatmaps[:, i, :, :].detach().cpu().numpy())
                pr_dist.add(detections[i], heatmaps[:, i, :, :].detach().cpu().numpy())
        valid_logger.add_scalar("PiB kart", pr_box[0].average_prec, global_step=gs)
        valid_logger.add_scalar("PC kart", pr_dist[0].average_prec, global_step=gs)
        valid_logger.add_scalar("PiB bomb", pr_box[1].average_prec, global_step=gs)
        valid_logger.add_scalar("PC bomb", pr_dist[1].average_prec, global_step=gs)
        valid_logger.add_scalar("PiB pickup", pr_box[2].average_prec, global_step=gs)
        valid_logger.add_scalar("PC pickup", pr_dist[2].average_prec, global_step=gs)
        if (
            pr_box[0].average_prec > 0.75
            and pr_box[1].average_prec > 0.45
            and pr_box[2].average_prec > 0.85
            and pr_dist[0].average_prec > 0.72
            and pr_dist[1].average_prec > 0.45
            and pr_dist[2].average_prec > 0.85
        ):
            break
        else:
            print(
                f"{pr_box[0].average_prec}/0.75",
                f"{pr_box[1].average_prec}/0.45",
                f"{pr_box[2].average_prec}/0.85",
                f"{pr_dist[0].average_prec}/0.72",
                f"{pr_dist[0].average_prec}/0.45",
                f"{pr_dist[0].average_prec}/0.85",
                end="\r"
            )

    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images("image", imgs[:16], global_step)
    logger.add_images("label", gt_det[:16], global_step)
    logger.add_images("pred", torch.sigmoid(det[:16]), global_step)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir")
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
