import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """

    # create a model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Detector().to(device)
    # model.load_state_dict(torch.load("homework/det.th"))
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    # load the data: train and valid
    transform = dense_transforms.Compose([
        dense_transforms.ToTensor(),
        dense_transforms.ToHeatmap(),
        dense_transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.25
        ),
        dense_transforms.RandomHorizontalFlip()
    ])
    train_data = load_detection_data("dense_data/train", transform)
    valid_data = load_detection_data("dense_data/valid")

    # Run SGD for several epochs
    global_step = 0
    while True:
        conf_matrix = ConfusionMatrix()
        for batch in train_data:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model.forward(inputs)
            conf_matrix.add(outputs.argmax(1), labels)
            error = loss.forward(outputs, labels.long())
            train_logger.add_scalar('loss', error, global_step=global_step)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
            global_step += 1
        train_logger.add_scalar('global_accuracy', conf_matrix.global_accuracy, global_step=global_step)
        train_logger.add_scalar('IoU', conf_matrix.iou, global_step=global_step)
        scheduler.step(conf_matrix.global_accuracy)
        conf_matrix = ConfusionMatrix()
        for batch in valid_data:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model.forward(inputs)
            conf_matrix.add(outputs.argmax(1), labels)
        valid_logger.add_scalar('global_accuracy', conf_matrix.global_accuracy, global_step=global_step)
        valid_logger.add_scalar('IoU', conf_matrix.iou, global_step=global_step)
        if conf_matrix.global_accuracy > 0.9 and conf_matrix.iou > 0.6:
            break

    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
