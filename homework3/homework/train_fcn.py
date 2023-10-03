import torch
import numpy as np
import torchvision

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix, accuracy
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = FCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """

    # create a model, loss, optimizer
    device = torch.device("cpu")
    model = FCN().to(device)
    # model.load_state_dict(torch.load("homework/cnn.th"))
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    # load the data: train and valid
    transform = dense_transforms.Compose([
        dense_transforms.ToTensor(),
        dense_transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.25
        ),
        dense_transforms.RandomHorizontalFlip()
    ])
    train_data = load_dense_data("dense_data/train", transform)
    valid_data = load_dense_data("dense_data/valid")

    # Run SGD for several epochs
    global_step = 0
    while True:
        score = 0
        n = 0
        for batch in train_data:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model.forward(inputs)
            score += accuracy(outputs, labels)
            error = loss.forward(outputs, labels.long())
            train_logger.add_scalar('loss', error, global_step=global_step)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
            global_step += 1
            n += 1
        score /= n
        train_logger.add_scalar('accuracy', score, global_step=global_step)
        score = 0
        n = 0
        for batch in valid_data:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model.forward(inputs)
            score += accuracy(outputs, labels)
            n += 1
        score /= n
        valid_logger.add_scalar('accuracy', score, global_step=global_step)
        if score > 0.8:
            break

    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
