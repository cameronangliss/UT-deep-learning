from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
from torch.optim import SGD
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    # create a model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier().to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    # load the data: train and valid
    train_data = load_data("data/train")
    valid_data = load_data("data/valid")

    # Run SGD for several epochs
    while True:
        for batch in train_data:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model.forward(inputs)
            print(inputs.size(), outputs.size(), labels.size())
            error = loss.forward(outputs, labels)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
        score = 0
        n = 0
        for batch in valid_data:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model.forward(inputs)
            score += accuracy(outputs, labels)
            n += 1
        score /= n
        if score > 0.5:
            break

    # Save your final model, using save_model
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
