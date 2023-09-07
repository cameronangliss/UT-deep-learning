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
    model = CNNClassifier([32, 64, 128, 256]).to(device)
    # model.load_state_dict(torch.load("homework/cnn.th"))
    loss = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.005)

    # load the data: train and valid
    train_data = load_data("data/train")
    valid_data = load_data("data/valid")

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
            error = loss.forward(outputs, labels)
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
        if score > 0.885:
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
