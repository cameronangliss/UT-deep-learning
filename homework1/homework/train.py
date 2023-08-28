import torch
from torch.optim import SGD

from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data


def train(args):
    # create a model, loss, optimizer
    model = model_factory[args.model]()
    loss = ClassificationLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    # load the data: train and valid
    train_data = load_data("data/train").dataset
    valid_data = load_data("data/valid").dataset

    # Run SGD for several epochs
    while True:
        batch = next(train_data)
        inputs = torch.tensor([data[0] for data in batch])
        outputs = model.forward(inputs)
        labels = torch.tensor([data[1] for data in batch])
        if accuracy(outputs, labels) > 0.9:
            break
        error = loss.forward(outputs, batch)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

    # Save your final model, using save_model
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
