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
    train_data = load_data("data/train")
    valid_data = load_data("data/valid")

    # Run SGD for several epochs
    for _ in range(10):
        for batch in train_data:
            for el in batch:
                print(el)
            inputs = torch.tensor([data[0] for data in batch])
            outputs = model.forward(inputs)
            labels = torch.tensor([data[1] for data in batch])
            error = loss.forward(outputs, batch)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
        # if accuracy(outputs, labels) > 0.9:
        #     break

    # Save your final model, using save_model
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
