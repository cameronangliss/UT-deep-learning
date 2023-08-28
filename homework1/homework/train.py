from torch.optim import SGD

from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data


def train(args):
    # create a model, loss, optimizer
    model = model_factory[args.model]()
    loss = ClassificationLoss()
    optimizer = SGD(model.parameters)

    # load the data: train and valid
    train_data = load_data("data/train")
    valid_data = load_data("data/valid")

    # Run SGD for several epochs
    while True:
        batch = next(train_data)
        outputs = model.forward(batch)
        labels = [row["label"] for row in batch]
        if accuracy(outputs, labels) < 0.9:
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
