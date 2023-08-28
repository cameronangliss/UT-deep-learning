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
    while True:
        for batch in train_data:
            inputs = batch[0].to(model.device)
            labels = batch[1]
            outputs = model.forward(inputs)
            error = loss.forward(outputs, labels)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
        for batch in valid_data:
            inputs = batch[0]
            labels = batch[1]
            outputs = model.forward(inputs)
            score = accuracy(outputs, labels)
        if score > 0.75:
            break

    # Save your final model, using save_model
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
