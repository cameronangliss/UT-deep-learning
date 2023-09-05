from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger: tb.SummaryWriter, valid_logger: tb.SummaryWriter):
    # This is a strongly simplified training loop
    global_step = 0
    for epoch in range(10):
        torch.manual_seed(epoch)
        mean_acc = 0
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            train_logger.add_scalar('loss', dummy_train_loss, global_step=global_step)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            mean_acc += torch.mean(dummy_train_accuracy)
            global_step += 1
        train_logger.add_scalar('accuracy', mean_acc / 20, global_step=global_step)
        torch.manual_seed(epoch)
        mean_acc = 0
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            mean_acc += torch.mean(dummy_validation_accuracy)
        valid_logger.add_scalar('accuracy', mean_acc / 10, global_step=global_step)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
