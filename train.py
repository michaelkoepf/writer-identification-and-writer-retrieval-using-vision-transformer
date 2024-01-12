#!/usr/bin/env python3

import argparse
import os

from src.lr_schedulers import WarmUpLR
from src.models import vit_lite_7_4
from src.preprocessing import defaults
from src.pytorch_utils import seed_worker, set_all_seeds, Trainer
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

OPTIMIZERS = ["adamw"]
MODEL_VARIANTS = {"vit-lite-7-4": vit_lite_7_4}


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="A command-line tool for training Vision Transformers (ViTs) for writer identification and writer "
                    "retrieval",
        epilog="Note, that the same parameter configuration (model, optimizer, learning rate, number of warmup "
               "epochs, batch size and seed) can only be ran once. If you wish to repeat the same run, delete the "
               "respective sub-directories in the `run` and `saved_models` directory.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--optim", choices=OPTIMIZERS, default=OPTIMIZERS[0],
                        help="Optimizer")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--num-epochs-warmup", type=int, default=10,
                        help="Number of epochs for learning rate warmup")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("-w", "--num-workers", type=int, default=8,
                        help="Number of PyTorch workers")
    parser.add_argument("-s", "--seed", type=int, default=417,  # default generated with random.org (range 0 to 2^16-1)
                        help="Seed")
    parser.add_argument("--num-epochs", type=int, default=60,
                        help="Maximum number of epochs the model should be trained")
    parser.add_argument("--num-epochs-patience", type=int, default=10,
                        help="Number of epochs after the training should be stopped, if the validation loss does not "
                             "improve any more")
    parser.add_argument("--model", choices=MODEL_VARIANTS.keys(), default="vit-lite-7-4",
                        help="Model variant")
    parser.add_argument("experiment", choices=[k for k, v in defaults.items() if v.num_classes_train],
                        help="Experiment")

    return parser.parse_args()


def _get_experiment_name(args):
    return f"{args.model}_optim-{args.optim}_lr-{args.lr}_wup-{args.num_epochs_warmup}_bsize-" \
           f"{args.batch_size}_seed-{args.seed}"


def _get_log_dir(args):
    return os.path.join("runs", f"{args.experiment}")


def _get_saved_models_dir(args):
    return os.path.join("saved_models", f"{args.experiment}")


def _get_hyper_params(args):
    return {
        "optimizer": args.optim,
        "lr": args.lr,
        "num_epochs_warmup": args.num_epochs_warmup,
        "batch_size": args.batch_size
    }


def _get_optimizer(model, hyper_params):
    if hyper_params["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=hyper_params["lr"],
                                 weight_decay=3e-2)
    else:
        assert False, "Optimizer not implemented"


def _get_data_loaders(args, hyper_params):
    train_set = datasets.ImageFolder(os.path.join("data", "preprocessed", args.experiment, "train"),
                                     transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.RandomRotation(degrees=(-25, 25),
                                                                                             fill=1)]))

    if len(train_set):
        train_set_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=hyper_params["batch_size"],
                                      num_workers=args.num_workers,
                                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed))

    val_set = datasets.ImageFolder(os.path.join("data", "preprocessed", args.experiment, "val"),
                                   transform=transforms.ToTensor())
    if len(val_set):
        val_set_loader = DataLoader(dataset=val_set, shuffle=False, batch_size=hyper_params["batch_size"],
                                    num_workers=args.num_workers)

    return train_set_loader, val_set_loader


def main():
    args = _parse_arguments()

    # preprocess
    experiment = defaults[args.experiment]
    experiment()

    hyper_params = _get_hyper_params(args)
    set_all_seeds(args.seed)

    # load datasets
    train_set_loader, val_set_loader = _get_data_loaders(args, hyper_params)

    # prepare training and train
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Device: " + str(device))

    model = MODEL_VARIANTS[args.model](experiment.num_classes_train).to(device=device)
    criterion = LabelSmoothingCrossEntropy().to(device=device)
    optimizer = _get_optimizer(model, hyper_params)
    scheduler = WarmUpLR(optimizer, hyper_params["lr"], num_epochs_warm_up=hyper_params["num_epochs_warmup"])

    trainer = Trainer(model, criterion, optimizer, scheduler, args.num_epochs, train_set_loader, val_set_loader,
                      experiment_name=_get_experiment_name(args), hyper_params=hyper_params,
                      num_epochs_early_stop=args.num_epochs_patience,
                      log_dir=_get_log_dir(args), saved_models_dir=_get_saved_models_dir(args))

    print("Starting training...")
    trainer()


if __name__ == "__main__":
    main()
