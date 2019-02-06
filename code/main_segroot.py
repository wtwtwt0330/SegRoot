import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
import argparse

from model import SegRoot
from dataloader import StaticTrainDataset, TestDataset, TrainDataset, LoopSampler
from utils import (
    dice_score,
    init_weights,
    evaluate,
    get_ids,
    load_vgg16,
    set_random_seed,
)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="set random seed")
parser.add_argument("--width", default=8, type=int, help="width of SegRoot")
parser.add_argument("--depth", default=5, type=int, help="depth of SegRoot")
parser.add_argument("--bs", default=64, type=int, help="batch size of dataloaders")
parser.add_argument("--lr", default=1e-2, type=float, help="learning rate")
parser.add_argument("--epochs", default=200, type=int, help="max epochs of training")
parser.add_argument(
    "--verbose", default=5, type=int, help="intervals to save and validate model"
)
parser.add_argument(
    "--dynamic", action="store_true", help="use dynamic sub-images during training"
)


def train_one_epoch(model, train_iter, optimizer, device):
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    for x, y in train_iter:
        x, y = x.to(device), y.to(device)
        bs = x.shape[0]
        optimizer.zero_grad()
        y_pred = model(x)
        loss = 1 - dice_score(y, y_pred)
        loss = torch.sum(loss) / bs
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    args = parser.parse_args()
    seed = args.seed
    bs = args.bs
    lr = args.lr
    width = args.width
    depth = args.depth
    epochs = args.epochs
    verbose = args.verbose

    # set random seed
    set_random_seed(seed)
    # define the device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get training ids
    train_ids, valid_ids, test_ids = get_ids(65)
    # define dataloaders
    if args.dynamic:
        train_data = TrainDataset(train_ids)
        train_iter = DataLoader(
            train_data, batch_size=bs, num_workers=6, sampler=LoopSampler
        )
    else:
        train_data = StaticTrainDataset(train_ids)
        train_iter = DataLoader(train_data, batch_size=bs, num_workers=6, shuffle=True)

    train_tdata = TestDataset(train_ids)
    valid_tdata = TestDataset(valid_ids)
    test_tdata = TestDataset(test_ids)

    # define model
    model = SegRoot(width, depth).to(device)
    model = model.apply(init_weights)

    # define optimizer and lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, verbose=True, patience=5
    )

    print(f"Start training SegRoot-({width},{depth}))......")
    print(f"Random seed is {seed}, batch size is {bs}......")
    print(f"learning rate is {lr}, max epochs is {epochs}......")
    best_valid = float("-inf")
    for epoch in tqdm(range(epochs)):
        train_one_epoch(model, train_iter, optimizer, device)
        if epoch % verbose == 0:
            train_dice = evaluate(model, train_tdata, device)
            valid_dice = evaluate(model, valid_tdata, device)
            scheduler.step(valid_dice)
            print(
                "Epoch {:05d}, train dice: {:.4f}, valid dice: {:.4f}".format(
                    epoch, train_dice, valid_dice
                )
            )
            if valid_dice > best_valid:
                best_valid = valid_dice
                test_dice = evaluate(model, test_tdata, device)
                print("New best validation, test dice: {:.4f}".format(test_dice))
                torch.save(
                    model.state_dict(),
                    f"../weights/best_segroot-({args.width},{args.depth}).pt",
                )
