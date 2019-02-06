import pickle
import torch
from torchvision import models
import random
import logging
import numpy as np
import json

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def to_np(x):
    return x.data.cpu().numpy()

def get_ids(length_dataset):
    ids = list(range(length_dataset))

    random.shuffle(ids)
    train_split = round(0.6 * length_dataset)
    t_v_spplit = (length_dataset - train_split) // 2
    train_ids = ids[:train_split]
    valid_ids = ids[train_split:train_split+t_v_spplit]
    test_ids = ids[train_split+t_v_spplit:]
    return train_ids, valid_ids, test_ids

def dice_score(y, y_pred, smooth=1.0, thres=0.9):
    n = y.shape[0]
    y = y.view(n, -1)
    y_pred = y_pred.view(n, -1)
    # y_pred_[y_pred>=thres] = 1.0
    # y_pred_[y_pred<thres] = 0.0 
    num = 2 * torch.sum(y * y_pred, dim=1, keepdim=True) + smooth
    den = torch.sum(y, dim=1, keepdim=True) + \
        torch.sum(y_pred, dim=1, keepdim=True) + smooth
    score = num / den
    return score

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        # torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)

def load_vgg16(segnet):
    vgg16 = models.vgg16_bn(pretrained=True)
    with open('paired_weight_vgg16.plk', 'rb') as handle:
        paired = pickle.load(handle)
    segnet_p = dict(segnet.state_dict())
    vgg16_p = vgg16.state_dict()

    for k, v in paired.items():
        for n, p in vgg16_p.items():
            if n == v:
                segnet_p[k].data.copy_(p.data)
    segnet.load_state_dict(segnet_p)
    return segnet

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

def evaluate(model, dataset, device, thres=0.9):
    model.eval()
    torch.cuda.empty_cache()    
    num, den = 0, 0
    # shutdown the autograd
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            y_pred = model(x)
            y = y.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
            y_pred[y_pred>=thres] = 1.0
            y_pred[y_pred<thres] = 0.0
            num += 2 * (y_pred * y).sum()
            den += y_pred.sum() + y.sum()
    torch.cuda.empty_cache() 
    return num / den