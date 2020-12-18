import sys
from os import path
import torch
import torch.nn.functional as F
import numpy as np

from models import LinearModel
from models import save_model, load_model
from utils import accuracy, load_data

def train_model(model, train_data, optimizer):
    model.train()
    device = torch.device('cuda')
    loss_vals, acc_vals = [], []
    for img, label in train_data:
        img, label = img.to(device), label.to(device)

        logit = model(img)
        loss_val = F.cross_entropy(logit, label)
        acc_val = accuracy(logit, label)

        loss_vals.append(loss_val.detach().cpu().numpy())
        acc_vals.append(acc_val.detach().cpu().numpy())

        # update weights
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    avg_loss = sum(loss_vals) / len(loss_vals)
    avg_acc = sum(acc_vals) / len(acc_vals)
    return avg_loss, avg_acc

def validate_model(model, valid_data):
    model.eval()
    device = torch.device('cuda')
    vacc_vals = []

    for img, label in valid_data:
        img, label = img.to(device), label.to(device)
        logit = model(img)

        acc = accuracy(logit, label).detach().cpu().numpy()
        vacc_vals.append(acc)
    avg_vacc = sum(vacc_vals) / len(vacc_vals)
    return avg_vacc

def train(args):
    if not torch.cuda.is_available():
        sys.exit("GPU is not available")

    device = torch.device('cuda')

    model = LinearModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.to(device)

    train_data = load_data('data/train')
    valid_data = load_data('data/valid')

    avg_avg_vacc = []
    # train model over all data epoch times
    for epoch in range(args.num_epoch):
        avg_loss, avg_acc = train_model(model, train_data, optimizer)
        avg_vacc = validate_model(model, valid_data)
        avg_avg_vacc.append(avg_vacc)
        #print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_loss, avg_acc, avg_vacc))
        save_model(model)

    print(np.mean(avg_avg_vacc), end=", ")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=30)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
