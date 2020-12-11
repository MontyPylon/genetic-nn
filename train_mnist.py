import sys
from os import path
import torch
import torch.nn.functional as F
import numpy as np
from keras.datasets import mnist
import random

from models import LinearModel, ConvModel, FCN
from models import save_model, load_model
from utils import accuracy

# Fetch a batch from the total image set
# return two tensors: [B, 1, 28, 28], [B]
def get_batch(batch_size, imgs, labels):
    indices = random.sample(range(len(imgs)), k=batch_size)

    img_batch = []
    label_batch = []
    for i in range(batch_size):
        main_index = indices[i]
        img_batch.append(imgs[main_index])
        label_batch.append(labels[main_index])

    imgs_T = torch.Tensor(img_batch)
    imgs_T = imgs_T[:, None] # change [B, 28, 28] -> [B, 1, 28, 28] for convolutions
    labels_T = torch.Tensor(label_batch)
    labels_T = labels_T.type(torch.long)
    return imgs_T, labels_T

def train_model(model, train_imgs, train_labels, optimizer):
    model.train()
    device = torch.device('cuda')
    loss_vals, acc_vals = [], []

    batch_size = 500
    num_loops = int(len(train_imgs) / batch_size)
    for i in range(num_loops):
        # get a batch
        img, label = get_batch(batch_size, train_imgs, train_labels)
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

def tally_good(model, tallies, img, logit, label):
    batch_size = logit.size()[0]
    
    for i in range(batch_size):
        row = logit[i]
        max_i = torch.argmax(row).item()
        correct = label[i].item()

        if max_i == correct:
            flat_img = torch.flatten(img[i])

            # for l1
            z = model.l1(flat_img)

            pos = (z > 0).type(torch.uint8)
            tallies += pos

def validate_model(model, valid_imgs, valid_labels):
    model.eval()
    device = torch.device('cuda')
    vacc_vals = []

    batch_size = 500
    num_loops = int(len(valid_imgs) / batch_size)
    for i in range(num_loops):
        img, label = get_batch(batch_size, valid_imgs, valid_labels)
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

    # load data
    (train_imgs, train_labels), (valid_imgs, valid_labels) = mnist.load_data()
    
    model = FCN()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for era in range(args.num_era):

        # train model over all data epoch times
        for epoch in range(args.num_epoch):
            avg_loss, avg_acc = train_model(model, train_imgs, train_labels, optimizer)
            avg_vacc = validate_model(model, valid_imgs, valid_labels)
            print('epoch %-3d \t loss = %0.5f \t acc = %0.5f \t val acc = %0.5f' % (epoch, avg_loss, avg_acc, avg_vacc))

        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-e', '--num_era', type=int, default=1)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
