import sys
from os import path
import torch
import torch.nn.functional as F
import numpy as np
from keras.datasets import mnist
import random
from sklearn.utils import shuffle

from models import ConvModel, BigConvModel, save_model, load_model

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    test = outputs_idx.eq(labels).float()
    test_mean = test.mean()
    #print("accuracy: ", test_mean.item())
    return test_mean

# Fetch a batch from an image set, with random noise interspersed
# return a tensors: [B*2, 1, 28, 28], [B*2]
def get_batch(batch_size, imgs, label):
    img_batch = []
    label_batch = []
    indices = random.sample(range(len(imgs)), k=batch_size)

    for i in range(batch_size):
        # add actual data
        main_index = indices[i]
        img_batch.append(imgs[main_index])
        label_batch.append(label)

        # add random noise img in with label 10
        rand_top = random.randrange(2, 256)
        noisy_img = [[random.randrange(0, rand_top) for j in range(28)] for k in range(28)]
        img_batch.append(noisy_img)
        label_batch.append(10)

    # shuffle but keep mapping
    img_batch, label_batch = shuffle(img_batch, label_batch)

    imgs_T = torch.Tensor(img_batch)
    imgs_T = imgs_T[:, None] # change [B, 28, 28] -> [B, 1, 28, 28] for convolutions
    labels_T = torch.Tensor(label_batch)
    labels_T = labels_T.type(torch.long)
    return imgs_T, labels_T


def get_full_batch(batch_size, imgs, labels):
    img_batch = []
    label_batch = []
    indices = random.sample(range(len(imgs)), k=batch_size)

    for i in range(batch_size):
        # add actual data
        main_index = indices[i]
        img_batch.append(imgs[main_index])
        label_batch.append(labels[i])

        # add random noise img in with label 10
        rand_top = random.randrange(2, 256)
        noisy_img = [[random.randrange(0, rand_top) for j in range(28)] for k in range(28)]
        img_batch.append(noisy_img)
        label_batch.append(10)

    # shuffle but keep mapping between imgs and labels
    img_batch, label_batch = shuffle(img_batch, label_batch)

    imgs_T = torch.Tensor(img_batch)
    imgs_T = imgs_T[:, None] # change [B, 28, 28] -> [B, 1, 28, 28] for convolutions
    labels_T = torch.Tensor(label_batch)
    labels_T = labels_T.type(torch.long)
    return imgs_T, labels_T

def train_model(model, imgs, train_label, optimizer):
    model.train()
    device = torch.device('cuda')
    loss_vals, acc_vals = [], []

    batch_size = 250
    num_loops = int(len(imgs) / batch_size)
    for i in range(num_loops):
        # get a batch
        img, label = get_batch(batch_size, imgs, train_label)
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

def validate_model(model, valid_imgs, valid_label):
    model.eval()
    device = torch.device('cuda')
    vacc_vals = []

    batch_size = 500
    num_loops = int(len(valid_imgs) / batch_size)
    for i in range(num_loops):
        img, label = get_batch(batch_size, valid_imgs, valid_label)
        img, label = img.to(device), label.to(device)
        logit = model(img)

        acc = accuracy(logit, label).detach().cpu().numpy()
        vacc_vals.append(acc)
    avg_vacc = sum(vacc_vals) / len(vacc_vals)
    return avg_vacc

def train_full(model, optimizer, digit_models, imgs, labels):
    model.train()
    device = torch.device('cuda')
    loss_vals, acc_vals = [], []

    batch_size = 250
    num_loops = int(len(imgs) / batch_size)
    for i in range(num_loops):
        # get a batch
        img, label = get_full_batch(batch_size, imgs, labels)
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

        # reset everything but last layer
        # for i in len(digit_models):
        #     digit_model = digit_models[i]
        #     each_width = model.width / 10
        #     index = i * each_width
        #     model.conv1.weight.data[index] = digit_model.conv1.weight.data[i]
        #     model.conv1.bias.data[index] = digit_model.conv1.bias.data[i]
        #     model.conv2.weight.data[index] = digit_model.conv2.weight.data[i]
        #     model.conv2.bias.data[index] = digit_model.conv2.bias.data[i]
        #     model.conv3.weight.data[index] = digit_model.conv3.weight.data[i]
        #     model.conv3.bias.data[index] = digit_model.conv3.bias.data[i]

    avg_loss = sum(loss_vals) / len(loss_vals)
    avg_acc = sum(acc_vals) / len(acc_vals)
    return avg_loss, avg_acc

def validate_full(model, valid_imgs, valid_label):
    model.eval()
    device = torch.device('cuda')
    vacc_vals = []

    batch_size = 500
    num_loops = int(len(valid_imgs) / batch_size)
    for i in range(num_loops):
        img, label = get_full_batch(batch_size, valid_imgs, valid_label)
        img, label = img.to(device), label.to(device)
        logit = model(img)

        acc = accuracy(logit, label).detach().cpu().numpy()
        vacc_vals.append(acc)
    avg_vacc = sum(vacc_vals) / len(vacc_vals)
    return avg_vacc

# organized train data by labels, looks like:
# org_imgs[0] = {set of all 0's}
# org_imgs[1] = {set of all 1's}
# ...
def organize_data(imgs, labels):
    org_imgs = [ [] for i in range(10) ]
    for i in range(imgs.shape[0]):
        n = labels[i]
        org_imgs[n].append(imgs[i])
    return org_imgs

def train(args):
    if not torch.cuda.is_available():
        sys.exit("GPU is not available")

    device = torch.device('cuda')

    # load data
    (train_imgs, train_labels), (valid_imgs, valid_labels) = mnist.load_data()

    # organize data into classes
    org_train_imgs = organize_data(train_imgs, train_labels)
    org_valid_imgs = organize_data(valid_imgs, valid_labels)

    # train a model separately for each digit
    digit_models = []
    width = 5

    for i in range(10):
        print("------------------")
        print("TRAINING DIGIT ", i)

        model = ConvModel(width)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in range(args.num_epoch):
            avg_loss, avg_acc = train_model(model, org_train_imgs[i], i, optimizer)
            avg_vacc = validate_model(model, org_valid_imgs[i], i)
            print('epoch %-3d \t loss = %0.7f \t train_acc = %0.7f \t val_acc = %0.7f' % (epoch, avg_loss, avg_acc, avg_vacc))

        digit_models.append(model)

    # Now combine the models and train the last layer
    for epoch in range(args.num_epoch):
        print("--------------------")
        print("TRAINING FULL MODEL!")
        model = BigConvModel(width*10, digit_models)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in range(5):
            avg_loss, avg_acc = train_full(model, optimizer, digit_models, train_imgs, train_labels)
            avg_vacc = validate_full(model, valid_imgs, valid_labels)
            print('epoch %-3d \t loss = %0.7f \t train_acc = %0.7f \t val_acc = %0.7f' % (epoch, avg_loss, avg_acc, avg_vacc))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=1)
    parser.add_argument('-e', '--num_era', type=int, default=1)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
