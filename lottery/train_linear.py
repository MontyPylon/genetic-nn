import sys
from os import path
import torch
import torch.nn.functional as F
import numpy as np

from models import LinearModel, ConvModel, FCN
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

def validate_model(model, valid_data, to_tally):
    model.eval()
    device = torch.device('cuda')
    vacc_vals = []

    # tallies of which rows in weight matrix are "good"
    tallies = torch.zeros(model.l1.weight.shape[0])
    tallies = tallies.to(device)

    for img, label in valid_data:
        img, label = img.to(device), label.to(device)
        logit = model(img)

        # loop logits and labels and count rows which were activated and correct
        if to_tally:
            tally_good(model, tallies, img, logit, label)

        acc = accuracy(logit, label).detach().cpu().numpy()
        vacc_vals.append(acc)
    avg_vacc = sum(vacc_vals) / len(vacc_vals)
    return tallies, avg_vacc

def train(args):

    torch.set_printoptions(sci_mode=False)

    if not torch.cuda.is_available():
        sys.exit("GPU is not available")

    device = torch.device('cuda')

    train_data = load_data('data/train')
    valid_data = load_data('data/valid')

    k = 700
    top = None
    model = None

    for era in range(args.num_era):

        if model is None:
            model = LinearModel()

            if args.continue_training:
                load_model(model)
        else:
            model = LinearModel(model.l1, top, k)

        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        avg_avg_vacc = []

        # train model over all data epoch times
        for epoch in range(args.num_epoch):
            avg_loss, avg_acc = train_model(model, train_data, optimizer)
            tallies, avg_vacc = validate_model(model, valid_data, False)
            avg_avg_vacc.append(avg_vacc)
            print('epoch %-3d \t loss = %0.5f \t acc = %0.5f \t val acc = %0.5f' % (epoch, avg_loss, avg_acc, avg_vacc))


        print("---------------------------------")
        print("mean valid accuracy: ", np.mean(avg_avg_vacc))
        print("---------------------------------")

        save_model(model)

        # calculate which rows in weight matrix are doing the work
        tallies, avg_vacc = validate_model(model, valid_data, True)
        print('tallies: ', tallies.detach().cpu().numpy())
        
        top = torch.topk(tallies, k)

        #print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_loss, avg_acc, avg_vacc))


    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=70)
    parser.add_argument('-e', '--num_era', type=int, default=500)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
