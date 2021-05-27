from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import *
import uuid
import scipy.io as sio

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=0.0005, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=32, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=256, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='citeseer', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.3, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
parser.add_argument('--GPR_coeff', type=int, default=10, help='evaluation on test set.')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

numSims = 100
ArrResults = np.zeros([numSims])


for j in range(numSims):
    # Load data
    adj, features, labels,idx_train,idx_val,idx_test = load_citation(args.data)
    cudaid = "cuda:"+str(args.dev)
    device = torch.device(cudaid)
    features = features.to(device)
    adj = adj.to(device)

    gpr_coeff = args.GPR_coeff
    pw = torch.zeros([gpr_coeff,adj.size(1),adj.size(1)]).to(device)
    print(pw.size())
    I = torch.eye(adj.size(1)).to(device)
    pw[0, :, :] = I
    for i in range(gpr_coeff - 1):
        pw[i + 1, :, :] = torch.spmm(adj, pw[i, :, :])

    pw = pw.to(device)
    print(pw.size())
    checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
    print(cudaid,checkpt_file)
    print(int(labels.max()) + 1)
    model = AdaGPR(nfeat=features.shape[1],
                    nlayers=args.layer,
                    nhidden=args.hidden,
                    adj=adj,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    lamda = args.lamda,
                    alpha=args.alpha,
                    variant=args.variant,
                    gpr_coeff= gpr_coeff,
                    pw = pw,
                    device_id=cudaid).to(device)

    optimizer = optim.Adam([
                            {'params':model.params1,'weight_decay':args.wd1},
                            {'params':model.params2,'weight_decay':args.wd2},
                            ],lr=args.lr)

    def train():
        model.train()
        optimizer.zero_grad()
        output = model(features,adj)
        acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
        loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
        #loss_train = accuracy(output[idx_train], labels[idx_train].to(device))
        loss_train.backward()
        optimizer.step()
        return loss_train.item(),acc_train.item()


    def validate():
        model.eval()
        with torch.no_grad():
            output = model(features,adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
            acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
            return loss_val.item(),acc_val.item()

    def test():
        model.load_state_dict(torch.load(checkpt_file))
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
            acc_test = accuracy(output[idx_test], labels[idx_test].to(device))

            l = np.zeros([args.layer, args.telportParam])
            for i in range(args.layer):
                # print(model.convs[i].pg_coeffs.detach().numpy().T)
                l[i, :] = model.convs[i].cf.cpu().detach().numpy()[:].T
                # L0.append(str(model.convs[i].cf.detach().numpy()[:].T).replace(' ', ' & ').replace('[','').replace(']','').replace('\n','') )
            print(l)

            return loss_test.item(),acc_test.item(),l

    t_total = time.time()
    bad_counter = 0
    best = 999999999
    best_epoch = 0
    acc = 0
    for epoch in range(args.epochs):
        loss_tra,acc_tra = train()
        loss_val,acc_val = validate()
        if(epoch+1)%1 == 0:
            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))
        if loss_val < best:
            best = loss_val
            best_epoch = epoch
            acc = acc_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    if args.test:
        acc = test()[1]

    print("Train cost: {:.4f}s".format(time.time() - t_total))
    print('Load {}th epoch'.format(best_epoch))
    print("Test" if args.test else "Val","acc.:{:.1f}".format(acc*100))
    res = [args.seed, acc, best_epoch]
    ArrResults[j] = acc
    filename2 = './semi_supervised_GPR_sparsemax/citeseer_GPR_sparsemax_iter_seed_acc_' + '_' + str(args.seed) + '_' + str(args.layer) + '_' + str(args.hidden) + '_' + str(args.alpha) + '_' + str(args.lamda) + '_' + str(args.dropout) + '_' + str(args.wd1) + '_' + str(args.wd2) + '_' + str(args.telportParam) + '.mat'
    sio.savemat(filename2, {'res': ArrResults})




