from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import *
from utils import *
import uuid
import scipy.io as sio
from model import *

# Training settings

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=8, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='texas', help='dateset')
parser.add_argument('--dev', type=int, default=1, help='device id')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
parser.add_argument('--lamda', type=float, default=1.0, help='lamda.')
parser.add_argument('--GPR_coeffs', type=int, default=4, help='Number of coeffs')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
args = parser.parse_args()

gpr_coeff =  args.GPR_coeffs
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cudaid = "cpu" #"cuda:"+str(args.dev)
device = torch.device(cudaid)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid,checkpt_file)

def train_step(model,optimizer,features,labels,adj,idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features,adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate_step(model,features,labels,adj,idx_val):
    model.eval()
    with torch.no_grad():
        output = model(features,adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test_step(model,features,labels,adj,idx_test):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(),acc_test.item()
    

def train(datastr,splitstr):
    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr,splitstr)
    features = features.to(device)
    adj = adj.to(device)

    pw = torch.zeros([gpr_coeff, adj.size(1), adj.size(1)]).to(device)
    print(pw.size())
    I = torch.eye(adj.size(1)).to(device)
    pw[0, :, :] = I
    for i in range(gpr_coeff-1):
        pw[i+1, :, :] = torch.spmm(adj, pw[i, :, :])

    # AdaPageRankII
    model = AdaGPR(nfeat=features.shape[1],
                                         nlayers=args.layer,
                                         nhidden=args.hidden,
                                         adj=adj,
                                         nclass=num_labels,
                                         dropout=args.dropout,
                                         lamda=args.lamda,
                                         alpha=args.alpha,
                                         variant=args.variant,
                                         gpr_coeff=gpr_coeff,
                                         pw=pw,
                                         device_id=cudaid).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    bad_counter = 0
    best = 999999999
    for epoch in range(args.epochs):
        loss_tra,acc_tra = train_step(model,optimizer,features,labels,adj,idx_train)
        loss_val,acc_val = validate_step(model,features,labels,adj,idx_val)
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
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    '''
    l = np.zeros([args.layer,args.telportParam])
    l1 = np.zeros([args.layer, args.telportParam])
    #L0 = list()
    #L = list()
    for i in range(args.layer):
        #print(model.convs[i].pg_coeffs.detach().numpy().T)
        l[i,:] = model.convs[i].cf.cpu().detach().numpy()[:].T
        #L0.append(str(model.convs[i].cf.detach().numpy()[:].T).replace(' ', ' & ').replace('[','').replace(']','').replace('\n','') )
    print(l)
    #print(L0)
    '''

    acc = test_step(model,features,labels,adj,idx_test)[1]

    return acc*100,l

t_total = time.time()
acc_list = []
cfs_list = []
for i in range(10):
    datastr = args.data
    splitstr = 'splits/'+args.data+'_split_0.6_0.2_'+str(i)+'.npz'
    accs,l = train(datastr, splitstr)
    acc_list.append(accs)
    cfs_list.append(l)
    print(i,": {:.2f}".format(acc_list[-1]))
print("Train cost: {:.4f}s".format(time.time() - t_total))
print("Test acc.:{:.2f}".format(np.mean(acc_list)))
filename = './full_supervsied_GPR_sparsemax_exp/texas_GPR_sparsemax_exp_acc_' + '_' + str(
    args.seed) + '_' + str(args.layer) + '_' + str(args.hidden) + '_' + str(args.alpha)  + '_' + str(args.lamda) + '_' + str(args.dropout)  + '_' + str(args.lr)  + '_' + str(args.weight_decay) + '_'  + str(telportParam) + '_.mat'
# filename = 'sims.mat'
sio.savemat(filename, {'res': acc_list, 'cfs' : cfs_list})
