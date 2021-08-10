from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import torch.optim as optim
from process import *
from utils import *
import uuid
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

def train_step(model,optimizer,features,labels,idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate_step(model,features,labels,idx_val):
    model.eval()
    with torch.no_grad():
        output = model(features)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test_step(model,features,labels,idx_test):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(),acc_test.item()
    

def train(datastr,splitstr):
    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr,splitstr)
    features = features.to(device)
    adj = adj.to(device)

    # Precompute all powers of the normalized adjacency matrix
    adj_powers = torch.zeros([gpr_coeff, adj.size(1), adj.size(1)]).to(device)
    I = torch.eye(adj.size(1)).to(device)
    adj_powers[0, :, :] = I
    for i in range(gpr_coeff-1):
        adj_powers[i+1, :, :] = torch.spmm(adj, adj_powers[i, :, :])

    # AdaGPR
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
                                         adj_powers=adj_powers,
                                         device_id=cudaid).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    bad_counter = 0
    best = 999999999
    for epoch in range(args.epochs):
        loss_tra,acc_tra = train_step(model,optimizer,features,labels,idx_train)
        loss_val,acc_val = validate_step(model,features,labels,idx_val)
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



    acc = test_step(model,features,labels,idx_test)[1]

    return acc*100

t_total = time.time()
acc_list = []
cfs_list = []
for i in range(10):
    datastr = args.data
    splitstr = 'splits/'+args.data+'_split_0.6_0.2_'+str(i)+'.npz'
    acc_list.append(train(datastr,splitstr))
    print(i,": {:.2f}".format(acc_list[-1]))
print("Train cost: {:.4f}s".format(time.time() - t_total))
print("Test acc.:{:.2f}".format(np.mean(acc_list)))

