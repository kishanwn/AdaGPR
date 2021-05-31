import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sparsemax_ import Sparsemax
##-------------------------------------------------------------------------------------------------------
# Generalized Pagerank with Sparsemax on GCNII
##-------------------------------------------------------------------------------------------------------

class GraphConvolution_sparsemax_exp(nn.Module):

    def __init__(self, in_features, out_features, num_nodes, residual=False, variant=False,gpr_coeff=10,adj_powers=None,device_id="cpu"):
        super(GraphConvolution_sparsemax_exp, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.num_nodes = num_nodes
        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()
        self.gpr_coeff = gpr_coeff
        self.adj_powers = adj_powers

        # create random coefficients for pageranks
        self.pg_coeffs = nn.Parameter(torch.randn( self.gpr_coeff, 1 ))
        self.reset_parametersWcoeffs()

        self.sparsemax_fn = Sparsemax(dim=0,device_id=device_id)
        self.cf = 0


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def reset_parametersWcoeffs(self):
        stdv = 1. / math.sqrt(self.gpr_coeff)
        self.pg_coeffs.data.uniform_(-stdv, stdv)


    def forward(self, input,  h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)

        predcitedPg= 0
        self.cf = self.sparsemax_fn(torch.exp(self.pg_coeffs))
        for i in range(self.gpr_coeff):
            predcitedPg += self.cf[i]*self.adj_powers[i,:,:]
        predcitedPg= predcitedPg.squeeze()

        hi = torch.spmm(predcitedPg, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output


class AdaGPR(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, adj, nclass, dropout, lamda, alpha, variant,gpr_coeff,adj_powers,device_id):
        super(AdaGPR, self).__init__()
        self.convs = nn.ModuleList()
        num_nodes = adj.size(0)
        i = 0
        for _ in range(nlayers):
            self.convs.append(GraphConvolution_sparsemax_exp(nhidden, nhidden,num_nodes,variant=variant,gpr_coeff=gpr_coeff,adj_powers=adj_powers,device_id=device_id  ) )
            i += 1
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.gpr_coeff = gpr_coeff
        self.adj_powers = adj_powers

    def forward(self, x):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)

