# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Main file to run Extreme Classification. Spectral Graph
Convolutional Layer
__description__ : Spectral Graph Convolutional Layer
__project__     : XCGCN
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found
 in the LICENSE file in the root directory of this source tree.
__classes__     : GCN_Spectral
"""

import math,time
from gc import collect
from os import makedirs
from os.path import join
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt

from data_loaders.common_data_handler import Common_Data_Handler
from data_loaders.Prepare_Data import Prepare_Data
from config import configuration as config,seed
from logger import logger

""" Requirements: fastai, pytorch, unidecode, gensim, scikit-learn, networkx
python -m spacy download en
"""


#
# from torch_geometric.nn import GCNConv
# class Net(torch.nn.Module):
#     def __init__(self,num_classes,input_feature_dim=300):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(input_feature_dim, 100)
#         self.conv2 = GCNConv(100, num_classes)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#
#         return F.log_softmax(x, dim=1)
#
#
# def extract_model_weights(model, layer_no=0):


def plot_occurance(losses: list,title="Losses",ylabel="Loss",xlabel="Epoch",
                   clear=True,log_scale=False,plot_name=None,
                   plot_dir=config["sampling"]["num_epochs"],show_plot=False):
    """ Plots the validation loss against epochs.

    :param show_plot:
    :param plot_name:
    :param plot_dir:
    :param xlabel:
    :param ylabel:
    :param title:
    :param losses:
    :param clear:
    :param log_scale:
    """
    ## Turn interactive plotting off
    plt.ioff()

    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel(xlabel)
    if log_scale:
        plt.yscale('log')
    plt.ylabel(ylabel)
    plt.title(title)

    if plot_name is None: plot_name = title + "_" + ylabel + "_" + xlabel + ".jpg"
    plot_dir = join("Plots",plot_dir)
    makedirs(plot_dir,exist_ok=True)
    plt.savefig(join(plot_dir,plot_name))
    logger.info("Saved plot with title [{}] and ylabel [{}] and xlabel [{}] at"
                " [{}].".format(title,ylabel,xlabel,join(plot_dir,plot_name)))

    if show_plot: plt.show()
    if clear: plt.cla()
    plt.close(fig)  # Closing the figure so it won't get displayed in console.


def adj_csr2t_coo(Docs_adj: sp.csr.csr_matrix) -> torch.Tensor:
    """Convert a scipy sparse "csr" matrix to a torch sparse tensor."""
    # Docs_adj = Docs_adj.tocoo().astype(np.float32)  ## TODO: make optional.
    indices = torch.from_numpy(
        np.vstack((Docs_adj.row,Docs_adj.col)).astype(np.int64))
    values = torch.from_numpy(Docs_adj.data)
    shape = torch.Size(Docs_adj.shape)
    return torch.sparse.FloatTensor(indices,values,shape)


class GCN_Spectral(Module):
    """ Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """

    def __init__(self,in_units: int,out_units: int,bias: bool = True) -> None:
        super(GCN_Spectral,self).__init__()
        self.in_units = in_units
        self.out_units = out_units
        self.weight = Parameter(torch.FloatTensor(in_units,out_units))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_units))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)

    def forward(self,input: torch.Tensor,adj: torch.Tensor) -> torch.Tensor:
        """

        weight=(input_dim X hid_dim)
        :param input: (#samples X input_dim)
        :param adj:
        :return:
        """
        support = torch.mm(input,self.weight)
        # logger.debug((adj.dtype,support.dtype))
        output = torch.spmm(adj,support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.in_units) + ' -> ' + str(self.out_units) + ')'


class GCN(nn.Module):
    def __init__(self,nfeat: int,nhid: int,nclass: int,dropout: float) -> None:
        super(GCN,self).__init__()
        self.gc1 = GCN_Spectral(nfeat,nhid)
        self.gc2 = GCN_Spectral(nhid,nclass)
        self.dropout = dropout

    def forward(self,x: torch.Tensor,adj: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x,adj))
        x = F.dropout(x,self.dropout,training=self.training)
        x = self.gc2(x,adj)
        return F.log_softmax(x,dim=1)


def accuracy(output: torch.Tensor,labels: torch.Tensor) -> torch.Tensor:
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def test(model,features,adj,labels,idx_test):
    """

    :param model:
    :param features:
    :param adj:
    :param labels:
    :param idx_test:
    """
    model.eval()
    output = model(features,adj)
    loss_test = F.nll_loss(output[idx_test],labels[idx_test])
    acc_test = accuracy(output[idx_test],labels[idx_test])
    logger.info(("Test set results:",
                 "loss= {:.4f}".format(loss_test.item()),
                 "accuracy= {:.4f}".format(acc_test.item())))


def train(epoch: int,model,optimizer,features:torch.Tensor,
          adj: torch.Tensor,labels: torch.Tensor,idx_train: torch.Tensor,idx_val: torch.Tensor) -> list:
    """

    :param epoch:
    :param model:
    :param optimizer:
    :param features:
    :param adj:
    :param labels:
    :param idx_train:
    :param idx_val:
    """
    # losses = []
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features,adj)
    loss_train = F.nll_loss(output[idx_train],labels[idx_train])
    acc_train = accuracy(output[idx_train],labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        ## Evaluate validation set performance separately, deactivates dropout
        ## during validation run.
        model.eval()
        output = model(features,adj)

    loss_val = F.nll_loss(output[idx_val],labels[idx_val])
    acc_val = accuracy(output[idx_val],labels[idx_val])
    logger.info(('Epoch: {:04d}'.format(epoch + 1),
                 'loss_train: {:.4f}'.format(loss_train.item()),
                 'acc_train: {:.4f}'.format(acc_train.item()),
                 'loss_val: {:.4f}'.format(loss_val.item()),
                 'acc_val: {:.4f}'.format(acc_val.item()),
                 'time: {:.4f}s'.format(time.time() - t)))
    # losses.append(loss_train.item())

    return loss_train.item(),acc_train.item(),loss_val.item(),acc_val.item(),\
           (time.time() - t)


def main(args):
    """
    Main function to run Matching Networks for Extreme Classification.

    :param args: Dict of all the arguments.
    """
    ## Training Phase
    data_loader = Common_Data_Handler()
    data_formatter = Prepare_Data(dataset_loader=data_loader)
    txts,sample2cats,_,cats = data_formatter.load_raw_data(load_type='all')
    txts2vec_map,cats2vec_map = data_formatter.create_vec_maps()

    input_vecs,cats_hot,keys,cats_idx = data_formatter.get_input_batch(
        txts2vec_map,sample2cats,return_cat_indices=True,
        multi_label=False)
    logger.debug(input_vecs.shape)

    input_adj_coo = data_formatter.load_graph_data(keys)
    logger.debug(input_adj_coo.shape)

    idx_train = torch.LongTensor(range(1500))
    idx_val = torch.LongTensor(range(1501,2000))
    idx_test = torch.LongTensor(range(2001,2491))

    input_vecs = torch.FloatTensor(input_vecs)
    cats_idx = torch.LongTensor(cats_idx)
    input_adj_coo_t = adj_csr2t_coo(input_adj_coo)
    logger.debug(input_adj_coo_t.shape)

    # Model and optimizer
    model = GCN(nfeat=input_vecs.shape[1],nhid=args.hidden,
                nclass=cats_hot.shape[1],dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),lr=args.lr,
                           weight_decay=args.weight_decay)

    # Train model
    train_losses,train_accs,val_losses,val_accs,train_times = [],[],[],[],[]
    t_total = time.time()
    for epoch in range(args.epochs):
        # train_losses.append(train(epoch,model,optimizer,input_vecs,input_adj_coo_t.float(),cats_idx,idx_train,idx_val))
        loss_train,acc_train,loss_val,acc_val,time_taken =\
            train(epoch=epoch,model=model,optimizer=optimizer,
                  features=input_vecs,adj=input_adj_coo_t.float(),
                  labels=cats_idx,idx_train=idx_train,idx_val=idx_val)
        collect()
        # torch.empty_cache()
        train_losses.append(loss_train)
        train_accs.append(acc_train)
        val_losses.append(loss_val)
        val_accs.append(acc_val)
        train_times.append(time_taken)
        logger.info(
            "\nLayer1 weights sum:[{}] \nLayer2 weights sum:[{}]".format(
                torch.sum(model.gc1.weight.data),
                torch.sum(model.gc2.weight.data)))
    logger.info("Optimization Finished!")
    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    plot_occurance(train_losses,
                   plot_name="train_losses_" + str(args.epochs) + ".jpg",
                   title="Train Losses",plot_dir=str(args.epochs))
    plot_occurance(train_accs,
                   plot_name="train_accs_" + str(args.epochs) + ".jpg",
                   ylabel="Accuracy",title="Train Accuracy",
                   plot_dir=str(args.epochs))
    plot_occurance(val_losses,
                   plot_name="val_losses_" + str(args.epochs) + ".jpg",
                   title="Validation Losses",plot_dir=str(args.epochs))
    plot_occurance(val_accs,plot_name="val_accs_" + str(args.epochs) + ".jpg",
                   ylabel="Accuracy",title="Validation Accuracy",
                   plot_dir=str(args.epochs))
    plot_occurance(train_times,
                   plot_name="train_time_" + str(args.epochs) + ".jpg",
                   ylabel="Time",title="Train Time",plot_dir=str(args.epochs))

    # Testing
    test(model,input_vecs,input_adj_coo_t.float(),cats_idx,idx_test)


if __name__ == '__main__':
    parser = ArgumentParser(description="Main script to setup and call XCGCN",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            epilog="Example: python MNXC_input.py --dataset_url /Users/monojitdey/Downloads/ "
                                   "--dataset_name Wiki10-31K --test_file /Wiki10/wiki10_test.txt"
                                   "--pretrain_dir /pretrain/glove6B.txt")
    # Config arguments
    parser.add_argument('--no-cuda',action='store_true',
                        default=config["model"]["use_cuda"],
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode',action='store_true',default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed',type=int,default=seed,help='Random seed.')
    parser.add_argument('--epochs',type=int,
                        default=config["sampling"]["num_epochs"],
                        help='Number of epochs to train.')
    parser.add_argument('--lr',type=float,
                        default=config["model"]["optimizer"]["learning_rate"],
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay',type=float,
                        default=config["model"]["optimizer"]["weight_decay"],
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden',type=int,default=config["model"]["hid_size"],
                        help='Number of hidden units.')
    parser.add_argument('--dropout',type=float,
                        default=config["model"]["dropout"],
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    # logger.debug("Arguments: {}".format(args))
    main(args)
