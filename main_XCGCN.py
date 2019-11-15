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
from torch.autograd import Variable
import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt

from data_loaders.common_data_handler import Common_Data_Handler
from data_loaders.Prepare_Data import Prepare_Data
from config import configuration as config,seed
from config import platform as plat
from config import username as user
from logger import logger
from file_utils import File_Util

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
        # self.gc2 = GCN_Spectral(nhid,nclass)
        self.gc2 = GCN_Spectral(nhid,nfeat)
        self.dropout = dropout

    def forward(self,x: torch.Tensor,adj: torch.Tensor) -> (
            torch.Tensor,torch.Tensor):
        x = F.relu(self.gc1(x,adj))
        x = F.dropout(x,self.dropout,training=self.training)
        x = self.gc2(x,adj)
        return F.log_softmax(x,dim=1),x


# class GCN_vectors(nn.Module):
#     def __init__(self,nfeat: int,nhid: int,nclass: int,dropout: float) -> None:
#         super(GCN_vectors,self).__init__()
#         self.gc1 = GCN_Spectral(nfeat,nhid)
#         self.gc2 = GCN_Spectral(nhid,nclass)
#         self.dropout = dropout
#
#     def forward(self,x: torch.Tensor,adj: torch.Tensor) -> (torch.Tensor,torch.Tensor):
#         x = F.relu(self.gc1(x,adj))
#         x = F.dropout(x,self.dropout,training=self.training)
#         x = self.gc2(x,adj)
#         return F.log_softmax(x,dim=1),x


def accuracy_emb(output: torch.Tensor,labels: torch.Tensor) -> torch.Tensor:
    # preds = output.max(1)[1].type_as(labels)
    equals = torch.eq(output,labels)
    correct = torch.sum(equals)
    # correct = preds.eq(labels).double()
    # correct = correct.sum()
    return correct.item() / labels.shape[1]


def accuracy_emb_np(output: np.ndarray,labels: np.ndarray) -> np.ndarray:
    # preds = output.max(1)[1].type_as(labels)
    equals = np.equal(output,labels)
    correct = np.sum(equals)
    # correct = preds.eq(labels).double()
    # correct = correct.sum()
    return correct / labels.shape[0]


# def accuracy_emb(output: torch.Tensor,labels: torch.Tensor) -> torch.Tensor:
#     # preds = output.max(1)[1].type_as(labels)
#     equals = np.equal(output,labels)
#     correct = np.sum(equals)
#     # correct = preds.eq(labels).double()
#     # correct = correct.sum()
#     return correct / labels.size


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


def neighrest_neighbors(test_features,train_features,k=5):
    """

    :param test_features:
    :param train_features:
    :param k:
    :return:
    """
    from sklearn.neighbors import NearestNeighbors

    NN = NearestNeighbors(n_neighbors=k)
    NN.fit(train_features)
    closest_neighbors_from_train = NN.kneighbors(test_features,
                                                 return_distance=False)
    return closest_neighbors_from_train


def knn_lbl_majority(test_features,train_features,train_labels,k=5):
    """

    :param test_features:
    :param train_features:
    :param train_labels:
    :param k:
    :return:
    """
    from sklearn.neighbors import KNeighborsClassifier

    NN = KNeighborsClassifier(n_neighbors=k)
    NN.fit(train_features,train_labels)
    test_labels = NN.predict(test_features)
    ## To get probabilities: test_labels_probas = NN.predict_proba(test_features)
    return test_labels


def test_emb(model,train_features,test_features,labels,idx_train,idx_test):
    """ Generates vectors for test samples by multiplying with weights
        all layer weights or last layer weights.

    :param model:
    :param train_features:
    :param test_features:
    :param labels:
    :param test_labels:
    :param idx_test:
    """
    t = time.time()
    model.eval()
    features_wl12 = torch.mm(
        torch.mm(test_features[idx_test],model.gc1.weight.data),
        model.gc2.weight.data)
    # features_wl2 = torch.mm(test_features[idx_test],model.gc2.weight.data)
    # for idx in idx_test:
    #     features_wl12.append(torch.mm(model.gc2.weight.data,
    #                                   torch.mm(model.gc1.weight.data,
    #                                            test_features[idx])))
    #
    #     features_wl2.append(torch.mm(model.gc2.weight.data,test_features[idx]))

    # features_wl2 = torch.stack(features_wl2)
    # test_outputs_wl2 = knn_lbl_majority(features_wl2,train_features,
    #                                     labels[idx_train],k=5)
    # acc_test_wl2 = accuracy(test_outputs_wl2[idx_test],labels[idx_test])

    # features_wl12 = torch.stack(features_wl12)

    features_wl12 = features_wl12.detach().numpy()
    train_features = train_features.detach().numpy()

    test_outputs_wl12 = knn_lbl_majority(features_wl12,train_features,
                                         labels,k=5)
    # test_outputs_wl12 = torch.FloatTensor(np.stack(list(test_outputs_wl12)))
    test_outputs_wl12 = np.stack(list(test_outputs_wl12))
    labels = labels.detach().numpy()
    acc_test_wl12 = accuracy_emb_np(test_outputs_wl12,labels[idx_test])

    logger.info(("Test set results:",
                 # "acc_test_wl2= {:.4f}".format(acc_test_wl2.item()),
                 "acc_test_wl12= {:.4f}".format(acc_test_wl12.item()),
                 'test_time: {:.4f}s'.format(time.time() - t)))


def train(epoch: int,model,optimizer,features: torch.Tensor,adj: torch.Tensor,
          labels: torch.Tensor,idx_train: torch.Tensor,
          idx_val: torch.Tensor) -> tuple:
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
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output,features_gcn = model(features[idx_train],adj)
    loss_train = F.nll_loss(output[idx_train],labels[idx_train])
    acc_train = accuracy(output[idx_train],labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        ## Evaluate validation set performance separately, deactivates dropout
        ## during validation run.
        model.eval()
        output,features_gcn = model(features[idx_train],adj)

    loss_val = F.nll_loss(output[idx_val],labels[idx_val])
    acc_val = accuracy(output[idx_val],labels[idx_val])
    logger.info(('Epoch: {:04d}'.format(epoch + 1),
                 'loss_train: {:.4f}'.format(loss_train.item()),
                 'acc_train: {:.4f}'.format(acc_train.item()),
                 'loss_val: {:.4f}'.format(loss_val.item()),
                 'acc_val: {:.4f}'.format(acc_val.item()),
                 'time: {:.4f}s'.format(time.time() - t)))

    return loss_train.item(),acc_train.item(),loss_val.item(),acc_val.item(),\
           (time.time() - t)


def plot_grad_flow(named_parameters):
    """ Plug this API after the loss.backward() during the training as follows -

    loss = self.criterion(outputs, labels)
    loss.backward()
    plot_grad_flow(model.named_parameters())

    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
    """
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    # plt.legend([Line2D([0], [0], color="c", lw=4),
    #             Line2D([0], [0], color="b", lw=4),
    #             Line2D([0], [0], color="k", lw=4)],
    #             ['max-gradient', 'mean-gradient', 'zero-gradient'])



def train_emb(epoch: int,model,optimizer,features: torch.Tensor,
              adj: torch.Tensor,label_emb: torch.Tensor,labels: torch.Tensor,
              idx_train: torch.Tensor,idx_val: torch.Tensor) -> tuple:
    """

    :param labels: Original label indices.
    :param label_emb: Embeddings of labels.
    :param epoch:
    :param model:
    :param optimizer:
    :param features:
    :param adj:
    :param idx_train:
    :param idx_val:
    """
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output,features_gcn = model(features,adj)

    emb_sim = torch.ones(idx_train.shape)
    loss_train_emb = F.cosine_embedding_loss(features_gcn[idx_train],
                                             label_emb[idx_train],
                                             target=emb_sim)
    # acc_train = accuracy(output[idx_train],label_emb[idx_train])

    output_np = output.detach().numpy()
    labels_knn_emb = knn_lbl_majority(output_np[idx_train],output_np[idx_train],labels[idx_train],k=5)
    acc_train_wl12 = accuracy_emb_np(labels_knn_emb,labels[idx_train].detach().numpy())
    loss_train_emb.backward()
    optimizer.step()

    if not args.fastmode:
        ## Evaluate validation set performance separately, deactivates dropout
        ## during validation run.
        model.eval()
        output,features_gcn = model(features,adj)

    emb_sim = torch.ones(idx_val.shape)
    loss_val_emb = F.cosine_embedding_loss(features_gcn[idx_val],
                                           label_emb[idx_val],target=emb_sim)
    # acc_val = accuracy(output[idx_val],label_emb[idx_val])
    output_np = output.detach().numpy()
    labels_knn_emb = knn_lbl_majority(output_np[idx_val],output_np[idx_train],labels[idx_train],k=5)
    acc_val_wl12 = accuracy_emb_np(labels_knn_emb,labels[idx_val].detach().numpy())
    logger.info(('Epoch: {:04d}'.format(epoch + 1),
                 'loss_train: {:.4f}'.format(loss_train_emb.item()),
                 'acc_train: {:.4f}'.format(acc_train_wl12.item()),
                 'loss_val: {:.4f}'.format(loss_val_emb.item()),
                 'acc_val: {:.4f}'.format(acc_val_wl12.item()),
                 'time: {:.4f}s'.format(time.time() - t)))

    return loss_train_emb.item(),acc_train_wl12,loss_val_emb.item(),acc_val_wl12,(time.time() - t)


def create_lbl_embs(samples2cats_map,cats2vec_map):
    samples2cats_avg_map = {}
    samples2cats_avg = []
    for sample,cats in samples2cats_map.items():
        sample2cats_avg = []
        for cat in cats:
            sample2cats_avg.append(cats2vec_map[cat])
        samples2cats_avg_map[sample] = np.mean(sample2cats_avg,axis=0)
        samples2cats_avg.append(samples2cats_avg_map[sample])

    # File_Util.save_json(samples2cats_avg_map,dataset_name + "_samples2cats_avg_map",filepath=dataset_dir)
    return samples2cats_avg_map,np.stack(samples2cats_avg)


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
    logger.debug((len(txts2vec_map),len(cats2vec_map)))

    input_vecs,cats_hot,keys,cats_idx = data_formatter.get_input_batch(
        txts2vec_map,sample2cats,return_cat_indices=True,
        multi_label=False)
    logger.debug(input_vecs.shape)

    input_adj_coo = data_formatter.load_graph_data(keys)
    logger.debug(input_adj_coo.shape)

    idx_train = torch.LongTensor(range(int(input_vecs.shape[0] * 0.7)))
    idx_val = torch.LongTensor(range(int(input_vecs.shape[0] * 0.7),int(input_vecs.shape[0] * 0.8)))
    idx_test = torch.LongTensor(range(int(input_vecs.shape[0] * 0.8),int(input_vecs.shape[0])))
    # logger.debug(idx_train)
    # logger.debug(idx_val)
    # logger.debug(idx_test)

    # input_vecs = torch.FloatTensor(input_vecs)
    input_vecs = Variable(torch.from_numpy(input_vecs),requires_grad=True).float()
    cats_idx = Variable(torch.from_numpy(cats_idx),requires_grad=False).float()
    # cats_idx = torch.LongTensor(cats_idx)
    input_adj_coo_t = adj_csr2t_coo(input_adj_coo)
    # input_adj_coo_t = input_adj_coo_t.requires_grad
    logger.debug(input_adj_coo_t.shape)

    # Model and optimizer
    model = GCN(nfeat=input_vecs.shape[1],nhid=args.hidden,
                nclass=cats_hot.shape[1],dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),lr=args.lr,
                           weight_decay=args.weight_decay)

    filepath = config["paths"]["dataset_dir"][plat][user]
    dataset = config["data"]["dataset_name"]
    samples2cats_map = File_Util.load_json(filename=dataset + "_sample2cats",
                                           filepath=join(filepath,dataset))
    _,label_embs = create_lbl_embs(samples2cats_map,cats2vec_map)

    # label_embs = torch.FloatTensor(label_embs)
    label_embs = Variable(torch.from_numpy(label_embs),requires_grad=True).float()

    # Train model
    train_losses,train_accs,val_losses,val_accs,train_times = [],[],[],[],[]
    t_total = time.time()
    for epoch in range(args.epochs):
        # train_losses.append(train(epoch,model,optimizer,input_vecs,input_adj_coo_t.float(),cats_idx,idx_train,idx_val))
        # loss_train,acc_train,loss_val,acc_val,time_taken =\
        loss_train,acc_train,loss_val,acc_val,time_taken = train_emb(epoch=epoch,model=model,optimizer=optimizer,features=input_vecs,adj=input_adj_coo_t.float(),label_emb=label_embs,labels=cats_idx,idx_train=idx_train,idx_val=idx_val)
        collect()
        # torch.empty_cache()
        train_losses.append(loss_train)
        train_accs.append(acc_train)
        val_losses.append(loss_val)
        val_accs.append(acc_val)
        train_times.append(time_taken)
        logger.info(
            "\nLayer1 weights sum:[{}] \nLayer2 weights sum:[{}]".format(torch.sum(model.gc1.weight.data),torch.sum(model.gc2.weight.data)))
    logger.info("Optimization Finished!")
    _,train_features = model(input_vecs,input_adj_coo_t.float())
    # W1 = model.gc1.weight.data
    logger.info(
        "Layer 1 weight matrix shape: [{}]".format(model.gc1.weight.data.shape))
    logger.info(
        "Layer 2 weight matrix shape: [{}]".format(model.gc2.weight.data.shape))
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
    # test(model,input_vecs,input_adj_coo_t.float(),cats_idx,idx_test)
    test_emb(model=model,train_features=train_features,
             test_features=input_vecs,
             labels=cats_idx,idx_train=idx_train,idx_test=idx_test)


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
