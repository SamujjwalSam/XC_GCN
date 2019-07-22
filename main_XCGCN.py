# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Spectral Graph Convolutional Layer

__description__ : Spectral Graph Convolutional Layer
__project__     : XCGCN
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : GCN_Spectral

__variables__   :

__methods__     :
"""

import numpy as np
from torch.autograd import Variable
from collections import namedtuple
from networkx import read_edgelist,set_node_attributes
from pandas import read_csv,Series
from numpy import array

from logger import logger

DataSet = namedtuple('DataSet',field_names=['X_train','y_train','X_test','y_test','network'])


def load_karate_club():
    """

    :return:
    """
    network = read_edgelist(
        'karate.edgelist',
        nodetype=int)

    attributes = read_csv(
        'karate.attributes.csv',
        index_col=['node'])

    for attribute in attributes.columns.values:
        set_node_attributes(
            network,
            values=Series(
                attributes[attribute],
                index=attributes.index).to_dict(),
            name=attribute
        )

    X_train,y_train = map(array,zip(*[
        ([node],data['role'] == 'Administrator')
        for node,data in network.nodes(data=True)
        if data['role'] in {'Administrator','Instructor'}
    ]))
    X_test,y_test = map(array,zip(*[
        ([node],data['community'] == 'Administrator')
        for node,data in network.nodes(data=True)
        if data['role'] == 'Member'
    ]))

    y_train_idx = []
    for val in y_train:
        if val:
            y_train_idx.append([1])
        else:
            y_train_idx.append([0])
    y_train_idx = np.stack(y_train_idx)

    y_test_idx = []
    for val in y_test:
        if val:
            y_test_idx.append([1])
        else:
            y_test_idx.append([0])
    y_test_idx = np.stack(y_test_idx)

    return DataSet(
        X_train,y_train,
        X_test,y_test,
        network)


from networkx import to_numpy_matrix,shortest_path_length
import torch

zkc = load_karate_club()

A = to_numpy_matrix(zkc.network)
# A = torch.tensor(A).type(torch.DoubleTensor)

X_train = zkc.X_train
y_train = zkc.y_train
X_test = zkc.X_test
y_test = zkc.y_test

import math
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GCN_Spectral(nn.Module):
    """ Custom class for GCN. """

    def __init__(self,A: np.matrix,in_units: int,out_units: int,bias=True) -> None:
        super(GCN_Spectral,self).__init__()

        self.in_units,self.out_units = in_units,out_units

        A_torch = torch.from_numpy(A)
        I = torch.eye(*A.shape).type(torch.DoubleTensor)
        A_hat = A_torch + I
        D = A_hat.sum(dim=0)
        D_inv = D ** -0.5
        D_inv = torch.diag(D_inv)
        A_hat = D_inv * A_hat * D_inv
        self.A_hat = torch.tensor(A_hat).type(torch.DoubleTensor)

        self.W = Parameter(torch.rand(self.in_units,self.out_units).type(torch.FloatTensor))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_units))
        else:
            self.register_parameter('bias',None)

        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.W,a=math.sqrt(1))
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
        #     bound = 1 / math.sqrt(fan_in)
        #     init.uniform_(self.bias, -bound, bound)

    def forward(self,X: torch.Tensor) -> torch.Tensor:
        """

        :param X:
        :return:
        """
        # support = torch.mm(self.A_hat,X)
        # propagate = torch.tanh((torch.mm(support,self.W)))
        # return propagate
        support = torch.mm(X,self.W)
        output = torch.spmm(self.A_hat,support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_units) + ' -> ' + str(self.out_units) + ')'

    def extra_repr(self):
        return self.__class__.__name__ + 'in_units={}, out_units={}, bias={}'.format(self.in_units, self.out_units,
                                                                                     self.bias is not None)


class LogisticRegressor(nn.Module):
    """ Logistic regression layer to be appended after GCN layers for classification. """

    def __init__(self,in_units: int) -> None:
        super(LogisticRegressor,self).__init__()
        self.w = Parameter(torch.rand(1,in_units).type(torch.DoubleTensor))
        self.b = Parameter(torch.rand(1,1).type(torch.DoubleTensor))
        logger.debug((self.w,self.b))

    def forward(self,X: torch.Tensor):
        """

        :param X:
        :return:
        """
        # Broadcast b to MM size.
        # b = torch.broadcast_axis(self.b,axis=(0,1),size=(34,1))
        y = torch.mm(X,self.w.t()) + self.b

        return torch.sigmoid(y)


# from Weight_Init import weight_init


def build_model(A: np.matrix,input_size: int):
    """ Builds GCN model with 2 GCN layers.

    :param A: Adjecency mat.
    :param input_size: node feature dim.
    :return:
    """
    model_features = nn.Sequential(
        GCN_Spectral(A,in_units=input_size,out_units=4),
        GCN_Spectral(A,in_units=4,out_units=2)
    )
    # model_features_params = list(model_features.parameters())
    out_units = 2
    logger.info("GCN Summary: \n{}".format(model_features))

    classifier = LogisticRegressor(out_units)
    model = nn.Sequential(model_features,classifier)
    # model_params = list(model.parameters())
    logger.info("GCN + LR Summary: \n{}".format(model))

    # weight_init(model)

    return model,model_features


# ## Model 1: Identity Matrix as Features
X_1 = I = torch.eye(*A.shape).type(torch.DoubleTensor)
model_1,features_1 = build_model(A,X_1.shape[1])
logger.debug(model_1(X_1))

# ## Model 2: Distance to Administrator and Instructor as Additional Features
X_2 = torch.zeros((A.shape[0],2)).type(torch.DoubleTensor)
node_distance_instructor = shortest_path_length(zkc.network,target=33)
node_distance_administrator = shortest_path_length(zkc.network,target=0)

for node in zkc.network.nodes():
    X_2[node][0] = node_distance_administrator[node]
    X_2[node][1] = node_distance_instructor[node]

X_2 = torch.cat((X_1,X_2),1)
model_2,features_2 = build_model(A,X_2.shape[1])
model_2(X_2)


def SigmoidBinaryCrossEntropyLoss(pred: torch.Tensor,label: torch.Tensor,batch_axis: int = 0):
    """ Calculates loss based on MXNet SigmoidBinaryCrossEntropyLoss.

    https://beta.mxnet.io/_modules/mxnet/gluon/loss.html#SigmoidBinaryCrossEntropyLoss

    :param batch_axis:
    :param pred:
    :param label:
    :return:
    """
    # label = torch._reshape_like(label,pred)

    loss = -(torch.log(pred + 1e-12) * label + torch.log(1. - pred + 1e-12) * (1. - label))

    # loss = torch.relu(pred) - pred * label + torch.Activation(-torch.abs(pred),act_type='softrelu')
    return torch.mean(loss,dim=batch_axis)


# # Train and Test Models
def train(model: torch.nn.modules.container.Sequential,model_features: torch.nn.modules.container.Sequential,
          X: torch.Tensor,X_train_np: np.ndarray,y_train_np: np.ndarray,epochs: int):
    """

    :param X_train_np:
    :param y_train_np:
    :param model:
    :param model_features:
    :param X:
    :param X_train:
    :param y_train:
    :param epochs:
    :return:
    """
    # optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=1)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=1)

    # feature_representations = [model_features(X).copy_()]
    feature_representations = []

    for e in range(1,epochs + 1):
        optimizer.zero_grad()
        cum_loss = 0
        cum_preds = []

        X = Variable(X)
        # X2 = X.copy()
        X_train = torch.from_numpy(X_train_np)
        y_train = torch.from_numpy(y_train_np).type(torch.FloatTensor)

        for i,x in enumerate(X_train.flatten()):
            y = y_train[i]
            preds = model(X)
            pred = preds[x]
            logger.debug((pred.float().unsqueeze(0).shape,y.long().unsqueeze(0).shape))
            logger.debug((pred.float().unsqueeze(0),y.long().unsqueeze(0)))
            loss = F.nll_loss(pred.float().unsqueeze(0),y.long().unsqueeze(0))
            # loss = SigmoidBinaryCrossEntropyLoss(pred.float(),y.unsqueeze(0))
            # loss = F.binary_cross_entropy_with_logits(pred.float(),y.unsqueeze(0))
            ## MXNet loss: https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SigmoidBinaryCrossEntropyLoss
            # MXNet_loss = torch.mean(F.relu(pred) - pred * label + torch.Activation(-F.abs(pred), act_type='softrelu'), axis=self._batch_axis, exclude=True)
            # loss = F.cross_entropy(pred.float(),y.unsqueeze(0))
            # logger.debug("x:[{}], y:[{}], pred:[{}], loss:[{}]".format(x,y,pred.data.numpy(),loss.item()))
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            cum_preds += [pred.item()]

        feature_representations.append(model_features(X).data.numpy())

        if (e % (epochs // 5)) == 0:
            logger.debug(f"Epoch {e}/{epochs} -- Loss: {cum_loss: f}")
            logger.debug(cum_preds)
    return feature_representations


def predict(model,X,nodes):
    """

    :param model:
    :param X:
    :param nodes:
    :return:
    """
    preds = model(X)[nodes].data.numpy().flatten()
    return np.where(preds >= 0.5,1,0)


from sklearn.metrics import classification_report

# ## Performance of Model 1
feature_representations_1 = train(model_1,features_1,X_1,X_train,y_train,epochs=5000)
y_pred_1 = predict(model_1,X_1,X_test)
logger.debug(classification_report(y_test,y_pred_1))

# ## Performance of Model 2
feature_representations_2 = train(model_2,features_2,X_2,X_train,y_train,epochs=250)
y_pred_2 = predict(model_2,X_2,X_test)
logger.debug(classification_report(y_test,y_pred_2))
