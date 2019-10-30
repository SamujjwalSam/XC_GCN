# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Handles data preparation operations.

__description__ : Prepares the datasets as per requirements of model.
__project__     : XCGCN
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.
__classes__     : Prepare_Data
__variables__   :
__methods__     :
"""

import numpy as np
from random import shuffle
from os.path import join,isfile,exists
from sklearn.preprocessing import MultiLabelBinarizer
from collections import OrderedDict

from neighborhood import Neighborhood_Graph
from text_process import Text_Process
from text_process.text_encoder import Text_Encoder
from file_utils import File_Util
from logger import logger
from config import configuration as config
from config import platform as plat
from config import username as user


class Prepare_Data:
    """ Prepare data into proper format.

        Converts strings to vectors,
        Converts category ids to multi-hot vectors,
        etc.
    """

    def __init__(self,dataset_loader,dataset_name: str = config["data"]["dataset_name"],
                 dataset_dir: str = config["paths"]["dataset_dir"][plat][user]) -> None:
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.dataset_loader = dataset_loader

        self.txts2vec_map = None
        self.doc2vec_model = None
        self.cats_all = None
        self.txt_encoder_model = None
        self.txts_sel,self.sample2cats_sel,self.cats_sel = None,None,None

        self.graph = Neighborhood_Graph()
        self.txt_process = Text_Process()
        self.txt_encoder = Text_Encoder()
        self.mlb = MultiLabelBinarizer()

        # dataset_loader.gen_data_stats()
        self.oov_words_dict = OrderedDict()

    def load_graph_data(self,nodes):
        """ Loads graph data for XC datasets. """
        Docs_G = self.graph.load_doc_neighborhood_graph(nodes=nodes,get_stats=config["graph"]["stats"])
        Docs_adj_coo = self.graph.get_adj_matrix(Docs_G,adj_format='coo')
        # Docs_adj_coo_t = adj_csr2t_coo(Docs_adj_coo)
        return Docs_adj_coo

    @staticmethod
    def create_batch_repeat(X: dict,Y: dict,keys):
        """
        Generates batch from keys.

        :param X:
        :param Y:
        :param keys:
        :return:
        """
        batch_x = []
        batch_y = []
        shuffle(keys)
        for k in keys:
            batch_x.append(X[k])
            batch_y.append(Y[k])
        return np.stack(batch_x),np.stack(batch_y)

    def get_input_batch(self,txts:dict,sample2cats:dict,keys:list=None,return_cat_indices: bool = False,multi_label: bool = True) ->\
            [np.ndarray,np.ndarray,np.ndarray]:
        """Generates feature vectors of input documents.

        :param txts:
        :param sample2cats:
        :param keys:
        :param return_cat_indices:
        :param multi_label:
        :return:
        """
        if keys is None:
            sample_ids = list(txts.keys())
            batch_size = int(0.7 * len(sample_ids))
            _,keys = File_Util.get_batch_keys(sample_ids,batch_size=batch_size, remove_keys=False)
        txt_vecs_keys,sample2cats_keys = self.create_batch_repeat(txts,sample2cats,keys)
        sample2cats_keys_hot = self.mlb.transform(sample2cats_keys)

        if return_cat_indices:
            if multi_label:
                ## For Multi-Label, multi-label-margin loss
                cats_idx = [self.mlb.inverse_transform(sample2cats_keys_hot)]
            else:
                ## For Multi-Class, cross-entropy loss
                cats_idx = sample2cats_keys_hot.argmax(1)
            return txt_vecs_keys,sample2cats_keys_hot,keys,cats_idx

        return txt_vecs_keys,sample2cats_keys_hot,keys

    def invert_cat2samples(self,classes_dict: dict = None):
        """Converts sample : cats to cats : samples

        :returns: A dictionary of cats to sample mapping.
        """
        cat2id = OrderedDict()
        if classes_dict is None: classes_dict = self.sample2cats_sel
        for k,v in classes_dict.items():
            for cat in v:
                if cat not in cat2id:
                    cat2id[cat] = []
                cat2id[cat].append(k)
        return cat2id

    def create_vec_maps(self,txts:dict=None,cats:dict=None):
        """ Maps text and categories to their vector representation.

        :param txts:
        :param cats:
        :return:
        """
        logger.debug(join(self.dataset_dir,self.dataset_name,self.dataset_name + "_txts2vec_map.pkl"))
        if isfile(join(self.dataset_dir,self.dataset_name,self.dataset_name + "_txts2vec_map.pkl"))\
                and isfile(join(self.dataset_dir,self.dataset_name,self.dataset_name + "_cats2vec_map.pkl")):
            logger.info("Loading pre-processed mappings from: [{}] and [{}]"
                        .format(join(self.dataset_dir,self.dataset_name,self.dataset_name + "_txts2vec_map.pkl"),
                                join(self.dataset_dir,self.dataset_name,self.dataset_name + "_cat2vec_map.pkl")))
            txts2vec_map = File_Util.load_pickle(self.dataset_name + "_txts2vec_map",
                                                 filepath=join(self.dataset_dir,self.dataset_name))
            cats2vec_map = File_Util.load_pickle(self.dataset_name + "_cats2vec_map",
                                                 filepath=join(self.dataset_dir,self.dataset_name))
        else:
            if txts is None or cats is None:
                txts,_,_,cats = self.load_raw_data(load_type='all',return_values=True)
            ## Generate txts2vec_map and cats2vec_map
            logger.info("Generating pre-processed mappings.")
            txts2vec_map = self.txt_process.gen_sample2vec_map(txts=txts)
            catid2cattxt = File_Util.inverse_dict_elm(cats)
            cats2vec_map = self.txt_process.gen_cats2vec_map(cats=catid2cattxt)

            logger.info("Saving pre-processed mappings to: [{}] and [{}]"
                        .format(join(self.dataset_dir,self.dataset_name,self.dataset_name + "_txts2vec_map.pkl"),
                                join(self.dataset_dir,self.dataset_name,self.dataset_name + "_cat2vec_map.pkl")))
            File_Util.save_pickle(txts2vec_map,self.dataset_name + "_txts2vec_map",
                                  filepath=join(self.dataset_dir,self.dataset_name))
            File_Util.save_pickle(cats2vec_map,self.dataset_name + "_cats2vec_map",
                                  filepath=join(self.dataset_dir,self.dataset_name))
        return txts2vec_map,cats2vec_map

    def load_raw_data(self,load_type: str = 'all', return_values=True):
        """ Loads the json data provided by param "load_type".

        :param return_values:
        :param load_type: Which data to load: Options: ['train', 'val', 'test']
        """
        self.txts_sel,self.sample2cats_sel,self.cats_sel,self.cats_all = self.dataset_loader.get_data(load_type=load_type)
        self.remain_sample_ids = list(self.txts_sel.keys())
        self.cat2sample_map = self.invert_cat2samples(self.sample2cats_sel)
        self.remain_cat_ids = list(self.cats_sel.keys())

        ## MultiLabelBinarizer only take list of list as input. Need to convert "list of int" to "list of list".
        cat_ids = []
        for cat_id in self.cats_all.values():
            cat_ids.append([cat_id])
        self.mlb.fit(cat_ids)

        self.idf_dict = self.txt_process.calculate_idf_per_token(docs=self.txts_sel.values())

        if return_values:
            return self.txts_sel,self.sample2cats_sel,self.cats_sel,self.cats_all

    def normalize_inputs(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1.

        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        logger.debug(
            ("train_shape",self.x_train.shape,"test_shape",self.x_test.shape,"val_shape",self.x_val.shape))
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_val = (self.x_val - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

    def create_multihot(self,batch_classes_dict):
        """
        Creates multi-hot vectors for a batch of data.

        :param batch_classes_dict:
        :return:
        """
        classes_multihot = self.mlb.fit_transform(batch_classes_dict.values())
        return classes_multihot


if __name__ == '__main__':
    logger.debug("Preparing Data...")
    from data_loaders.common_data_handler import Common_Data_Handler

    data_loader = Common_Data_Handler(dataset_type=config["xc_datasets"][config["data"]["dataset_name"]],
                                      dataset_name=config["data"]["dataset_name"],
                                      dataset_dir=config["paths"]["dataset_dir"][plat][user])

    data_formatter = Prepare_Data(dataset_loader=data_loader,
                                  dataset_name=config["data"]["dataset_name"],
                                  dataset_dir=config["paths"]["dataset_dir"][plat][user])

    data_formatter.load_raw_data(load_type='val')
    Adj = data_formatter.load_graph_data()
    features,labels = data_formatter.get_input_batch()
    logger.debug(Adj.shape)
    logger.debug(features.shape)
