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

    def load_graph_data(self):
        """ Loads graph data for XC datasets. """
        Docs_G = self.graph.load_doc_neighborhood_graph(get_stats=config["graph"]["stats"])
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

    def get_features(self,return_cat_indices=False,multi_label=False):
        """ Generates feature vectors of input documents. """
        keys = list(self.classes_selected.keys())
        train_sentences,train_classes = self.create_batch_repeat(self.sentences_selected,self.classes_selected,keys)
        features = self.txt2vec(train_sentences)
        categories_hot = self.mlb.transform(train_classes)

        if return_cat_indices:
            if multi_label:
                ## For Multi-Label, multi-label-margin loss
                cats_idx = [self.mlb.inverse_transform(sample2cats_keys_hot)]
            else:
                ## For Multi-Class, cross-entropy loss
                cats_idx = sample2cats_keys_hot.argmax(1)
            return txt_vecs_keys,sample2cats_keys_hot,cats_idx

        return txt_vecs_keys,sample2cats_keys_hot

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

        ## MultiLabelBinarizer only takes list of lists as input. Need to convert our list of ints to list of lists.
        cat_ids = []
        for cat_id in self.cats_all.values():
            cat_ids.append([cat_id])
        self.mlb.fit(cat_ids)

        # self.idf_dict = self.clean.calculate_idf(docs=self.sentences_selected.values())
        if return_loaded:
            return self.sentences_selected,self.classes_selected,self.categories_selected,self.categories_all

    def txt2vec(self,sentences: list,vectorizer=config["prep_vecs"]["vectorizer"],
                tfidf_avg=config["prep_vecs"]["tfidf_avg"]):
        """
        Creates vectors from input_texts based on [vectorizer].

        :param max_vec_len: Maximum vector length of each document.
        :param num_chunks: Number of chunks the input_texts are to be divided. (Applicable only when vectorizer = "chunked")
        :param input_size: Embedding dimension for each word.
        :param sentences:
        :param vectorizer: Decides how to create the vector.
            "chunked" : Partitions the whole text into equal length chunks and concatenates the avg of each chunk.
                        vec_len = num_chunks * input_size
                        [["these", "are"], ["chunks", "."]]

            "sentences" : Same as chunked except each sentence forms a chunk. "\n" is sentence separator.
                        vec_len = max(num_sents, max_len) * input_size
                        ["this", "is", "a", "sentence", "."]
                        NOTE: This will create variable length vectors.

            "concat" : Concatenates the vectors of each word, adds padding to make equal length.
                       vec_len = max(num_words) * input_size

            "word_avg" : Take the average of vectors of all the words.
                       vec_len = input_size
                       [["these"], ["are"], ["words"], ["."]]

            "doc2vec" : Use Gensim Doc2Vec to generate vectors.
                        https://radimrehurek.com/gensim/models/doc2vec.html
                        vec_len = input_size
                        ["this", "is", "a", "document", "."]

        :param tfidf_avg: If tf-idf weighted avg is to be taken or simple.
            True  : Take average based on each words tf-idf value.
            False : Take simple average.

        :returns: Vector length, numpy.ndarray(batch_size, vec_len)
        """
        # sentences = util.clean_sentences(sentences, specials="""_-@*#'"/\\""", replace='')
        if vectorizer == "doc2vec":
            if self.doc2vec_model is None:
                self.doc2vec_model = self.text_encoder.load_doc2vec(sentences)
            vectors_dict = self.text_encoder.get_doc2vecs(sentences,self.doc2vec_model)
            return vectors_dict
        elif vectorizer == "word2vec":
            w2v_model = self.text_encoder.load_word2vec()
            sentences = list(filter(None,sentences))  ## Removing empty items.
            return self.create_doc_vecs(sentences,w2v_model=w2v_model)
        else:
            raise Exception("Unknown vectorizer: [{}]. \n"
                            "Available options: ['doc2vec','word2vec']"
                            .format(vectorizer))

    oov_words_dict = {}

    def create_doc_vecs(self,sentences: list,w2v_model,use_idf=config["prep_vecs"]["idf"],concat_axis=0,
                        input_size=config["prep_vecs"]["input_size"]):
        """
        Calculates the average of vectors of all words within a chunk and concatenates the chunks.

        :param use_idf: Flag to decide if idf is to be used.
        :param input_size:
        :param concat_axis: The axis the vectors should be concatenated.
        :param sents_chunk_mode:
        :param w2v_model:
        :param sentences: Dict of texts.
        :returns: Average of vectors of chunks. Dim: input_size.
        """
        docs_vecs = []  # List to hold generated vectors.
        for i,doc in enumerate(sentences):
            chunks = self.partition_doc(doc)
            chunks = list(filter(None,chunks))  ## Removing empty items.
            vecs = self.sum_word_vecs(chunks,w2v_model)
            docs_vecs.append(vecs)
        return np.stack(docs_vecs)

    def sum_word_vecs(self,words: list,w2v_model,input_size=config["prep_vecs"]["input_size"],avg=True):
        """ Generates a vector of [input_size] using the [words] and [w2v_model].

        :param avg: If average to be calculated.
        :param words: List of words
        :param w2v_model: Word2Vec model by Gensim
        :param input_size: Dimension of each vector
        :return: Vector of sum of all the words.
        """
        # oov_words_dict = {}  ## To hold out-of-vocab words.
        sum_vec = None
        for i,word in enumerate(words):
            ## Multiply idf of that word with the vector
            try:  ## If word exists in idf dict
                idf = self.idf_dict[word]
            except KeyError as e:  ## If word does not exists in idf_dict, multiply 1
                # logger.info("[{}] not found in idf_dict.".format(word))
                idf = 1
            if word in w2v_model.vocab:  ## If word is present in model
                if sum_vec is None:
                    sum_vec = w2v_model[word] * idf
                else:
                    sum_vec = np.add(sum_vec,w2v_model[word] * idf)
            elif word in Prepare_Data.oov_words_dict:  ## If word is OOV
                if sum_vec is None:
                    sum_vec = Prepare_Data.oov_words_dict[word] * idf
                else:
                    sum_vec = np.add(sum_vec,Prepare_Data.oov_words_dict[word] * idf)
            else:  ## New unknown word, need to create random vector.
                new_oov_vec = np.random.uniform(-0.5,0.5,input_size)
                # w2v_model.add(word, new_oov_vec)  ## For some reason, gensim word2vec.add() not working.
                Prepare_Data.oov_words_dict[word] = new_oov_vec
                if sum_vec is None:
                    sum_vec = Prepare_Data.oov_words_dict[word] * idf
                else:
                    sum_vec = np.add(sum_vec,Prepare_Data.oov_words_dict[word] * idf)
        if avg:
            sum_vec = np.divide(sum_vec,float(len(words)))

        return np.stack(sum_vec)

    def partition_doc(self,sentence: str,sents_chunk_mode: str = config["text_process"]["sents_chunk_mode"],
                      num_chunks: int = config["prep_vecs"]["num_chunks"]) -> list:
        """
        Divides a document into chunks based on the vectorizer.

        :param num_chunks:
        :param sentence:
        :param sents_chunk_mode:
        :param doc_len:
        :return:
        """
        chunks = []
        if sents_chunk_mode == "concat":
            words = self.txt_process.tokenizer_spacy(sentence)
            for word in words:
                chunks.append(word)
        elif sents_chunk_mode == "word_avg":
            chunks = self.txt_process.tokenizer_spacy(sentence)
        elif sents_chunk_mode == "sentences":
            chunks = self.txt_process.sents_split(sentence)
        elif sents_chunk_mode == "chunked":
            splitted_doc = self.txt_process.tokenizer_spacy(sentence)
            doc_len = len(splitted_doc)
            chunk_size = doc_len // num_chunks  ## Calculates how large each chunk should be.
            index_start = 0
            for i in range(num_chunks):
                batch_portion = doc_len / (chunk_size * (i + 1))
                if batch_portion > 1.0:
                    index_end = index_start + chunk_size
                else:  ## Available data is less than chunk_size
                    index_end = index_start + (doc_len - index_start)
                logger.info('Making chunk of tokens from [{0}] to [{1}]'.format(index_start,index_end))
                chunk = splitted_doc[index_start:index_end]
                chunks.append(chunk)
                index_start = index_end
        else:
            raise Exception("Unknown document partition mode: [{}]. \n"
                            "Available options: ['concat','word_avg (Default)','sentences','chunked']"
                            .format(sents_chunk_mode))
        chunks = list(filter(None,chunks))  ## Removes empty items, like: ""
        return chunks

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
