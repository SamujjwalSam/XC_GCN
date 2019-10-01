# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Class to process and load pretrained models.

__description__ : Class to process and load pretrained models.
__project__     : XCGCN
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : Text_Encoder
"""

import numpy as np
from os import mkdir
from os.path import join,exists,split

import gensim
from gensim.models import word2vec,doc2vec
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.utils import simple_preprocess

from pytorch_pretrained_bert import BertTokenizer,BertModel,BertForMaskedLM

from logger import logger
from config import configuration as config
from config import platform as plat
from config import username as user

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class Text_Encoder:
    """
    Class to process and load pretrained models.

    Supported models: glove, word2vec, fasttext, googlenews, bert, lex, etc.
    """

    def __init__(self,model_type: str = "googlenews",model_dir: str = config["paths"]["pretrain_dir"][plat][user],
                 embedding_dim: int = config["prep_vecs"]["input_size"]):
        """
        Initializes the pretrain class and checks for paths validity.

        Args:
            model_type : Path to the file containing the html files.
            Supported model_types:
                glove (default)
                word2vec
                fasttext_wiki
                fasttext_crawl
                fasttext_wiki_subword
                fasttext_crawl_subword
                lex_crawl
                lex_crawl_subword
                googlenews
                bert_multi
                bert_large_uncased
        """
        super(Text_Encoder,self).__init__()
        self.model_type = model_type
        self.model_dir = model_dir
        self.embedding_dim = embedding_dim

        if model_type == "googlenews":
            filename = "GoogleNews-vectors-negative300.bin"
            binary_file = True
        elif model_type == "glove":
            filename = "glove.6B.300d.txt"
            binary_file = False
        elif model_type == "fasttext_wiki":
            filename = "wiki-news-300d-1M.vec"
            binary_file = False
        elif model_type == "fasttext_crawl":
            filename = "crawl-300d-2M.vec.zip"
            binary_file = False
        elif model_type == "fasttext_wiki_subword":
            filename = "wiki-news-300d-1M-subword.vec.zip"
            binary_file = False
        elif model_type == "fasttext_crawl_subword":
            filename = "crawl-300d-2M-subword.vec.zip"
            binary_file = False
        elif model_type == "bert_multi":
            filename = "BERT_multilingual_L-12_H-768_A-12.zip"
            binary_file = True
        elif model_type == "bert_large_uncased":
            filename = "BERT_large_uncased_L-24_H-1024_A-16.zip"
            binary_file = True
        else:
            raise Exception("Unknown pretrained model type: [{}]".format(model_type))
        # logger.debug("Creating Text_Encoder.")
        self.model_file_name = filename
        self.binary = binary_file
        self.pretrain_model = None
        # self.pretrain_model = self.load_word2vec(self.model_dir, model_file_name=self.model_file_name, model_type=model_type)

    def load_bert_pretrained(self,txts):
        # Load pre-trained model tokenizer (vocabulary)
        # marked_txts = self.add_spl_tokens(txts)
        for idx,txt in txts.items():
            txts[idx] = "[CLS] " + txt + " [SEP]"
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_text = tokenizer.tokenize(list(txts.values()))
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # from keras.preprocessing.sequence import pad_sequences
        # input_ids = pad_sequences(indexed_tokens, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length.

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        """
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_examples_to_features(self,txts,label_list,max_seq_length,tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        label_map = {label : i for i, label in enumerate(label_list)}

        features = []
        for (idx, txt) in enumerate(txts):
            tokens_a = tokenizer.tokenize(txt.text_a)

            tokens_b = None
            if txt.text_b:
                tokens_b = tokenizer.tokenize(txt.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            labels_ids = []
            for label in txt.labels:
                labels_ids.append(float(label))

    #         label_id = label_map[txt.label]
            if idx < 0:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (txt.guid))
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %s)" % (txt.labels, labels_ids))

            features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_ids=labels_ids))
        return features

    def load_doc2vec_model(self,documents,vector_size=config["prep_vecs"]["input_size"],
                           window=config["prep_vecs"]["window"],
                           min_count=config["prep_vecs"]["min_count"],workers=config["text_process"]["workers"],seed=0,
                           clean_tmp=False,save_model=True,
                           doc2vec_model_file=config["data"]["dataset_name"] + "_doc2vec",
                           doc2vec_dir=join(config["paths"]["dataset_dir"][plat][user],config["data"]["dataset_name"]),
                           negative=config["prep_vecs"]["negative"]):
        """
        Generates vectors from documents.
        https://radimrehurek.com/gensim/models/doc2vec.html

        :param save_model:
        :param clean_tmp: Flag to set if cleaning is to be done.
        :param doc2vec_dir:
        :param doc2vec_model_file: Name of Doc2Vec model.
        :param negative: If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20).
        :param documents:
        :param vector_size:
        :param window:
        :param min_count:
        :param workers:
        :param seed:
        """
        full_model_name = doc2vec_model_file + "_" + str(vector_size) + "_" + str(window) + "_" + str(min_count) + "_"\
                          + str(negative)
        if exists(join(doc2vec_dir,full_model_name)):
            logger.info("Loading doc2vec model [{}] from: [{}]".format(full_model_name,doc2vec_dir))
            doc2vec_model = doc2vec.Doc2Vec.load(join(doc2vec_dir,full_model_name))
        else:
            train_corpus = list(self.read_corpus(documents))
            doc2vec_model = doc2vec.Doc2Vec(train_corpus,vector_size=vector_size,window=window,min_count=min_count,
                                            workers=workers,seed=seed,negative=negative)
            # doc2vec_model.build_vocab(train_corpus)
            doc2vec_model.train(train_corpus,total_examples=doc2vec_model.corpus_count,epochs=doc2vec_model.epochs)
            if save_model:
                save_path = get_tmpfile(join(doc2vec_dir,full_model_name))
                doc2vec_model.save(save_path)
                logger.info("Saved doc2vec model to: [{}]".format(save_path))
            if clean_tmp:  ## when finished training a model (no more updates, only querying, reduce memory usage)
                doc2vec_model.delete_temporary_training_data(keep_doctags_vectors=True,keep_inference=True)
        return doc2vec_model

    @staticmethod
    def read_corpus(documents,tokens_only=False):
        """
        Read the documents, pre-process each line using a simple gensim pre-processing tool and return a list of words. The tag is simply the zero-based line number.

        :param documents: List of documents.
        :param tokens_only:
        """
        for i,line in enumerate(documents):
            if tokens_only:
                yield simple_preprocess(line)
            else:  ## For training data, add tags, tags are simply zero-based line number.
                yield doc2vec.TaggedDocument(simple_preprocess(line),[i])

    def load_word2vec(self,model_dir: str = config["paths"]["pretrain_dir"][plat][user],model_type: str = 'googlenews',
                      encoding: str = 'latin-1',model_file_name: str = "GoogleNews-vectors-negative300.bin") ->\
            gensim.models.keyedvectors.Word2VecKeyedVectors:
        """
        Loads Word2Vec model and returns initial weights for embedding layer.

        inputs:
        model_type      # GoogleNews / glove
        embedding_dim    # Word vector dimensionality
        """
        if self.pretrain_model is not None: return self.pretrain_model

        assert exists(join(model_dir,model_file_name)),"Model file not found at: [{}].".format(
            join(model_dir,model_file_name))
        logger.debug("Using [{0}] model from [{1}]".format(model_type,join(model_dir,model_file_name)))
        if model_type == 'googlenews' or model_type == "fasttext_wiki":
            if exists(join(model_dir,model_file_name + '.bin')):
                try:
                    pretrain_model = FastText.load_fasttext_format(
                        join(model_dir,model_file_name + '.bin'))  ## For original fasttext *.bin format.
                except Exception as e:
                    pretrain_model = KeyedVectors.load_word2vec_format(join(model_dir,model_file_name + '.bin'),
                                                                       binary=True,encoding=encoding)
            else:
                try:
                    pretrain_model = KeyedVectors.load_word2vec_format(join(model_dir,model_file_name),
                                                                       binary=self.binary)
                except Exception as e:  ## On exception, trying a different format.
                    logger.info('Loading original word2vec format failed. Trying Gensim format.')
                    pretrain_model = KeyedVectors.load(join(model_dir,model_file_name))
                ## Save model in binary format for faster loading in future.
                pretrain_model.save_word2vec_format(join(model_dir,model_file_name + ".bin"),binary=True)
                logger.info("Saved binary model at: [{0}]".format(join(model_dir,model_file_name + ".bin")))
                logger.info(type(pretrain_model))

        elif model_type == 'glove':
            logger.info('Loading existing Glove model: [{0}]'.format(join(model_dir,model_file_name)))
            from gensim.scripts.glove2word2vec import glove2word2vec
            from gensim.test.utils import datapath,get_tmpfile

            glove_file = datapath(join(model_dir,model_file_name))
            tmp_file = get_tmpfile(join(model_dir,model_file_name + "_word2vec"))
            _ = glove2word2vec(glove_file,tmp_file)
            pretrain_model = KeyedVectors.load_word2vec_format(tmp_file)

        elif model_type == "bert_multi":
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            bert_model.eval()
            # pretrain_model = FastText.load_fasttext_format(join(model_dir,model_file_name))
            # pretrain_model = FastText.load_binary_data (join(model_dir,model_file_name))
            pretrain_model = KeyedVectors.load_word2vec_format(join(model_dir,model_file_name),binary=False)
            # import io
            # fin = io.open(join(model_dir, model_file_name), encoding=encoding, newline=newline,
            #               errors=errors)
            # n, d = map(int, fin.readline().split())
            # pretrain_model = OrderedDict()
            # for line in fin:
            #     tokens = line.rstrip().split(' ')
            #     pretrain_model[tokens[0]] = map(float, tokens[1:])
            """embedding_dict = gensim.models.KeyedVectors.load_word2vec_format(dictFileName, binary=False) embedding_dict.save_word2vec_format(dictFileName+".bin", binary=True) embedding_dict = gensim.models.KeyedVectors.load_word2vec_format(dictFileName+".bin", binary=True)"""
        else:
            raise ValueError('Unknown pretrain model type: %s!' % model_type)
        self.pretrain_model = pretrain_model
        return self.pretrain_model

    @staticmethod
    def train_w2v(sentence_matrix,vocabulary_inv,embedding_dim=config["prep_vecs"]["input_size"],
                  min_word_count=config["prep_vecs"]["min_count"],context=config["prep_vecs"]["window"]):
        """
        Trains, saves, loads Word2Vec model
        Returns initial weights for embedding layer.

        inputs:
        sentence_matrix # int matrix: num_txts x max_sentence_len
        vocabulary_inv  # dict {str:int}
        embedding_dim    # Word vector dimensionality
        min_word_count  # Minimum word count
        context         # Context window size
        """
        model_dir = 'word2vec_models'
        model_name = "{:d}features_{:d}minwords_{:d}context".format(embedding_dim,min_word_count,context)
        model_name = join(model_dir,model_name)
        if exists(model_name):
            pretrain_model = word2vec.Word2Vec.load(model_name)
            logger.debug('Loading existing Word2Vec model \'%s\'' % split(model_name)[-1])
        else:
            ## Set values for various parameters
            num_workers = 2  ## Number of threads to run in parallel
            downsampling = 1e-3  ## Downsample setting for frequent words

            ## Initialize and train the model
            logger.info("Training Word2Vec model...")
            txts = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
            pretrain_model = word2vec.Word2Vec(txts,workers=num_workers,
                                               size=embedding_dim,
                                               min_count=min_word_count,
                                               window=context,
                                               sample=downsampling)

            ## If we don't plan to train the model any further, calling init_sims will make the model much more
            ## memory-efficient.
            pretrain_model.init_sims(replace=True)

            ## Saving the model for later use. You can load it later using Word2Vec.load()
            if not exists(model_dir):
                mkdir(model_dir)
            logger.debug('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
            pretrain_model.save(model_name)

        ## add unknown words
        embedding_weights = [np.array([pretrain_model[w] if w in pretrain_model else
                                       np.random.uniform(-0.25,0.25,pretrain_model.vector_size)
                                       for w in vocabulary_inv])]
        return embedding_weights

    def get_embedding_matrix(self,vocabulary_inv: dict):
        """
        Generates the embedding matrix.
        :param vocabulary_inv:
        :param embedding_model:
        :return:
        """
        embedding_weights = [self.pretrain_model[w] if w in self.pretrain_model
                             else np.random.uniform(-0.25,0.25,self.embedding_dim)
                             for w in vocabulary_inv]
        embedding_weights = np.array(embedding_weights).astype('float32')

        return embedding_weights


if __name__ == '__main__':
    logger.info("Loading pretrained...")
    cls = Text_Encoder()
    sentence_obama = 'Obama speaks to the media in Illinois'
    sentence_president = 'The president greets the press in Chicago'

    docs = [sentence_obama,sentence_president]
    doc2vec_model = cls.load_doc2vec_model(docs,vector_size=10,window=2,negative=2,save_model=False)
    vectors = cls.get_doc2vectors(docs,doc2vec_model)
    logger.debug(vectors)
