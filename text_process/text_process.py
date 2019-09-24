# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Cleans the input texts along with text labels.

__description__ :
__project__     : XCGCN
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "06-05-2019"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : Text_Process
"""

import re,spacy
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS
import unicodedata
from unidecode import unidecode
from os.path import join,isfile,exists
from collections import OrderedDict,Counter

from logger import logger
from text_process.text_encoder import Text_Encoder
from file_utils import File_Util
from config import configuration as config
from config import platform as plat
from config import username as user

# spacy_en = spacy.load("en")
spacy_en = spacy.load("en_core_web_sm")

headings = ['## [edit] ','### [edit] ','<IMG>',"[citation needed]",]
wiki_patterns = ("It has been suggested that Incremental reading be merged into this article or section. (Discuss)",
                 "This section may require cleanup to meet Wikipedia's quality standards.",
                 'This article includes a list of references or external links, but its sources remain unclear because it lacks inline citations. Please improve this article by introducing more precise citations where appropriate.')


class Text_Process(object):
    """ Class related to cleaning the input texts along with text labels. """
    oov_words_dict = OrderedDict()

    def __init__(self,dataset_name: str = config["data"]["dataset_name"],
                 dataset_dir: str = config["paths"]["dataset_dir"][plat][user]):
        """ Initializes the parts of cleaning to be done. """
        super(Text_Process,self).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.text_encoder = Text_Encoder()
        self.vectorizer_model = None
        self.txts2vec_map = None
        self.cats2vec_map = None

    def gen_cats2vec_map(self,cats: dict,vectorizer_model=None):
        """
        Generates a dict of sample text to it's vector map.

        :param vectorizer_model: Doc2Vec model object.
        :param cats:
        :return:
        """
        if self.cats2vec_map is not None:
            return self.cats2vec_map
        else:
            if cats is None: cats = File_Util.load_json(filename=self.dataset_name + "_cats",
                                                        filepath=join(self.dataset_dir,self.dataset_name))
            if vectorizer_model is None:  ## If model is not supplied, load model.
                if self.vectorizer_model is None:
                    self.vectorizer_model = self.text_encoder.load_word2vec()
                vectorizer_model = self.vectorizer_model
            cats2vec_dict = OrderedDict()
            for sample_id,cat in cats.items():
                tokens = self.tokenizer_spacy(cat)
                tokens_vec = self.get_vecs_from_tokens(tokens,vectorizer_model)
                cats2vec_dict[sample_id] = tokens_vec  ## Generate vector for a new sample.
                # cats2vec_dict[sample_id] = vectorizer_model.infer_vector(self.tokenizer_spacy(cat))  ## Generate vector for a new sample using Doc2Vec model only.

        self.cats2vec_map = cats2vec_dict
        return self.cats2vec_map

    def gen_sample2vec_map(self,txts: dict,vectorizer_model=None):
        """
        Generates a dict of sample text to it's vector map.

        :param vectorizer_model: Doc2Vec model object.
        :param txts:
        :return:
        """
        if self.txts2vec_map is not None:
            return self.txts2vec_map
        else:
            if txts is None: txts = File_Util.load_json(filename=self.dataset_name + "_txts",
                                                        filepath=join(self.dataset_dir,self.dataset_name))
            if vectorizer_model is None:  ## If model is not supplied, load model.
                if self.vectorizer_model is None:
                    self.vectorizer_model = self.text_encoder.load_word2vec()
                vectorizer_model = self.vectorizer_model
            txts2vec_dict = OrderedDict()
            for sample_id,txt in txts.items():
                tokens = self.tokenizer_spacy(txt)
                tokens_vec = self.get_vecs_from_tokens(tokens,vectorizer_model)
                txts2vec_dict[sample_id] = tokens_vec  ## Generate vector for a new sample.
                # txts2vec_dict[sample_id] = vectorizer_model.infer_vector(self.tokenizer_spacy(txt))  ## Generate vector for a new sample using Doc2Vec model only.

        self.txts2vec_map = txts2vec_dict
        return self.txts2vec_map

    def get_vecs_from_tokens(self,tokens: list,vectorizer_model,input_size=config["prep_vecs"]["input_size"],avg=True,use_idf=False):
        """ Generates a vector of [input_size] using the [words] and [vectorizer_model].

        :param use_idf:
        :param avg: If average to be calculated.
        :param tokens: List of words
        :param vectorizer_model: Word2Vec model by Gensim
        :param input_size: Dimension of each vector
        :return: Vector of sum of all the words.
        """
        # oov_words_dict = OrderedDict()  ## To hold out-of-vocab tokens.
        sum_vec = None
        # logger.info("Setting default idf = [{}].".format(idf))
        for i,word in enumerate(tokens):
            ## Multiply idf of that word with the vector
            idf = 1
            if use_idf:
                try:  ## If word exists in idf dict
                    idf = self.idf_dict[word]
                except KeyError as e:  ## If word does not exists in idf_dict, multiply 1
                    logger.info("Token [{}] not found in idf_dict.".format(word))
            if word in vectorizer_model.vocab:  ## If word is present in model
                if sum_vec is None:
                    sum_vec = vectorizer_model[word] * idf
                else:
                    sum_vec = np.add(sum_vec,vectorizer_model[word] * idf)
            elif word in self.oov_words_dict:  ## If word is OOV
                if sum_vec is None:
                    sum_vec = self.oov_words_dict[word] * idf
                else:
                    sum_vec = np.add(sum_vec,self.oov_words_dict[word] * idf)
            else:  ## New unknown word, need to create random vector.
                new_oov_vec = np.random.uniform(-0.5,0.5,input_size)
                # vectorizer_model.add(word, new_oov_vec)  ## For some reason, gensim word2vec.add() not working.
                self.oov_words_dict[word] = new_oov_vec
                if sum_vec is None:
                    sum_vec = self.oov_words_dict[word] * idf
                else:
                    sum_vec = np.add(sum_vec,self.oov_words_dict[word] * idf)
        if avg:
            sum_vec = np.divide(sum_vec,float(len(tokens)))

        return np.stack(sum_vec)

    def partition_large_txt(self,doc: str,sents_chunk_mode: str = config["text_process"]["sents_chunk_mode"],
                            num_chunks: int = config["prep_vecs"]["num_chunks"]) -> list:
        """
        Divides a document into chunks based on the vectorizer.

        :param num_chunks:
        :param doc:
        :param sents_chunk_mode:
        :param doc_len:
        :return:
        """
        chunks = []
        if sents_chunk_mode == "concat":
            words = self.tokenizer_spacy(doc)
            for word in words:
                chunks.append(word)
        elif sents_chunk_mode == "word_avg":
            chunks = self.tokenizer_spacy(doc)
        elif sents_chunk_mode == "txts":
            chunks = self.sents_split(doc)
        elif sents_chunk_mode == "chunked":
            splitted_doc = self.tokenizer_spacy(doc)
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
                            "Available options: ['concat','word_avg (Default)','txts','chunked']"
                            .format(sents_chunk_mode))
        chunks = list(filter(None,chunks))  ## Removes empty items, like: ""
        return chunks

    @staticmethod
    def build_vocab(documents):
        """ Builds a list of all words present in list of lists [documents].

        :param documents:
        :return:
        """
        vocab = []
        for d in documents:
            vocab += d

        return vocab

    def remove_min_df(self,documents: list,min_df=5):
        """ Removes words from documents with count <= [min_df].

        :param documents:
        :param min_df: Minimum document frequency.
        """
        vocab = self.build_vocab(documents)
        freq_counts = Counter(vocab)

        documents_df = []
        for i,d in enumerate(documents):
            d2 = []
            for token in d:
                if freq_counts[token] >= min_df:
                    d2.append(token)
            if len(d2) < 1: logger.warning("Empty document no [{}] after removing min_df = [{}].".format(i,min_df))
            documents_df.append([" ".join(d2)])

        return documents_df

    def process_cats(self,labels: dict,remove_stopwords=True):
        """ Process cats like cleaning and tokenization.

        :param remove_stopwords:
        :param labels:
        :return:
        """
        labels_processed,_ = self.clean_txts(labels,specials="""?!_-@<>#,.*?'{}()[]()$%^~`:;"/\\:|""")
        labels_tokens = OrderedDict()
        for lbl_id,lbl_txt in labels_processed.items():
            tokens = self.tokenizer_spacy(lbl_txt)
            if remove_stopwords:
                tokens = [token for token in tokens if token not in STOP_WORDS]
            labels_tokens[lbl_id] = tokens

        return labels_tokens

    def gen_lbl2vec(self,cats_tokenized_dict: dict):
        """ Generate label vectors using pretrained [model].

        :param model:
        :param cats_tokenized_dict:
        :return:
        """
        cat_vecs = []
        cat2vecs_dict = OrderedDict()
        oov_tokens = OrderedDict()
        self.txt_encoder_model = self.text_encoder.load_word2vec()
        for cat_id,cat in cats_tokenized_dict.items():
            cat_vec = np.zeros_like((1,300))
            for token in cat:
                try:
                    cat_vec = self.txt_encoder_model.get_vector(token)
                except KeyError:
                    try:
                        cat_vec = oov_tokens[token]
                    except KeyError:
                        # logger.debug("Vector for token [{}] not found.".format(token))
                        # logger.debug("Error: [{}]".format(e))
                        cat_vec = np.random.uniform(-0.5,0.5,300)
                        oov_tokens[token] = cat_vec
                np.add(cat_vec,cat_vec)
            vec_avg = np.divide(cat_vec,len(cat))
            cat_vecs.append(vec_avg)
            cat2vecs_dict[cat_id] = vec_avg

        return np.stack(cat_vecs), cat2vecs_dict

    @staticmethod
    def dedup_data(Y: dict,dup_cat_map: dict):
        """
        Replaces category ids in Y if it's duplicate.

        :param Y:
        :param dup_cat_map: {old_cat_id : new_cat_id}
        """
        for k,v in Y.items():
            commons = set(v).intersection(set(dup_cat_map.keys()))
            if len(commons) > 0:
                for dup_key in commons:
                    dup_idx = v.index(dup_key)
                    v[dup_idx] = dup_cat_map[dup_key]
        return Y

    @staticmethod
    def split_docs(docs: dict,criteria=' '):
        """
        Splits a dict of idx:documents based on [criteria].

        :param docs: idx:documents
        :param criteria:
        :return:
        """
        splited_docs = OrderedDict()
        for idx,doc in docs:
            splited_docs[idx] = doc.split(criteria)
        return splited_docs

    @staticmethod
    def make_trans_table(specials="""< >  * ? " / \\ : |""",replace=' '):
        """
        Makes a transition table to replace [specials] chars within [text] with [replace].

        :param specials:
        :param replace:
        :return:
        """
        trans_dict = {chars:replace for chars in specials}
        trans_table = str.maketrans(trans_dict)
        return trans_table

    def clean_cats(self,cats: dict,replace=' ',
                         specials=""" ? ! _ - @ < > # , . * ? ' { } [ ] ( ) $ % ^ ~ ` : ; " / \\ : |"""):
        """ Cleans cats dict by removing any symbols and lower-casing and returns set of cleaned cats
        and the dict of duplicate cats.

        :param: cats: dict of cat:id
        :param: specials: list of characters to clean.
        :returns:
            category_cleaned_dict : contains cats which are unique after cleaning.
            dup_cat_map : Dict of new category id mapped to old category id. {old_cat_id : new_cat_id}
        """
        category_cleaned_dict = OrderedDict()
        dup_cat_map = OrderedDict()
        dup_cat_text_map = OrderedDict()
        trans_table = self.make_trans_table(specials=specials,replace=replace)
        for cat,cat_id in cats.items():
            cat_clean = unidecode(str(cat)).translate(trans_table).lower().strip()
            if cat_clean in category_cleaned_dict.keys():
                dup_cat_map[cats[cat]] = category_cleaned_dict[cat_clean]
                dup_cat_text_map[cat] = cat_clean
            else:
                category_cleaned_dict[cat_clean] = cat_id
        return category_cleaned_dict,dup_cat_map,dup_cat_text_map

    def clean_txts(self,txts: dict,specials="""_-@*#'"/\\""",replace=''):
        """Cleans txts dict and returns dict of cleaned txts.

        :param: txts: dict of idx:label
        :returns:
            sents_cleaned_dict : contains cleaned txts.
        """
        sents_cleaned_dict = OrderedDict()
        sents_cleaned_list = []
        trans_table = self.make_trans_table(specials=specials,replace=replace)
        for idx,text in txts.items():
            sents_cleaned_list.append(unidecode(str(text)).translate(trans_table))
            sents_cleaned_dict[idx] = unidecode(str(text)).translate(trans_table)
        return sents_cleaned_dict, sents_cleaned_list

    @staticmethod
    def remove_wiki_first_lines(doc: str,num_lines=6):
        """
        Removes first [num_lines] lines from wikipedia texts.

        :param doc:
        :param num_lines:
        """
        doc = doc.split("\n")[num_lines:]
        doc = list(filter(None,doc))
        return doc

    @staticmethod
    def tokenizer_spacy(input_text: str,remove_stopwords=True):
        """ Document tokenizer using spacy.

        :param remove_stopwords: If stopwords should be removed.
        :param input_text:
        :return:
        """
        input_text = spacy_en(input_text)
        tokens = []
        for token in input_text:
            if remove_stopwords and token.text in STOP_WORDS:
                continue
            tokens.append(token.text)
        return tokens

    def calculate_idf(self,docs: list,subtract: int = 1) -> dict:
        """ Calculates idf scores for each token in the corpus.

        :param docs:
        :param subtract: Removes this value from idf scores. Sometimes needed to get better scores.
        :return: Dict of token to idf score.
        """
        if isfile(join(self.dataset_dir,self.dataset_name + "_idf_dict.json")):
            idf_dict = File_Util.load_json(filename=self.dataset_name + "_idf_dict",filepath=self.dataset_dir)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            ## Using TfidfVectorizer with spacy tokenizer; same tokenizer should be used everywhere.
            vectorizer = TfidfVectorizer(decode_error='ignore',lowercase=False,smooth_idf=False,
                                         tokenizer=self.tokenizer_spacy)
            tfidf_matrix = vectorizer.fit_transform(docs)
            idf = vectorizer.idf_
            idf_dict = dict(zip(vectorizer.get_feature_names(),idf - subtract))  ## Subtract 1 from idf to get better scores

            File_Util.save_json(idf_dict,filename=self.dataset_name + "_idf_dict",filepath=self.dataset_dir)

        return idf_dict

    @staticmethod
    def remove_wiki_cats(doc: list):
        """ Removes category (and some irrelevant and repetitive) information from wiki pages.

        :param doc:
        """
        for i,sent in enumerate(reversed(doc)):  ## Looping from the end of list till first match
            if sent.lower().startswith("Categories:".lower()) or sent.lower().startswith("Category:".lower()):
                del doc[-1]  ## Match found; remove from end before breaking
                del doc[-1]
                del doc[-1]
                break
            del doc[-1]  ## Need to remove from end
        return doc

    @staticmethod
    def remove_wiki_contents(doc: list,start: str = '## Contents',end: str = '## [edit]') -> list:
        """ Removes the contents section from wikipedia pages as it overlaps among documents.

        :param start:
        :param end:
        :param doc:
        """
        start_idx = -1
        end_idx = -1
        content_flag = False
        for i,sent in enumerate(doc):
            if sent.startswith(start):  ## Starting index of content
                content_flag = True
                start_idx = i
                continue
            if sent.startswith(end) and content_flag:  ## Ending index of content
                end_idx = i
                break
        del doc[start_idx:end_idx]
        return doc

    @staticmethod
    def remove_wiki_headings(doc: list) -> list:
        """ Removes wikipedia headings like '### [edit] ', '## [edit] '

        :param doc:
        """
        for i,sent in enumerate(doc):  ## Looping from the end of list till first match
            for heading in headings:
                if sent.startswith(heading):
                    del doc[i]  ## Match found; remove end before breaking
        return doc

    def clean_doc(self,doc: list,symbols=('_','-','=','@','<','>','*','{','}','[',']','(',')','$','%','^','~','`',':',
                                          "\"","\'",'\\','/','|','#','##','###','####','#####'),replace='',
                  pattern=re.compile(r"\[\d\]")) -> list:
        """ Replaces all [symbols] in a list of str with [replace].

        :param doc:
        :param pattern:
        :param replace:
        :param symbols:
        :return:
        """
        doc_cleaned = []
        for i,sent in enumerate(doc):
            for heading in headings:
                if sent.startswith(heading) or not sent:
                    sent = ''
                    break

            if sent:
                sent = sent.strip()
                sent = re.sub(pattern=pattern,repl=replace,string=sent)
                sent = re.sub(r'[^\x00-\x7F]+',' ',sent)

                sent = sent.strip()
                tokens = self.tokenizer_spacy(sent)
                tokens,numbers = self.find_numbers(tokens)

                sent_new = []
                for token in tokens:
                    if token not in symbols:
                        sent_new.append(token)  ## Only add words which are not present in [symbols].
                doc_cleaned.append(" ".join(sent_new))
        return doc_cleaned

    @staticmethod
    def read_stopwords(filepath: str = '',file_name: str = 'stopwords_en.txt',encoding: str = "iso-8859-1") -> list:
        """ Reads the stopwords list from file, useful for customized stopwords.

        :param filepath:
        :param file_name:
        :param encoding:
        """
        so_list = []
        if isfile(join(filepath,file_name)):
            with open(join(filepath,file_name),encoding=encoding) as so_ptr:
                for s_word in so_ptr:
                    so_list.append(s_word.strip())
        else:
            raise Exception("File not found at: [{}]".format(join(filepath,file_name)))

        return so_list

    @staticmethod
    def remove_symbols(doc: list,symbols=(
            '_','-','=','@','<','>','*','{','}','[',']','(',')','$','%','^','~','`',':',"\"","\'",'\\','#','##','###',
            '####','#####')):
        """ Replaces [symbols] from [doc] with [replace].

        :param doc:
        :param symbols:
        :return:
        """
        txt_new = []
        for i,sent in enumerate(doc):
            sent_new = []
            sent = sent.strip()  ## Stripping extra spaces at front and back.
            for token in sent.split():
                if token not in symbols:
                    sent_new.append(token)  ## Add words which are not present in [symbols] or [stopwords].
            txt_new.append(" ".join(sent_new))
        return txt_new

    @staticmethod
    def remove_symbols_trans(doc: str,symbols="""* , _ [ ] > < { } ( )""",replace=' '):
        """ Replaces [symbols] from [doc] with [replace].

        :param doc:
        :param symbols:
        :param replace:
        :return:
        """
        doc = unidecode(str(doc))
        trans_dict = {chars:replace for chars in symbols}
        trans_table = str.maketrans(trans_dict)
        return doc.translate(trans_table)

    @staticmethod
    def remove_nonascii(doc: list) -> list:
        """ Removes all non-ascii characters.

        :param doc:
        :return:
        """
        txt_ascii = []
        for sent in doc:
            sent = re.sub(r'[^\x00-\x7F]+',' ',sent)
            ## Alternatives::
            ## sent2 = ''.join([i if ord(i) < 128 else ' ' for i in sent])
            ## sent3 = unidecode(str(sent))
            ## sent4 = sent.encode('ascii', 'replace').decode()
            ## sent5 = sent.encode('ascii', 'ignore').decode()
            txt_ascii.append(sent)

        return txt_ascii

    def doc_unicode2ascii(self,doc: list) -> list:
        """ Converts a list of non-ascii str to ascii str based on unicode complient.

        :param doc:
        :return:
        """
        txt_ascii = []
        for sent in doc:
            sent2 = self.unicode2ascii(sent)
            txt_ascii.append(sent2)

        return txt_ascii

    @staticmethod
    def unicode2ascii(sent: str):
        """ Turn a Unicode string to plain ASCII. Thanks to http://stackoverflow.com/a/518232/2809427

        :param sent:
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD',sent)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def remove_patterns(doc: list,pattern=None,replace=''):
        """ Remove references from wiki pages.

        :param replace:
        :param pattern:
        :param doc:
        """
        if pattern is None:
            pattern = re.compile(r"\[\d\]")
        doc2 = []
        for sent in doc:
            sent = re.sub(pattern=pattern,repl=replace,string=sent)
            doc2.append(sent)
        return doc2

    @staticmethod
    def sents_split(doc: str):
        """ Splits the document into txts and remove stopwords.

        :param doc:
        """
        doc_spacy = spacy_en(doc)
        txts = list(doc_spacy.sents)
        return txts

    @staticmethod
    def spacy_sents2string(doc: list):
        """ Converts a list of spacy span to concatenated string.

        We usually get this type from text_process.py file.
        :param doc:
        """
        sents = str()
        for sent in doc:
            sents = sents + ' ' + sent.text
        return sents

    @staticmethod
    def find_label_occurrences(doc,label: str):
        """ Finds all label indices within document.

        :param doc:
        :param label:
        """
        label_idx = []
        logger.debug(label)
        ## re can not work with patterns having '+' or '*' in it, ignoring str with these characters.
        if '+' in label: return False
        if '*' in label: return False

        for m in re.finditer(label,doc):
            label_idx.append((m.start(),m.end()))
        if label_idx:
            return label_idx
        return False

    @staticmethod
    def find_label_occurrences_2(doc,labels: list):
        """ Finds all label indices within document.

        :param doc:
        :param labels:
        """
        labels_indices = OrderedDict()
        for lbl in labels:
            for m in re.finditer(lbl,doc):
                if lbl not in labels_indices:
                    labels_indices[lbl] = []
                labels_indices[lbl].append((m.start(),m.end()))
        return labels_indices

    @staticmethod
    def filter_html_categories_reverse(txt: list):
        """Filters categories from html text."""
        category_lines_list = []
        category_lines = ""
        copy_flag = False
        del_line = True
        remove_first_chars = 12  ## Length of "Categories:", to be removed from line.

        ## Categories are written in multiple lines, need to read all lines (till "##### Views").
        for i,line in enumerate(reversed(txt)):  ## Looping reversed
            if line.lower().startswith("##### Views".lower()):
                copy_flag = True  ## Start coping "Categories:" as "##### Views" started.
                del_line = False
            if line.lower().startswith("Retrieved from".lower()) or line.lower().startswith(
                    "\"http://en.wikipedia.org".lower()):
                copy_flag = False
                del txt[-1]
            if copy_flag:
                category_lines = line + " " + category_lines
                category_lines_list.insert(0,line)
                del txt[-1]
            if line.lower().startswith("Categories:".lower()):
                break
            elif line.lower().startswith("Category:".lower()):
                remove_first_chars = 11
                break
            if del_line:
                del txt[-1]

        category_lines = category_lines[:-12]  ## To remove "##### Views "
        hid_cats = None
        if "Hidden categories:".lower() in category_lines.lower():  ## Process hidden categories
            category_lines,hid_cats = category_lines.split("Hidden categories:")
            hid_cats = (hid_cats.split(" | "))  ## Do not add hidden categories to categories.
            hid_cats = [cat.strip() for cat in hid_cats]
            hid_cats = list(filter(None,hid_cats))  ## Removing empty items.

        ## Filtering Categories
        category_lines = category_lines[remove_first_chars:].split(" | ")

        category_lines = [cat.strip() for cat in category_lines]
        category_lines = list(filter(None,category_lines))  ## Removing empty items.
        return txt,category_lines,hid_cats

    @staticmethod
    def remove_url(doc):
        """ Removes URls from str.

        :param doc:
        :return:
        """
        return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*','',doc,flags=re.MULTILINE)  ## Removes URLs

    @staticmethod
    def case_folding(sent: list,all_caps: bool = False) -> list:
        """ Converts all text to lowercase except all capital letter words.

        :param sent:
        :param all_caps:
        :return:
        """
        for pos in range(len(sent)):
            if sent[pos].isupper():
                continue
            else:
                sent[pos] = sent[pos].lower()
        return sent

    def format_digits(self,doc: list):
        """ Replaces number from a str.

        :return: text, list of numbers
        Ex:
        '1230,1485': 8d
        '-2': 1d
        3.0 : 2f
        """
        doc2 = []
        for sent in doc:
            sent,numbers = self.find_numbers(sent)
            doc2.append(sent)

        return doc2

    @staticmethod
    def find_numbers(text: list,ignore_len: int = 4,replace=True):
        """ Finds and replaces numbers in list of str.

        :param ignore_len: Ignore numbers less than [ignore_len] digits.
        :param text: strings that contains digit and words
        :param replace: bool to decide if numbers need to be replaced.
        :return: text, list of numbers

        Ex:
        '1230,1485': 8d
        '-2': 1d
        3.0 : 2f
        """
        import re

        numexp = re.compile(r'(?:(?:\d+,?)+(?:\.?\d+)?)')
        numbers = numexp.findall(" ".join(text))
        numbers = [num for num in numbers if len(str(num)) > ignore_len]  ## Ignore numbers less than 4 digits.
        if replace:
            for num in numbers:
                try:
                    i = text.index(num)
                    if num.isdigit():
                        text[i] = str(len(num)) + "d"
                    else:
                        try:  ## Check if float
                            num = float(num)
                            text[i] = str(len(str(num)) - 1) + "f"
                        except ValueError as e:  ## if not float, process as int.
                            text[i] = str(len(num) - 1) + "d"
                except ValueError as e:  ## No numbers within text
                    pass
        return text,numbers

    @staticmethod
    def stemming(token,lemmatize=False):
        """ Stems tokens.

        :param token:
        :param lemmatize:
        :return:
        """
        if lemmatize:
            from nltk.stem import WordNetLemmatizer

            wnl = WordNetLemmatizer()
            return wnl.lemmatize(token)
        else:
            from nltk.stem import PorterStemmer

            ps = PorterStemmer()
            return ps.stem(token)

    @staticmethod
    def tokenizer_re(sent: str,lowercase=False,remove_emoticons=True):
        """ Tokenize a string.

        :param sent:
        :param lowercase:
        :param remove_emoticons:
        :return:
        """
        import re

        emoticons_str = r'''
            (?:
                [:=;] # Eyes
                [oO\-]? # Nose (optional)
                [D\)\]\(\]/\\OpP] # Mouth
            )'''
        regex_str = [
            emoticons_str,
            r'<[^>]+>',  # HTML tags
            r'(?:@[\w_]+)',  # @-mentions
            r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f]['
            r'0-9a-f]))+',  # URLs
            r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
            r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
            r'(?:[\w_]+)',  # other words
            r'(?:\S)'  # anything else
        ]
        tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')',
                               re.VERBOSE | re.IGNORECASE)
        emoticon_re = re.compile(r'^' + emoticons_str + '$',
                                 re.VERBOSE | re.IGNORECASE)

        ## TODO: remove emoticons only (param: remove_emoticons).
        tokens = tokens_re.findall(str(sent))
        if lowercase:
            tokens = [token if emoticon_re.search(token) else token.lower() for
                      token in tokens]
        return tokens

    @staticmethod
    def remove_symbols(tweet,stopword=False,punct=False,specials=False):
        """ Removes symbols.

        :param tweet:
        :param stopword:
        :param punct:
        :param specials:
        :return:
        """
        if stopword:
            from nltk.corpus import stopwords

            stopword_list = stopwords.words('english') + ['rt','via','& amp',
                                                          '&amp','amp','mr']
            tweet = [term for term in tweet if term not in STOP_WORDS]

        if punct:
            from string import punctuation

            tweet = [term for term in tweet if term not in list(punctuation)]

        if specials:
            trans_dict = {chars:' ' for chars in specials}
            trans_table = str.maketrans(trans_dict)
            tweet = tweet.translate(trans_table)

            for pos in range(len(tweet)):
                tweet[pos] = tweet[pos].replace("@","")
                tweet[pos] = tweet[pos].replace("#","")
                tweet[pos] = tweet[pos].replace("-"," ")
                tweet[pos] = tweet[pos].replace("&"," and ")
                tweet[pos] = tweet[pos].replace("$"," dollar ")
                tweet[pos] = tweet[pos].replace("  "," ")
        return tweet


def clean_wiki2(doc: list,num_lines: int = 6) -> str:
    """ Cleans the wikipedia documents. """
    cls = Text_Process()
    doc = doc[num_lines:]
    doc = list(filter(None,doc))
    doc = cls.remove_wiki_contents(doc)
    doc = cls.remove_wiki_headings(doc)
    doc = cls.remove_nonascii(doc)  ## Remove all non-ascii characters
    doc = cls.remove_patterns(doc)
    # doc = cls.remove_patterns(doc, pattern=re.compile(r"\""))  ## Removes " with txts
    doc = cls.remove_symbols(doc)
    # doc = cls.format_digits(doc)
    doc = cls.sents_split(" ".join(doc))
    doc = cls.spacy_sents2string(doc)
    doc = cls.remove_url(doc)

    return doc


def clean_wiki(doc: list,num_lines: int = 6) -> str:
    """ Cleans a wikipedia document. """
    cls = Text_Process()
    doc = doc[num_lines:]
    doc = list(filter(None,doc))
    doc = cls.remove_wiki_contents(doc)
    doc = cls.clean_doc(doc)
    doc = cls.sents_split(" ".join(doc))
    doc = cls.spacy_sents2string(doc)
    doc = cls.remove_url(doc)

    return doc


def clean_wiki_pages(docs):
    """ Cleans wikipedia documents in list of lists format.

    :param docs:
    :return:
    """
    docs_cleaned = []
    for doc in docs:
        docs_cleaned.append(clean_wiki(doc))
    return docs_cleaned


def main() -> None:
    """ main module to start code """
    doc = """
# Encryption

### From Wikipedia, the free encyclopedia

Jump to: navigation, search

"Encrypt" redirects here. For the film, see Encrypt (film).

This article is about algorithms for encryption and decryption. For an
overview of cryptographic technology in general, see Cryptography.

In cryptography, encryption is the process of transforming information
(referred to as plaintext) using an algorithm (called cipher) to make it
unreadable to anyone except those possessing special knowledge, usually
referred to as a key. The result of the process is encrypted information (in
cryptography, referred to as ciphertext). In many contexts, the word
encryption also implicitly refers to the reverse process, decryption (e.g.
âsoftware for encryptionâ can typically also perform decryption), to make
the encrypted information readable again (i.e. to make it unencrypted).

Encryption has long been used by militaries and governments to facilitate
secret communication. Encryption is now commonly used in protecting
information within many kinds of civilian systems. For example, in 2007 the
U.S. government reported that 71% of companies surveyed utilized encryption
for some of their data in transit.[1] Encryption can be used to protect data
"at rest", such as files on computers and storage devices (e.g. USB flash
drives). In recent years there have been numerous reports of confidential data
such as customers' personal records being exposed through loss or theft of
laptops or backup drives. Encrypting such files at rest helps protect them
should physical security measures fail. Digital rights management systems
which prevent unauthorized use or reproduction of copyrighted material and
protect software against reverse engineering (see also copy protection)are
another somewhat different example of using encryption on data at rest.

Encryption is also used to protect data in transit, for example data being
transferred via networks (e.g. the Internet, e-commerce), mobile telephones,
wireless microphones, wireless intercom systems, Bluetooth devices and bank
automatic teller machines. There have also been numerous reports of data in
transit being intercepted in recent years [2]. Encrypting data in transit also
helps to secure it as it is often difficult to physically secure all access to
networks. Encryption, by itself, can protect the confidentiality of messages,
but other techniques are still needed to protect the integrity and
authenticity of a message; for example, verification of a message
authentication code (MAC) or a digital signature. Standards and cryptographic
software and hardware to perform encryption are widely available, but
successfully using encryption to ensure security may be a challenging problem.
A single slip-up in system design or execution can allow successful attacks.
Sometimes an adversary can obtain unencrypted information without directly
undoing the encryption. See, e.g., traffic analysis, TEMPEST, or Trojan horse.

One of the earliest public key encryption applications was called Pretty Good
Privacy (PGP), according to Paul Rubens. It was written in 1991 by Phil
Zimmermann and was bought by Network Associates in 1997 and is now called PGP
Corporation.

There are a number of reasons why an encryption product may not be suitable in
all cases. First e-mail must be digitally signed at the point it was created
to provide non-repudiation for some legal purposes, otherwise the sender could
argue that it was tampered with after it left their computer but before it was
encrypted at a gateway according to Paul. An encryption product may also not
be practical when mobile users need to send e-mail from outside the corporate
network.* [3]

## [edit] See also

  * Cryptography
  * Cold boot attack
  * Encryption software

  * Cipher
  * Key
  * Famous ciphertexts

<IMG> Cryptography portal  
  * Disk encryption
  * Secure USB drive
  * Secure Network Communications

  
## [edit] References

  1. ^ 2008 CSI Computer Crime and Security Survey, by Robert Richardson, p19
  2. ^ Fiber Optic Networks Vulnerable to Attack, Information Security Magazine, November 15, 2006, Sandra Kay Miller
  3. ^ [1]

  * Helen FouchÃ© Gaines, âCryptanalysisâ, 1939, Dover. ISBN 0-486-20097-3
  * David Kahn, The Codebreakers - The Story of Secret Writing (ISBN 0-684-83130-9) (1967)
  * Abraham Sinkov, Elementary Cryptanalysis: A Mathematical Approach, Mathematical Association of America, 1966. ISBN 0-88385-622-0

## [edit] External links

  * [2]

<IMG>

Look up encryption in Wiktionary, the free dictionary.

  * SecurityDocs Resource for encryption whitepapers
  * Accumulative archive of various cryptography mailing lists. Includes Cryptography list at metzdowd and SecurityFocus Crypto list.

v â¢ d â¢ e

Cryptography  
History of cryptography Â· Cryptanalysis Â· Cryptography portal Â· Topics in
cryptography  
Symmetric-key algorithm Â· Block cipher Â· Stream cipher Â· Public-key
cryptography Â· Cryptographic hash function Â· Message authentication code Â·
Random numbers Â· Steganography  
Retrieved from "http://en.wikipedia.org/wiki/Encryption"

Categories: Cryptography

##### Views

  * Article
  * Discussion
  * Edit this page
  * History

##### Personal tools

  * Log in / create account

##### Navigation

  * Main page
  * Contents
  * Featured content
  * Current events
  * Random article

##### Search



##### Interaction

  * About Wikipedia
  * Community portal
  * Recent changes
  * Contact Wikipedia
  * Donate to Wikipedia
  * Help

##### Toolbox

  * What links here
  * Related changes
  * Upload file
  * Special pages
  * Printable version
  * Permanent link
  * Cite this page

##### Languages

  * Ø§ÙØ¹Ø±Ø¨ÙØ©
  * Bosanski
  * Dansk
  * Deutsch
  * Eesti
  * EspaÃ±ol
  * Esperanto
  * FranÃ§ais
  * Bahasa Indonesia
  * Ãslenska
  * Bahasa Melayu
  * Nederlands
  * æ¥æ¬èª
  * Polski
  * Ð ÑÑÑÐºÐ¸Ð¹
  * Simple English
  * Svenska
  * à¹à¸à¸¢
  * Tiáº¿ng Viá»t
  * Ð£ÐºÑÐ°ÑÐ½ÑÑÐºÐ°
  * ä¸­æ

Powered by MediaWiki

Wikimedia Foundation

  * This page was last modified on 5 April 2009, at 23:58.
  * All text is available under the terms of the GNU Free Documentation License. (See Copyrights for details.)   
Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a U.S.
registered 501(c)(3) tax-deductible nonprofit charity.  

  * Privacy policy
  * About Wikipedia
  * Disclaimers



"""
    cls = Text_Process()
    doc = doc.split("\n")
    doc,filtered_categories,filtered_hid_categories = cls.filter_html_categories_reverse(doc)
    doc_spacy = clean_wiki(doc)
    logger.debug(doc_spacy)
    logger.debug(filtered_categories)
    logger.debug(filtered_hid_categories)


if __name__ == "__main__":
    main()
