# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Common_Data_Handler.

__description__ : Class to handle pre-processed json files.
__project__     : XCGCN
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : Common_Data_Handler,
"""

import pandas as pd
from gc import collect
from os import makedirs
from os.path import join,isfile,exists
from collections import OrderedDict

from data_loaders import html_loader as html
from data_loaders import json_loader as json
from data_loaders import txt_loader as txt
from file_utils import File_Util
from logger import logger
from config import configuration as config
from text_process import Text_Process
from config import platform as plat
from config import username as user


class Common_Data_Handler:
    """
    Class to load and prepare and split pre-built json files.

    txts : Wikipedia english texts after parsing and cleaning.
    txts = {"id1": "wiki_text_1", "id2": "wiki_text_2"}

    sample2cats   : OrderedDict of id to sample2cats.
    sample2cats = {"id1": [class_id_1,class_id_2],"id2": [class_id_2,class_id_10]}

    categories : Dict of class texts.
    categories = {"Computer Science":class_id_1, "Machine Learning":class_id_2}
    """

    def __init__(self,
                 dataset_name: str = config["data"]["dataset_name"],
                 dataset_type=config["xc_datasets"][
                     config["data"]["dataset_name"]],
                 dataset_dir: str = config["paths"]["dataset_dir"][plat][user]):
        """
        Loads train val or test data based on run_mode.

        Args:
            dataset_dir : Path to directory of the dataset.
            dataset_name : Name of the dataset.
        """
        super(Common_Data_Handler,self).__init__()
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.dataset_dir = join(dataset_dir,self.dataset_name)

        self.clean = Text_Process()

        self.cattext2catid_map = None
        self.catid2cattxt_map = None
        self.txts_sel,self.sample2cats_sel,self.cats_sel = None,None,None
        self.txts_train,self.sample2cats_train,self.cats_sel = None,None,None
        self.txts_test,self.sample2cats_test,self.cats_test = None,None,None
        self.txts_val,self.sample2cats_val,self.cats_val = None,None,None

    def multilabel2multiclass_df(self,df: pd.DataFrame):
        """Converts Multi-Label data in DataFrame format to Multi-Class data by replicating the samples.

        :param df: Dataframe containing repeated sample id and it's associated category.
        :returns: DataFrame with replicated samples.
        """
        if self.catid2cattxt_map is None:
            self.catid2cattxt_map = File_Util.load_json(filename=self.dataset_name
                                                        + "_catid2cattxt_map",
                                                        filepath=self.dataset_dir)
        idxs,cat = [],[]
        for row in df.values:
            lbls = row[3][1:-1].split(',')  ## When DataFrame is saved as csv, list is converted to str
            for lbl in lbls:
                lbl = lbl.strip()
                idxs.append(row[1])
                cat.append(lbl)

        df = pd.DataFrame.from_dict({"idx"    :idxs,
                                     "cat"    :cat})
        df = df[~df['cat'].isna()]
        df.to_csv(path_or_buf=join(self.dataset_dir,
                                   self.dataset_name + "_multiclass_df.csv"))

        logger.info("Data shape = {} ".format(df.shape))

        return df

    def gen_data_stats(self,txts: dict = None,sample2cats: dict = None,
                       cats: dict = None):
        """ Generates statistics about the data.

        Like:
         freq category id distribution: Category frequency distribution (sorted).
         sample ids with max number of categories:
         Top words: Most common words.
         category specific word dist: Words which are dominant in a particular categories.
         words per sample dist: Distribution of word count in a sample.
         words per category dist: Distribution of words per category.
         most co-occurring categories: Categories which has highest common sample.
         """
        # dict(sorted(words.items(), key=lambda x: x[1]))  # Sorting a dict by value.
        # sorted_d = sorted((value, key) for (key,value) in d.items())  # Sorting a dict by value.
        # dd = OrderedDict(sorted(d.items(), key=lambda x: x[1]))  # Sorting a dict by value.
        if sample2cats is None: txts,sample2cats,cats = self.load_full_json(
            return_values=True)

        cat_freq = OrderedDict()
        for k,v in sample2cats.items():
            for cat in v:
                if cat not in cat_freq:
                    cat_freq[cat] = 1
                else:
                    cat_freq[cat] += 1
        cat_freq_sorted = OrderedDict(sorted(cat_freq.items(),key=lambda x:x[
            1]))  # Sorting a dict by value.
        logger.info("Category Length: {}".format(len(cat_freq_sorted)))
        logger.info("Category frequencies: {}".format(cat_freq_sorted))

    def load_full_json(self,return_values: bool = False):
        """
        Loads full dataset and splits the data into train, val and test.
        """
        if isfile(join(self.dataset_dir,self.dataset_name + "_txts.json"))\
                and isfile(
            join(self.dataset_dir,self.dataset_name + "_sample2cats.json"))\
                and isfile(
            join(self.dataset_dir,self.dataset_name + "_cats.json")):
            logger.info("Loading pre-processed json files from: [{}]".format(
                join(self.dataset_dir,self.dataset_name + "_txts.json")))
            txts = File_Util.load_json(self.dataset_name + "_txts",
                                       filepath=self.dataset_dir,show_path=True)
            classes = File_Util.load_json(self.dataset_name + "_sample2cats",
                                          filepath=self.dataset_dir,
                                          show_path=True)
            categories = File_Util.load_json(self.dataset_name + "_cats",
                                             filepath=self.dataset_dir,
                                             show_path=True)
            assert len(txts) == len(classes),\
                "Count of txts [{0}] and sample2cats [{1}] should match.".format(
                    len(txts),len(classes))
        else:
            logger.warn("Pre-processed json files not found at: [{}]".format(
                join(self.dataset_dir,self.dataset_name + "_txts.json")))
            logger.info(
                "Loading raw data and creating 3 separate dicts of txts [id->texts], sample2cats [id->class_ids]"
                " and categories [class_name : class_id].")
            txts,classes,categories = self.load_raw_data(self.dataset_type)
            File_Util.save_json(categories,self.dataset_name + "_cats",
                                filepath=self.dataset_dir)
            File_Util.save_json(txts,self.dataset_name + "_txts",
                                filepath=self.dataset_dir)
            File_Util.save_json(classes,self.dataset_name + "_sample2cats",
                                filepath=self.dataset_dir)
            logger.info("Cleaning categories.")
            categories,categories_dup_dict,dup_cat_text_map = self.clean.clean_categories(
                categories)
            File_Util.save_json(dup_cat_text_map,
                                self.dataset_name + "_dup_cat_text_map",
                                filepath=self.dataset_dir,
                                overwrite=True)
            File_Util.save_json(categories,self.dataset_name + "_cats",
                                filepath=self.dataset_dir,overwrite=True)
            if categories_dup_dict:  # Replace old category ids with new ids if duplicate categories found.
                File_Util.save_json(categories_dup_dict,
                                    self.dataset_name + "_categories_dup_dict",
                                    filepath=self.dataset_dir,
                                    overwrite=True)  # Storing the duplicate categories for future dedup removal.
                classes = self.clean.dedup_data(classes,categories_dup_dict)
            assert len(txts) == len(classes),\
                "Count of txts [{0}] and sample2cats [{1}] should match.".format(
                    len(txts),len(classes))
            File_Util.save_json(txts,self.dataset_name + "_txts",
                                filepath=self.dataset_dir,overwrite=True)
            File_Util.save_json(classes,self.dataset_name + "_sample2cats",
                                filepath=self.dataset_dir,overwrite=True)
            logger.info(
                "Saved txts [{0}], sample2cats [{1}] and categories [{2}] as json files.".format(
                    join(self.dataset_dir + "_txts.json"),
                    join(self.dataset_dir + "_sample2cats.json"),
                    join(self.dataset_dir + "_cats.json")))
        if return_values:
            return txts,classes,categories
        else:
            # Splitting data into train, validation and test sets.
            self.txts_train,self.sample2cats_train,self.cats_sel,self.txts_val,self.sample2cats_val,\
            self.cats_val,self.txts_test,self.sample2cats_test,self.cats_test,catid2cattxt_map =\
                self.split_data(txts=txts,classes=classes,categories=categories)
            txts,classes,categories = None,None,None  # Remove large dicts and free up memory.
            collect()

            File_Util.save_json(self.txts_test,self.dataset_name + "_txts_test",
                                filepath=self.dataset_dir)
            File_Util.save_json(self.sample2cats_test,
                                self.dataset_name + "_sample2cats_test",
                                filepath=self.dataset_dir)
            File_Util.save_json(self.txts_val,self.dataset_name + "_txts_val",
                                filepath=self.dataset_dir)
            File_Util.save_json(self.sample2cats_val,
                                self.dataset_name + "_sample2cats_val",
                                filepath=self.dataset_dir)
            File_Util.save_json(self.txts_train,
                                self.dataset_name + "_txts_train",
                                filepath=self.dataset_dir)
            File_Util.save_json(self.sample2cats_train,
                                self.dataset_name + "_sample2cats_train",
                                filepath=self.dataset_dir)
            File_Util.save_json(self.cats_sel,self.dataset_name + "_cats_train",
                                filepath=self.dataset_dir)
            File_Util.save_json(self.cats_val,self.dataset_name + "_cats_val",
                                filepath=self.dataset_dir)
            File_Util.save_json(self.cats_test,self.dataset_name + "_cats_test",
                                filepath=self.dataset_dir)
            File_Util.save_json(catid2cattxt_map,
                                self.dataset_name + "_catid2cattxt_map",
                                filepath=self.dataset_dir)
            return self.txts_train,self.sample2cats_train,self.cats_sel,self.txts_val,self.sample2cats_val,\
                   self.cats_val,self.txts_test,self.sample2cats_test,self.cats_test

    def load_raw_data(self,dataset_type: str = None):
        """
        Loads raw data based on type of dataset.

        :param dataset_type: Type of dataset.
        """
        if dataset_type is None: dataset_type = self.dataset_type
        if dataset_type == "html":
            self.dataset = html.HTMLLoader(dataset_name=self.dataset_name,
                                           dataset_dir=self.dataset_dir)
        elif dataset_type == "json":
            self.dataset = json.JSONLoader(dataset_name=self.dataset_name,
                                           dataset_dir=self.dataset_dir)
        elif dataset_type == "txt":
            self.dataset = txt.TXTLoader(dataset_name=self.dataset_name,
                                         dataset_dir=self.dataset_dir)
        else:
            raise Exception("Dataset type for dataset [{}] not found. \n"
                            "Possible reasons: Dataset not added to the config file.".format(
                self.dataset_name))
        txts,classes,categories = self.dataset.get_data()
        return txts,classes,categories

    def split_data(self,txts: OrderedDict,classes: OrderedDict,
                   categories: OrderedDict,
                   test_split: int = config["data"]["test_split"],
                   val_split: int = config["data"]["val_split"]):
        """ Splits input data into train, val and test.

        :return:
        :param categories:
        :param classes:
        :param txts:
        :param val_split: Validation split size.
        :param test_split: Test split size.
        :return:
        """
        logger.info("Total number of samples: [{}]".format(len(classes)))
        sample2cats_train,sample2cats_test,txts_train,txts_test =\
            File_Util.split_dict(classes,txts,
                                 batch_size=int(len(classes) * test_split))
        logger.info("Test count: [{}]. Remaining count: [{}]".format(
            len(sample2cats_test),len(sample2cats_train)))

        sample2cats_train,sample2cats_val,txts_train,txts_val =\
            File_Util.split_dict(sample2cats_train,txts_train,
                                 batch_size=int(len(txts_train) * val_split))
        logger.info("Validation count: [{}]. Train count: [{}]".format(
            len(sample2cats_val),len(sample2cats_train)))

        if isfile(join(self.dataset_dir,
                       self.dataset_name + "_catid2cattxt_map.json")):
            catid2cattxt_map = File_Util.load_json(
                self.dataset_name + "_catid2cattxt_map",
                filepath=self.dataset_dir)
            # Integer keys are converted to string when saving as JSON. Converting back to integer.
            catid2cattxt_map_int = OrderedDict()
            for k,v in catid2cattxt_map.items():
                catid2cattxt_map_int[int(k)] = v
            catid2cattxt_map = catid2cattxt_map_int
        else:
            logger.info("Generating inverted categories.")
            catid2cattxt_map = File_Util.inverse_dict_elm(categories)

        logger.info("Creating train categories.")
        cats_train = OrderedDict()
        for k,v in sample2cats_train.items():
            for cat_id in v:
                if cat_id not in cats_train:
                    cats_train[cat_id] = catid2cattxt_map[cat_id]
        cats_train = cats_train

        logger.info("Creating validation categories.")
        cats_val = OrderedDict()
        for k,v in sample2cats_val.items():
            for cat_id in v:
                if cat_id not in cats_val:
                    cats_val[cat_id] = catid2cattxt_map[cat_id]
        cats_val = cats_val

        logger.info("Creating test categories.")
        cats_test = OrderedDict()
        for k,v in sample2cats_test.items():
            for cat_id in v:
                if cat_id not in cats_test:
                    cats_test[cat_id] = catid2cattxt_map[cat_id]
        cats_test = cats_test
        return txts_train,sample2cats_train,cats_train,txts_val,sample2cats_val,cats_val,txts_test,sample2cats_test,cats_test,catid2cattxt_map

    def gen_cat2samples_map(self,classes_dict: dict = None):
        """ Generates a dictionary of category to samples mapping.i.e. sample : categories -> categories : samples

        :returns: A dictionary of categories to sample mapping.
        """
        cat2samples_map = OrderedDict()
        if classes_dict is None: classes_dict = self.sample2cats_sel
        for sample_id,categories_list in classes_dict.items():
            for cat in categories_list:
                if cat not in cat2samples_map:
                    cat2samples_map[cat] = []
                cat2samples_map[cat].append(sample_id)
        return cat2samples_map

    def find_cats_with_few_samples(self,sample2cats_map: dict = None,
                                   cat2samples_map: dict = None,
                                   remove_count=20):
        """ Finds categories with <= [remove_count] samples. Default few-shot = <=20.

        :returns:
            cat2samples_few: Category to samples map without tail categories.
            tail_cats: List of tail cat ids.
            samples_with_tail_cats: Set of sample ids which belong to tail categories.
        """
        if cat2samples_map is None: cat2samples_map = self.gen_cat2samples_map(
            sample2cats_map)
        tail_cats = []
        samples_with_tail_cats = set()
        cat2samples_few = OrderedDict()
        for cat,sample_list in cat2samples_map.items():
            if len(sample_list) <= remove_count:
                cat2samples_few[cat] = len(sample_list)
            else:
                tail_cats.append(cat)
                samples_with_tail_cats.update(sample_list)

        return tail_cats,samples_with_tail_cats,cat2samples_few

    def get_data(self,load_type: str = "all",calculate_idf=False) -> (
            OrderedDict,OrderedDict,OrderedDict):
        """:returns loaded dictionaries based on "load_type" value. Loads all if not provided."""
        self.cattext2catid_map = self.load_categories()
        if load_type == "train":
            self.txts_sel,self.sample2cats_sel,self.cats_sel = self.load_train()
        elif load_type == "val":
            self.txts_sel,self.sample2cats_sel,self.cats_sel = self.load_val()
        elif load_type == "test":
            self.txts_sel,self.sample2cats_sel,self.cats_sel = self.load_test()
        elif load_type == "all":
            self.txts_sel,self.sample2cats_sel,self.cats_sel = self.load_all()
        else:
            raise Exception("Unsupported load_type: [{}]. \n"
                            "Available options: ['all (Default)','train','val','test']"
                            .format(load_type))
        # self.gen_data_stats(self.txts_sel, self.sample2cats_sel, self.cats_sel)
        df = self.json2csv(self.txts_sel,self.sample2cats_sel)
        if calculate_idf:
            idf_dict = self.clean.calculate_idf_per_token(
                txts=list(self.txts_sel.values()))
            return df,self.cattext2catid_map,idf_dict
        return df,self.cattext2catid_map

    def load_categories(self) -> OrderedDict:
        """Loads and returns the whole categories set."""
        if self.cattext2catid_map is None:
            logger.debug(
                join(self.dataset_dir,
                     self.dataset_name + "_cattext2catid_map.json"))
            if isfile(join(self.dataset_dir,
                           self.dataset_name + "_cattext2catid_map.json")):
                self.cattext2catid_map = File_Util.load_json(
                    self.dataset_name + "_cattext2catid_map",
                    filepath=self.dataset_dir)
            else:
                _,_,self.cattext2catid_map = self.load_full_json(
                    return_values=True)
        return self.cattext2catid_map

    def load_all(self) -> (OrderedDict,OrderedDict,OrderedDict):
        """Loads and returns the whole data."""
        logger.debug(join(self.dataset_dir,self.dataset_name + "_txts.json"))
        if self.txts_sel is None:
            if isfile(join(self.dataset_dir,self.dataset_name + "_txts.json")):
                self.txts_sel = File_Util.load_json(self.dataset_name + "_txts",
                                                    filepath=self.dataset_dir)
            else:
                self.txts_sel,self.sample2cats_sel,self.cats_sel = self.load_full_json(
                    return_values=True)

        if self.sample2cats_sel is None:
            if isfile(join(self.dataset_dir,
                           self.dataset_name + "_sample2cats.json")):
                self.sample2cats_sel = File_Util.load_json(
                    self.dataset_name + "_sample2cats",
                    filepath=self.dataset_dir)
            else:
                self.txts_sel,self.sample2cats_sel,self.cats_sel = self.load_full_json(
                    return_values=True)

        if self.cats_sel is None:
            if isfile(join(self.dataset_dir,self.dataset_name + "_cats.json")):
                self.cats_sel = File_Util.load_json(self.dataset_name + "_cats",
                                                    filepath=self.dataset_dir)
            else:
                self.txts_sel,self.sample2cats_sel,self.cats_sel = self.load_full_json(
                    return_values=True)
        collect()

        logger.info(
            "Total data counts:\n\ttxts = [{}],\n\tsample2cats = [{}],\n\tcattext2catid_map = [{}]"
                .format(len(self.txts_sel),len(self.sample2cats_sel),
                        len(self.cats_sel)))
        return self.txts_sel,self.sample2cats_sel,self.cats_sel

    def load_train(self) -> (OrderedDict,OrderedDict,OrderedDict):
        """Loads and returns training set."""
        logger.debug(
            join(self.dataset_dir,self.dataset_name + "_txts_train.json"))
        if self.txts_train is None:
            if isfile(join(self.dataset_dir,
                           self.dataset_name + "_txts_train.json")):
                self.txts_train = File_Util.load_json(
                    self.dataset_name + "_txts_train",
                    filepath=self.dataset_dir)
            else:
                self.load_full_json()

        if self.sample2cats_train is None:
            if isfile(join(self.dataset_dir,
                           self.dataset_name + "_sample2cats_train.json")):
                self.sample2cats_train = File_Util.load_json(
                    self.dataset_name + "_sample2cats_train",
                    filepath=self.dataset_dir)
            else:
                self.load_full_json()

        if self.cats_sel is None:
            if isfile(join(self.dataset_dir,
                           self.dataset_name + "_cats_train.json")):
                self.cats_sel = File_Util.load_json(
                    self.dataset_name + "_cats_train",
                    filepath=self.dataset_dir)
            else:
                self.load_full_json()
        collect()

        # logger.info("Training data counts:\n\ttxts = [{}],\n\tClasses = [{}],\n\tCategories = [{}]"
        #             .format(len(self.txts_train), len(self.sample2cats_train), len(self.cats_train)))
        return self.txts_train,self.sample2cats_train,self.cats_sel

    def load_val(self) -> (OrderedDict,OrderedDict,OrderedDict):
        """Loads and returns validation set."""
        if self.txts_val is None:
            if isfile(join(self.dataset_dir,
                           self.dataset_name + "_txts_val.json")):
                self.txts_val = File_Util.load_json(
                    self.dataset_name + "_txts_val",filepath=self.dataset_dir)
            else:
                self.load_full_json()

        if self.sample2cats_val is None:
            if isfile(join(self.dataset_dir,
                           self.dataset_name + "_sample2cats_val.json")):
                self.sample2cats_val = File_Util.load_json(
                    self.dataset_name + "_sample2cats_val",
                    filepath=self.dataset_dir)
            else:
                self.load_full_json()

        if self.cats_val is None:
            if isfile(join(self.dataset_dir,
                           self.dataset_name + "_cats_val.json")):
                self.cats_val = File_Util.load_json(
                    self.dataset_name + "_cats_val",filepath=self.dataset_dir)
            else:
                self.load_full_json()
        collect()

        # logger.info("Validation data counts:\n\ttxts = [{}],\n\tClasses = [{}],\n\tCategories = [{}]"
        #             .format(len(self.txts_val), len(self.sample2cats_val), len(self.cats_val)))
        return self.txts_val,self.sample2cats_val,self.cats_val

    def load_test(self) -> (OrderedDict,OrderedDict,OrderedDict):
        """Loads and returns test set."""
        if self.txts_test is None:
            if isfile(join(self.dataset_dir,
                           self.dataset_name + "_txts_test.json")):
                self.txts_test = File_Util.load_json(
                    self.dataset_name + "_txts_test",filepath=self.dataset_dir)
            else:
                self.load_full_json()

        if self.sample2cats_test is None:
            if isfile(join(self.dataset_dir,
                           self.dataset_name + "_sample2cats_test.json")):
                self.sample2cats_test = File_Util.load_json(
                    self.dataset_name + "_sample2cats_test",
                    filepath=self.dataset_dir)
            else:
                self.load_full_json()

        if self.cats_test is None:
            if isfile(join(self.dataset_dir,
                           self.dataset_name + "_cats_test.json")):
                self.cats_test = File_Util.load_json(
                    self.dataset_name + "_cats_test",
                    filepath=self.dataset_dir)
            else:
                self.load_full_json()
        collect()

        # logger.info("Testing data counts:\n\ttxts = [{}],\n\tClasses = [{}],\n\tCategories = [{}]"
        #             .format(len(self.txts_test), len(self.sample2cats_test), len(self.cats_test)))
        return self.txts_test,self.sample2cats_test,self.cats_test

    def create_new_data(self,new_data_name: str = "_pointer",
                        save_files: bool = True,save_dir: str = None,
                        catid2cattxt_map: OrderedDict = None):
        """Creates new dataset based on new_data_name value, currently supports: "_fixed5" and "_onehot".

        _fixed5: Creates a dataset of samples which belongs to any of the below 5 sample2cats only.
        _onehot: Creates a dataset which belongs to single class only.

        NOTE: This method is used only for sanity testing using fixed multi-class scenario.
        """
        if save_dir is None: save_dir = join(self.dataset_dir,
                                             self.dataset_name + new_data_name)
        if isfile(join(save_dir,
                       self.dataset_name + new_data_name + "_sample2cats.json")) and isfile(
            join(save_dir,
                 self.dataset_name + new_data_name + "_txts.json")) and isfile(
            join(save_dir,self.dataset_name + new_data_name + "_cats.json")):
            logger.info("Loading files from: [{}]".format(save_dir))
            txts_new = File_Util.load_json(
                self.dataset_name + new_data_name + "_txts",filepath=save_dir)
            sample2cats_new = File_Util.load_json(
                self.dataset_name + new_data_name + "_sample2cats",
                filepath=save_dir)
            cats_new = File_Util.load_json(
                self.dataset_name + new_data_name + "_cats",filepath=save_dir)
        else:
            logger.info(
                "No existing files found at [{}]. Generating {} files.".format(
                    save_dir,new_data_name))
            if catid2cattxt_map is None: catid2cattxt_map =\
                File_Util.load_json(self.dataset_name + "_catid2cattxt_map",
                                    filepath=self.dataset_dir)

            txts,classes,_ = self.load_full_json(return_values=True)
            if new_data_name is "_fixed5":
                txts_one,classes_one,_ = self._create_oneclass_data(txts,
                                                                    classes,
                                                                    catid2cattxt_map=catid2cattxt_map)
                txts_new,sample2cats_new,cats_new =\
                    self._create_fixed_cat_data(txts_one,classes_one,
                                                catid2cattxt_map=catid2cattxt_map)
            elif new_data_name is "_onehot":
                txts_new,sample2cats_new,cats_new =\
                    self._create_oneclass_data(txts,classes,
                                               catid2cattxt_map=catid2cattxt_map)
            elif new_data_name is "_pointer":
                txts_new,sample2cats_new,cats_new =\
                    self._create_pointer_data(txts,classes,
                                              catid2cattxt_map=catid2cattxt_map)
            elif new_data_name is "_fewshot":
                txts_new,sample2cats_new,cats_new =\
                    self._create_fewshot_data(txts,classes,
                                              catid2cattxt_map=catid2cattxt_map)
            elif new_data_name is "_firstsent":
                txts_new,sample2cats_new,cats_new =\
                    self._create_firstsent_data(txts,classes,
                                                catid2cattxt_map=catid2cattxt_map)
            else:
                raise Exception(
                    "Unknown 'new_data_name': [{}]. \n Available options: ['_fixed5','_onehot', '_pointer']"
                        .format(new_data_name))
            if save_files:  # Storing new data
                logger.info(
                    "New dataset will be stored inside original dataset directory at: [{}]".format(
                        save_dir))
                makedirs(save_dir,exist_ok=True)
                File_Util.save_json(txts_new,
                                    self.dataset_name + new_data_name + "_txts",
                                    filepath=save_dir)
                File_Util.save_json(sample2cats_new,
                                    self.dataset_name + new_data_name + "_sample2cats",
                                    filepath=save_dir)
                File_Util.save_json(cats_new,
                                    self.dataset_name + new_data_name + "_cats",
                                    filepath=save_dir)

        return txts_new,sample2cats_new,cats_new

    def _create_fixed_cat_data(self,txts: OrderedDict,classes: OrderedDict,
                               fixed5_cats: list = None,
                               catid2cattxt_map=None) -> (
            OrderedDict,OrderedDict,OrderedDict):
        """Creates a dataset of samples which belongs to any of the below 5 sample2cats only.

        Selected sample2cats: [114, 3178, 3488, 1922, 517], these sample2cats has max number of samples associated with them.
        NOTE: This method is used only for sanity testing using fixed multi-class scenario.
        """
        if fixed5_cats is None: fixed5_cats = [114,3178,3488,1922,3142]
        if catid2cattxt_map is None: catid2cattxt_map = File_Util.load_json(
            self.dataset_name +
            "_catid2cattxt_map",
            filepath=self.dataset_dir)
        txts_one_fixed5 = OrderedDict()
        classes_one_fixed5 = OrderedDict()
        categories_one_fixed5 = OrderedDict()
        for doc_id,lbls in classes.items():
            if lbls[0] in fixed5_cats:
                classes_one_fixed5[doc_id] = lbls
                txts_one_fixed5[doc_id] = txts[doc_id]
                for lbl in classes_one_fixed5[doc_id]:
                    if lbl not in categories_one_fixed5:
                        categories_one_fixed5[catid2cattxt_map[str(lbl)]] = lbl

        return txts_one_fixed5,classes_one_fixed5,categories_one_fixed5

    def _create_oneclass_data(self,txts: OrderedDict,classes: OrderedDict,
                              catid2cattxt_map: OrderedDict = None) -> (
            OrderedDict,OrderedDict,OrderedDict):
        """Creates a dataset which belongs to single class only.

        NOTE: This method is used only for sanity testing using multi-class scenario.
        """
        if catid2cattxt_map is None: catid2cattxt_map = File_Util.load_json(
            self.dataset_name +
            "_catid2cattxt_map",
            filepath=self.dataset_dir)
        txts_one = OrderedDict()
        classes_one = OrderedDict()
        categories_one = OrderedDict()
        for doc_id,lbls in classes.items():
            if len(lbls) == 1:
                classes_one[doc_id] = lbls
                txts_one[doc_id] = txts[doc_id]
                for lbl in classes_one[doc_id]:
                    if lbl not in categories_one:
                        categories_one[catid2cattxt_map[str(lbl)]] = lbl

        return txts_one,classes_one,categories_one

    def cat_token_counts(self,catid2cattxt_map=None):
        """ Counts the number of tokens in categories.

        :return:
        :param catid2cattxt_map:
        """
        if catid2cattxt_map is None: catid2cattxt_map = File_Util.load_json(
            self.dataset_name + "_catid2cattxt_map",
            filepath=self.dataset_dir)
        cat_word_counts = {}
        for cat in catid2cattxt_map:
            cat_word_counts[cat] = len(self.clean.tokenizer_spacy(cat))

        return cat_word_counts

    def json2csv(self,txts_all: dict = None,sample2cats_all: dict = None):
        """ Converts existing multiple json files and returns a single pandas dataframe.

        :param txts_all:
        :param sample2cats_all:
        """
        if exists(join(self.dataset_dir,self.dataset_name + "_df.csv")):
            df = pd.read_csv(filepath_or_buffer=join(self.dataset_dir,
                                                     self.dataset_name + "_df.csv"))
            df = df[~df['txts'].isna()]
        else:
            if txts_all is None or sample2cats_all is None:
                txts_all,sample2cats_all,cats_all,cats_all = self.get_data(
                    load_type="all")
            catid2cattxt_map = File_Util.load_json(
                self.dataset_name + "_catid2cattxt_map",
                filepath=self.dataset_dir)
            txts_all_list,sample2cats_all_list,idxs,sample2catstext_all_list = [],[],[],[]
            for idx in sample2cats_all.keys():
                idxs.append(idx)
                txts_all_list.append(txts_all[idx])
                sample2cats_all_list.append(sample2cats_all[idx])
                sample2catstext = []
                for lbl in sample2cats_all[idx]:
                    sample2catstext.append(catid2cattxt_map[str(lbl)])
                sample2catstext_all_list.append(sample2catstext)

            df = pd.DataFrame.from_dict({"idx"    :idxs,
                                         "txts"   :txts_all_list,
                                         "cat"    :sample2cats_all_list,
                                         "cat_txt":sample2catstext_all_list})
            df = df[~df['txts'].isna()]
            df.to_csv(path_or_buf=join(self.dataset_dir,
                                       self.dataset_name + "_df.csv"))
        logger.info("Data shape = {} ".format(df.shape))
        return df

    def check_cat_present_txt(self,txts: OrderedDict,classes: OrderedDict,
                              catid2cattxt_map: OrderedDict = None) -> OrderedDict:
        """Generates a dict of dicts containing the positions of all categories within each text.

        :param classes:
        :param txts:
        :param catid2cattxt_map:
        :return:
        """
        if catid2cattxt_map is None:
            catid2cattxt_map = File_Util.load_json(
                self.dataset_name + "_catid2cattxt_map",
                filepath=self.dataset_dir)
        label_ptrs = OrderedDict()
        for doc_id,txt in txts.items():
            label_ptrs[doc_id] = OrderedDict()
            for lbl_id in catid2cattxt_map:
                label_ptrs[doc_id][lbl_id] = self.clean.find_label_occurrences(
                    txt,catid2cattxt_map[str(lbl_id)])
                label_ptrs[doc_id]["true"] = classes[doc_id]

        return label_ptrs

    def _create_pointer_data(self,txts: OrderedDict,classes: OrderedDict,
                             catid2cattxt_map: OrderedDict = None) -> (
            OrderedDict,OrderedDict,OrderedDict):
        """ Creates pointer network type dataset, i.e. labels are marked within document text. """
        if catid2cattxt_map is None:
            catid2cattxt_map = File_Util.load_json(
                self.dataset_name + "_catid2cattxt_map",
                filepath=self.dataset_dir)
        txts_ptr = OrderedDict()
        classes_ptr = OrderedDict()
        categories_ptr = OrderedDict()
        for doc_id,lbl_ids in classes.items():
            for lbl_id in lbl_ids:
                label_ptrs = self.clean.find_label_occurrences(txts[doc_id],
                                                               catid2cattxt_map[
                                                                   str(lbl_id)])
                if label_ptrs:  ## Only if categories exists within the document.
                    classes_ptr[doc_id] = {lbl_id:label_ptrs}
                    txts_ptr[doc_id] = txts[doc_id]

                    if lbl_id not in categories_ptr:
                        categories_ptr[lbl_id] = catid2cattxt_map[str(lbl_id)]

        return txts_ptr,classes_ptr,categories_ptr

    def _create_fewshot_data(self,txts: OrderedDict,classes: OrderedDict,
                             catid2cattxt_map: OrderedDict = None) -> (
            OrderedDict,OrderedDict,OrderedDict):
        """Creates few-shot dataset, i.e. categories with <= 20 samples.

        :param classes:
        :param txts:
        :param catid2cattxt_map:
        :return:
        """
        if catid2cattxt_map is None:
            catid2cattxt_map = File_Util.load_json(
                self.dataset_name + "_catid2cattxt_map",
                filepath=self.dataset_dir)

        tail_cats,samples_with_tail_cats,cat2samples_filtered = self.find_cats_with_few_samples(
            sample2cats_map=classes,
            cat2samples_map=None)
        txts_few = OrderedDict()
        classes_few = OrderedDict()
        categories_few = OrderedDict()
        for doc_id,lbls in classes.items():
            if len(lbls) == 1:
                classes_few[doc_id] = lbls
                txts_few[doc_id] = txts[doc_id]
                for lbl in classes_few[doc_id]:
                    if lbl not in categories_few:
                        categories_few[catid2cattxt_map[str(lbl)]] = lbl

        return txts_few,classes_few,categories_few

    def _create_firstsent_data(self,txts: OrderedDict,classes: OrderedDict,
                               catid2cattxt_map: OrderedDict = None) -> (
            OrderedDict,OrderedDict,OrderedDict):
        """Creates a version of wikipedia dataset with only first sentences and discarding the text.

        :param classes:
        :param txts:
        :param catid2cattxt_map:
        :return:
        """
        if catid2cattxt_map is None:
            catid2cattxt_map = File_Util.load_json(
                self.dataset_name + "_catid2cattxt_map",
                filepath=self.dataset_dir)

        txts_firstsent = OrderedDict()
        classes_firstsent = OrderedDict()
        categories_firstsent = OrderedDict()
        for doc_id,lbls in classes.items():
            if len(lbls) == 1:
                classes_firstsent[doc_id] = lbls
                txts_firstsent[doc_id] = txts[doc_id]
                for lbl in classes_firstsent[doc_id]:
                    if lbl not in categories_firstsent:
                        categories_firstsent[catid2cattxt_map[str(lbl)]] = lbl

        return txts_firstsent,classes_firstsent,categories_firstsent


def main():
    # save_dir = join(config["paths"]["dataset_dir"][plat][user],config["data"]["dataset_name"],
    #                 config["data"]["dataset_name"] + "_pointer")

    common_handler = Common_Data_Handler(
        dataset_type=config["xc_datasets"][config["data"]["dataset_name"]],
        dataset_name=config["data"]["dataset_name"],
        dataset_dir=config["paths"]["dataset_dir"][plat][user])
    # df = common_handler.json2csv()

    df = pd.read_csv(filepath_or_buffer=join(config["paths"]["dataset_dir"][plat][user],config["data"]["dataset_name"],
                                                     config["data"]["dataset_name"] + "_df.csv"))
    multiclass_df = common_handler.multilabel2multiclass_df(df)
    logger.debug(multiclass_df.head())
    # txts_new,sample2cats_new,cats_new = common_handler.create_new_data(new_data_name="_fewshot",save_dir=save_dir)
    # logger.debug(len(txts_new))
    # logger.debug(len(sample2cats_new))
    # logger.debug(len(cats_new))

    # txts_one, classes_one, categories_one = common_handler.create_oneclass_data(save_dir)
    # catid2cattxt_map = File_Util.load_json(config["data"]["dataset_name"] + "_catid2cattxt_map",
    #                                       filepath=join(config["paths"]["dataset_dir"][plat][user],
    #                                                      config["data"]["dataset_name"]),
    #                                       show_path=True)
    # txts_new,sample2cats_new,cats_new = common_handler.create_new_data(new_data_name="_pointer",save_files=True,
    #                                                                           save_dir=None,
    #                                                                           catid2cattxt_map=catid2cattxt_map)
    # logger.debug(len(txts_new))
    # logger.debug(len(sample2cats_new))
    # logger.debug(len(cats_new))

    # txts_new, sample2cats_new, cats_new = \
    #     common_handler.create_new_data(new_data_name="_onehot", save_files=True, save_dir=None,
    #                                    catid2cattxt_map=catid2cattxt_map)
    # logger.debug(len(txts_new))
    # logger.debug(len(sample2cats_new))
    # logger.debug(len(cats_new))
    # txts_train, sample2cats_train, cats, _, _, _, txts_test, sample2cats_test, cats_test, catid2cattxt_map = common_handler.split_data(
    #     txts_one, classes_one, categories_one, val_split=0.0)
    # logger.debug(len(txts_train))
    # logger.debug(len(sample2cats_train))
    # logger.debug(len(cats))
    # logger.debug(len(txts_test))
    # logger.debug(len(sample2cats_test))
    # logger.debug(len(cats_test))
    # util.save_json(txts_train, config["data"]["dataset_name"] + "_txts_train", filepath=save_dir)
    # util.save_json(sample2cats_train, config["data"]["dataset_name"] + "_sample2cats_train", filepath=save_dir)
    # util.save_json(cats, config["data"]["dataset_name"] + "_cats_train", filepath=save_dir)
    # util.save_json(txts_test, config["data"]["dataset_name"] + "_txts_test", filepath=save_dir)
    # util.save_json(sample2cats_test, config["data"]["dataset_name"] + "_sample2cats_test", filepath=save_dir)
    # util.save_json(cats_test, config["data"]["dataset_name"] + "_cats_test", filepath=save_dir)
    # Using Val set as Test set also.
    # util.save_json(txts_test, config["data"]["dataset_name"] + "_txts_val", filepath=save_dir)
    # util.save_json(sample2cats_test, config["data"]["dataset_name"] + "_sample2cats_val", filepath=save_dir)
    # util.save_json(cats_test, config["data"]["dataset_name"] + "_cats_val", filepath=save_dir)

    # util.print_dict(classes_one, count=5)
    # util.print_dict(txts_one, count=5)
    # util.print_dict(categories_one, count=5)
    # logger.debug(classes_one)
    # txts_val, sample2cats_val, cats_val = common_handler.load_val()
    # util.print_dict(txts_val)
    # util.print_dict(sample2cats_val)
    # util.print_dict(cats_val)


if __name__ == '__main__':
    main()
