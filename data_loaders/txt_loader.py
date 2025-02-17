# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Class to process and load txt files from a directory.

__description__ : Class to process and load txt files from a directory.
__project__     : XCGCN
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : TXTLoader,
"""

from os.path import join
import torch.utils.data
from collections import OrderedDict
from smart_open import open as sopen  # Better alternative to Python open().

from logger import logger
from config import configuration as config
from config import platform as plat
from config import username as user


class TXTLoader(torch.utils.data.Dataset):
    """
    Class to process and load txt files from a directory.

    Datasets: AmazonCat-14K

    txts : Amazon products title + description after parsing and cleaning.
    txts = {"id1": "azn_ttl_1", "id2": "azn_ttl_2"}

    sample2cats   : OrderedDict of id to sample2cats.
    sample2cats = {"id1": [class_id_1,class_id_2],"id2": [class_id_2,class_id_10]}

    cattext2catid_map : Dict of class texts.
    cattext2catid_map = {"Computer Science":class_id_1, "Machine Learning":class_id_2}

    samples : {
        "txts":"",
        "sample2cats":""
        }
    """

    def __init__(self, dataset_name=config["data"]["dataset_name"], dataset_dir: str = config["paths"]["dataset_dir"][plat]):
        """
        Initializes the TXT loader.

        Args:
            dataset_dir : Path to directory containing the txt files.
            dataset_name : Name of the dataset.
        """
        super(TXTLoader, self).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = join(dataset_dir, self.dataset_name)
        self.raw_txt_dir = join(self.dataset_dir, self.dataset_name + "_RawData")
        # self.raw_txt_file = self.dataset_name + "_RawData.txt"
        logger.info("Dataset name: [{}], Directory: [{}]".format(self.dataset_name, self.dataset_dir))
        self.txts, self.classes, self.cats = self.gen_dicts()

    def gen_dicts(self, encoding='UTF-8'):
        """
        Loads the txt files.

        :return:
        """
        classes = OrderedDict()
        cats = OrderedDict()
        cat_idx = 0
        all_classes = self.read_classes(encoding=encoding)
        # logger.debug((len(all_classes)))
        # util.print_dict(all_classes)
        titles = self.read_titles(classes_keys=all_classes.keys(), encoding=encoding)
        descriptions = self.read_desc(classes_keys=all_classes.keys(), encoding=encoding)
        txts = self.create_txts(titles,descriptions)
        classes_extra = set(all_classes.keys()).symmetric_difference(set(txts.keys()))
        if len(classes_extra):
            for k, v in all_classes.items():
                if k not in classes_extra:
                    classes[k] = v
                    for lbl in classes[k]:
                        if lbl not in cats:  # If lbl does not exists in cats already, add it and assign a new category index.
                            cats[lbl] = cat_idx
                            cat_idx += 1
                        classes[k][classes[k].index(lbl)] = cats[lbl]  # Replacing cats text to cats id.
        logger.print_dict(txts)
        logger.print_dict(classes)
        logger.print_dict(cats)

        return txts, classes, cats

    def read_classes(self, classes_dir=None, classes_file="cats.txt", encoding=config["text_process"]["encoding"]):
        """
        Reads the cats.txt file and returns a OrderedDict of id : class ids.

        :param classes_file:
        :param classes_dir:
        :param encoding:
        :return:
        """
        logger.info("Reads the cats.txt file and returns a OrderedDict of id : class ids.")
        cat_line_phrase = "  "  # Phrase to recognize lines with category information.
        cat_sep_phrase = ", "  # Phrase to separate cats.
        classes = OrderedDict()
        cat_pool = set()
        if classes_dir is None: classes_dir = self.raw_txt_dir
        with sopen(join(classes_dir, classes_file), encoding=encoding) as raw_cat_ptr:
            sample_idx = raw_cat_ptr.readline().strip()
            for cnt, line in enumerate(raw_cat_ptr):
                if cat_line_phrase in line:
                    cats = line.split(cat_sep_phrase)  # Splliting line based on ', ' to get cats.
                    cats = [x.strip() for x in cats]  # Removing extra characters like: ' ','\n'.
                    cat_pool.update(cats)
                else:
                    classes[sample_idx] = list(cat_pool)
                    cat_pool.clear()
                    sample_idx = line.strip()

        return classes

    def read_titles(self, classes_keys=None, title_path=None, title_file="titles.txt", encoding=config["text_process"]["encoding"]):
        """
        Reads the titles.txt file and returns a OrderedDict of id : title.

        :param classes_keys: List of sample2cats keys to check only those keys are stored.
        :param title_file:
        :param title_path:
        :param encoding:
        :return:
        """
        logger.info("Reads the titles.txt file and returns a OrderedDict of id : title.")
        titles = OrderedDict()
        if title_path is None: title_path = join(self.raw_txt_dir, title_file)
        with sopen(title_path, encoding=encoding) as raw_title_ptr:
            for cnt, line in enumerate(raw_title_ptr):
                line = line.split()
                if classes_keys is None or line[0] in classes_keys:  # Add this sample if corresponding sample2cats exists.
                    titles[line[0].strip()] = " ".join(line[1:]).strip()
        return titles

    def read_desc(self, classes_keys=None, desc_path=None, desc_file="descriptions.txt",
                  encoding=config["text_process"]["encoding"]):
        """
        Reads the descriptions.txt file and returns a OrderedDict of id : desc.

        :param classes_keys:
        :param desc_file:
        :param desc_path:
        :param encoding:
        :return:
        """
        id_phrase = "product/productId: "  # Phrase to recognize lines with sample id.
        id_remove = 19  # Length of [id_phrase], to be removed from line.
        desc_phrase = "product/description: "  # Phrase to recognize lines with sample description.
        desc_remove = 21  # Length of [desc_phrase], to be removed from line.
        logger.info("Reads the descriptions.txt file and returns a OrderedDict of id : desc.")
        descriptions = OrderedDict()
        if desc_path is None: desc_path = join(self.raw_txt_dir, desc_file)
        import itertools
        with sopen(desc_path, encoding=encoding) as raw_desc_ptr:
            for idx_line, desc_line in itertools.zip_longest(
                    *[raw_desc_ptr] * 2):  # Reads multi-line [2] per iteration.
                if id_phrase in idx_line:
                    sample_id = idx_line[id_remove:].strip()
                    if classes_keys is None or sample_id in classes_keys:  # Add this sample if corresponding class exists.
                        if desc_phrase in desc_line:
                            sample_desc = desc_line[desc_remove:].strip()
                        else:
                            sample_desc = None  # Even if 'description' is not found, we are not ignoring the sample as it might still have text in 'title'.
                        descriptions[sample_id] = sample_desc
        return descriptions

    def create_txts(self, titles, descriptions=None):
        """
        Creates txts for each sample by using either title or descriptions if only one exists else appends desc to title.

        :param titles:
        :param descriptions:
        :return:
        """
        logger.info("Creates txts for each sample by using either title or descriptions if only one exists else appends desc to title.")
        txts = OrderedDict()
        if descriptions is not None:
            intersect = set(titles.keys()).intersection(set(descriptions.keys()))
            logger.info("[{}] samples have both 'title' and 'description'.".format(len(intersect)))
            for idx in intersect:
                txts[idx] = titles[idx] + ". \nDESC: " + descriptions[idx]
            sym_dif = set(titles.keys()).symmetric_difference(set(descriptions.keys()))
            if len(sym_dif):
                logger.info("[{}] samples either only have 'title' or 'description'.".format(len(sym_dif)))
                for idx in sym_dif:
                    if idx in titles.keys():
                        txts[idx] = titles[idx]
                    else:
                        txts[idx] = descriptions[idx]
        else:
            logger.info("'description' data not provided, only using 'title'.")
            for idx in titles.keys():
                txts[idx] = titles[idx]
        return txts

    def get_data(self):
        """
        Function to get the entire dataset
        """
        return self.txts, self.classes, self.cats

    def get_txts(self):
        """
        Function to get the entire set of features
        """
        return self.txts

    def get_classes(self):
        """
        Function to get the entire set of sample2cats.
        """
        return self.classes

    def get_cats(self) -> dict:
        """
        Function to get the entire set of cats
        """
        return self.cats


def main():
    # config = read_config(args)
    cls = TXTLoader()
    txts_val, sample2cats_val, cats_val = cls.get_val_data()
    logger.print_dict(txts_val)
    logger.print_dict(sample2cats_val)
    logger.print_dict(cats_val)


if __name__ == '__main__':
    main()
