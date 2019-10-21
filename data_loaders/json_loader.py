# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Class to process and load json files from a directory.

__description__ : Class to process and load json files from a directory.
__project__     : XCGCN
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : JSONLoader
"""

from os.path import join
import torch.utils.data
from collections import OrderedDict
from smart_open import open as sopen  # Better alternative to Python open().

from file_utils import File_Util
from logger import logger
from config import configuration as config
from config import platform as plat
from config import username as user

seed_val = 0


class JSONLoader(torch.utils.data.Dataset):
    """
    Class to process and load json files from data directory.

    Datasets: AmazonCat-14K

    txts : Amazon products title + description after parsing and cleaning.
    txts = {"id1": "azn_ttl_1", "id2": "azn_ttl_2"}

    sample2cats   : OrderedDict of id to sample2cats.
    sample2cats = {"id1": [class_id_1,class_id_2],"id2": [class_id_2,class_id_10]}

    cats : Dict of class texts.
    cats = {"Computer Science":class_id_1, "Machine Learning":class_id_2}

    samples : {
        "txts":"",
        "sample2cats":""
        }
    """

    def __init__(self, dataset_name=config["data"]["dataset_name"], dataset_dir: str = config["paths"]["dataset_dir"][plat][user]):
        """
        Initializes the JSON loader.

        Args:
            dataset_dir : Path to directory of the dataset.
            dataset_name : Name of the dataset.
        """
        super(JSONLoader, self).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = join(dataset_dir, self.dataset_name)
        self.raw_json_dir = join(self.dataset_dir, self.dataset_name + "_RawData")
        self.raw_json_file = self.dataset_name + "_RawData.json"
        logger.info("Dataset name: [{}], Directory: [{}]".format(self.dataset_name, self.dataset_dir))
        self.txts, self.classes, self.cats = self.gen_dicts(json_path=join(self.raw_json_dir,self.raw_json_file), encoding="UTF-8")

    def gen_dicts(self,json_path=None, encoding=config["text_process"]["encoding"],specials="""_-@*#'"/\\""", replace=' '):
        """
        Generates the data dictionaries from original json file.

        :param replace: Character to replace with.
        :param specials: Characters to clean from txts.
        :param json_path: Path to raw json file.
        :param encoding: Encoding for the raw json file.
        :return: txts, sample2cats, cats, no_cat_ids
            no_cat_ids: ids for which no cats were found.
        """
        import ast  # As the data is not proper JSON (single-quote instead of double-quote) format, "json" library will not work.
        from unidecode import unidecode

        logger.info("Generates the data dictionaries from original json file.")
        txts = OrderedDict()
        classes = OrderedDict()
        cats = OrderedDict()
        no_cat_ids = []  # To store ids for which no cats were found.

        if json_path is None: json_path = self.raw_json_dir
        with sopen(json_path, encoding=encoding) as raw_json_ptr:
            trans_table = File_Util.make_trans_table(specials=specials,replace=replace)  # Creating mapping to clean txts.
            cat_idx = 0  # Holds the category index.
            for cnt, line in enumerate(raw_json_ptr):
                # Instead of: line_dict = OrderedDict(json.loads(line));
                # Use: import ast; line_dict = ast.literal_eval(line.strip().replace('\n','\\n'));
                line_dict = ast.literal_eval(line.strip().replace('\n','\\n'))
                if "categories" in line_dict:  # Check if "cats" exists.
                    if "title" in line_dict:  # Check if "title" exists, add if True.
                        txts[line_dict["asin"]] = unidecode(str(line_dict["title"])).translate(trans_table)
                        if "description" in line_dict:  # Check if "description" exists and append to "title" with keyword: ". \nDESC: ", if true.
                            txts[line_dict["asin"]] = txts[line_dict["asin"]] + ". \nDESC: " + unidecode(str(line_dict["description"])).translate(trans_table)
                    else:
                        if "description" in line_dict:  # Check if "description" exists even though "title" does not, use only "description" if true.
                            txts[line_dict["asin"]] = ". \nDESC: " + line_dict["description"]
                        else:  # Report and skip the sample if neither "title" nor "description" exists.
                            logger.warning("Neither 'title' nor 'description' found for sample id: [{}]. Adding sample to 'no_cat_ids'.".format(line_dict["asin"]))
                            no_cat_ids.append(line_dict["asin"])  # As neither "title" nor "description" exists, adding the id to "no_cat_ids".
                            continue
                    classes[line_dict["asin"]] = line_dict["cats"][0]
                    for lbl in classes[line_dict["asin"]]:
                        if lbl not in cats:  # If lbl does not exists in cats already, add it and assign a new category index.
                            cats[lbl] = cat_idx
                            cat_idx += 1
                        classes[line_dict["asin"]][classes[line_dict["asin"]].index(lbl)] = cats[lbl]  # Replacing cats text to cats id.
                else:  # if "categories" does not exist, then add the id to "no_cat_ids".
                    no_cat_ids.append(line_dict["asin"])

        File_Util.save_json(no_cat_ids,self.dataset_name + "_no_cat_ids",filepath=self.dataset_dir)
        logger.info("Number of txts: [{}], sample2cats: [{}] and cats: [{}]."
                    .format(len(txts),len(classes),len(cats)))
        return txts, classes, cats

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
    cls = JSONLoader()
    cats_val = cls.get_categories()
    logger.print_dict(cats_val)


if __name__ == '__main__':
    main()
