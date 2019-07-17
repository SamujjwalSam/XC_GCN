# coding=utf-8
# !/usr/bin/python3.6
"""
__synopsis__    : Visualize vectors in 2D using tsne.

__description__ :
__project__     : XC_GCN
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ":  "
__date__        : "26/06/19"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : class_name

__variables__   :

__methods__     :
"""

import gensim
from gensim.test.utils import datapath,get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from sklearn.manifold import TSNE

from logger import logger
from file_utils import File_Util
from text_process import clean_wiki,Clean_Text
from config import configuration as config
from config import platform as plat
from config import username as user


class tsne_visualize:
    """ Visualize vectors in 2D using tsne. """

    def __init__(self) -> None:
        super(tsne_visualize,self).__init__()
        self.clean = Clean_Text()

    def display_closestwords_tsnescatterplot(self,model,word,word_dim=300,sim_words=5):
        """ Method to plot the top sim_words in 2D using TSNE.

        :param model:
        :param word:
        :param word_dim:
        :param sim_words:
        :param plot_title:
        """
        arr = np.empty((0,word_dim),dtype='f')
        word_labels = [word]

        # get close words
        close_words = model.similar_by_word(word,topn=sim_words)

        # add the vector for each of the closest words to the array
        arr = np.append(arr,np.array([model[word]]),axis=0)
        for wrd_score in close_words:
            wrd_vector = model[wrd_score[0]]
            word_labels.append(wrd_score[0])
            arr = np.append(arr,np.array([wrd_vector]),axis=0)

        self.use_tsne(arr,word_labels)

    @staticmethod
    def use_tsne(vecs,word_labels,plot_title="Similar Words"):
        """ Use tsne to project the vectors.

        :param vecs:
        :param word_labels:
        :param plot_title:
        """
        # find tsne coords for 2 dimensions
        tsne = TSNE(n_components=2,random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(vecs)

        x_coords = Y[:,0]
        y_coords = Y[:,1]

        # display scatter plot
        plt.scatter(x_coords,y_coords)

        for label,x,y in zip(word_labels,x_coords,y_coords):
            plt.annotate(label,xy=(x,y),xytext=(0,0),textcoords='offset points')
        plt.xlim(x_coords.min() + 0.00005,x_coords.max() + 0.00005)
        plt.ylim(y_coords.min() + 0.00005,y_coords.max() + 0.00005)
        plt.xticks(rotation=35)
        plt.title(plot_title)
        plt.show()

        return Y

    @staticmethod
    def use_pca(vecs,word_labels,plot_title="2D tsne representations"):
        """ Use tsne to project the vectors.

        :param vecs:
        :param word_labels:
        :param plot_title:
        """
        # find tsne coords for 2 dimensions
        tsne = TSNE(n_components=2,random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(vecs)

        x_coords = Y[:,0]
        y_coords = Y[:,1]

        # display scatter plot
        plt.scatter(x_coords,y_coords)

        for label,x,y in zip(word_labels,x_coords,y_coords):
            plt.annotate(label,xy=(x,y),xytext=(0,0),textcoords='offset points')
        plt.xlim(x_coords.min() + 0.00005,x_coords.max() + 0.00005)
        plt.ylim(y_coords.min() + 0.00005,y_coords.max() + 0.00005)
        plt.xticks(rotation=35)
        plt.title(plot_title)
        plt.show()

        return Y

    @staticmethod
    def load_model(path: str = config["paths"]["pretrain_dir"][plat][user],
                   filename='GoogleNews-vectors-negative300.bin'):
        """

        :param path:
        :param filename:
        :return:
        """
        logger.debug("Loading GoogleNews pre-trained model.")
        model = gensim.models.KeyedVectors.load_word2vec_format(join(path,filename),binary=True,encoding='latin-1')
        return model

    @staticmethod
    def load_glove(path: str = config["paths"]["pretrain_dir"][plat][user],filename='glove.840B.300d.txt'):
        """

        :param path:
        :param filename:
        :return:
        """
        glove_file = datapath(join(path,filename))
        tmp_file = get_tmpfile(join(path,filename + "_word2vec"))
        _ = glove2word2vec(glove_file,tmp_file)
        model = KeyedVectors.load_word2vec_format(tmp_file)
        return model

    @staticmethod
    def load_categories(path: str = config["paths"]["dataset_dir"][plat][user],label_file: str = 'categories.json'):
        """ Loads the categories file.

        :param path:
        :param label_file:
        """
        cats_dict = File_Util.load_json(join(path,label_file))
        return cats_dict.values()

    def process_categories(self,labels: dict):
        """

        :param labels:
        :return:
        """
        labels_processed = self.clean.clean_sentences_dict(labels,
                                                           specials="""?!_-@<>#,.*?'{}[]()$%^~`:;"/\\:|""")
        labels_tokens = []
        for lbl_txt in labels_processed.values():
            labels_tokens.append(self.clean.tokenizer_spacy(lbl_txt))

        return labels_tokens

    @staticmethod
    def gen_cat_vecs(model,labels_tokens: list):
        """

        :param model:
        :param labels_tokens:
        :return:
        """
        lbl_vecs = []
        # token_vec_shape = model.get_vector("any")
        oov_tokens = {}
        for lbl in labels_tokens:
            lbl_vec = np.zeros_like((1,300))
            for token in lbl:
                try:
                    lbl_vec = model.get_vector(token)
                except KeyError:
                    try:
                        lbl_vec = oov_tokens[token]
                    except KeyError:
                        # logger.debug("Vector for token [{}] not found.".format(token))
                        # logger.debug("Error: [{}]".format(e))
                        lbl_vec = np.random.uniform(-0.5,0.5,300)
                        oov_tokens[token] = lbl_vec
                np.add(lbl_vec,lbl_vec)
            lbl_vecs.append(np.divide(lbl_vec,len(lbl)))

        return np.stack(lbl_vecs)

    @staticmethod
    def get_cosine_dist(vecs):
        """

        :param vecs:
        :return:
        """
        from sklearn.metrics.pairwise import cosine_similarity

        pair_cosine = cosine_similarity(vecs,vecs)
        return pair_cosine


def main():
    """ main module to start code.

    :param param:
        Type: tuple
        Required
        Read Only
    :return:
    """
    cats = {
  "9431": "Memory management",
  "5708": "English idioms",
  "29387": "VoIP companies",
  "30665": "West Bank",
  "9120": "Entrepreneurship",
  "22886": "Accounting systems",
  "2324": "XML",
  "16255": "Journalism",
  "5504": "Diodes",
  "614": "Attitude change",
  "11279": "Live action role playing games",
  "4258": "Web development",
  "114": "Disambiguation pages",
  "850": "Cryptographic attacks",
  "6347": "Artificial life",
  "1094": "Chess variants",
  "24162": "Gender inclusive language",
  "2388": "Philosophical terminology",
  "33946": "2005",
  "34923": "Eminent domain",
  "18163": "Virtual memory",
  "1587": "Management",
  "23216": "Medical classification",
  "2272": "Web mapping",
  "2066": "XML based standards",
  "1281": "Derivatives",
  "36363": "Mormonism",
  "1559": "Surrealism",
  "28976": "Energy drinks",
  "3488": "MBTI types",
  "34318": "Animation",
  "14980": "Lists of comedy television series episodes",
  "17037": "Library automation",
  "4248": "Sociological terms",
  "3686": "Rhetorical techniques",
  "1795": "Systems engineering",
  "33427": "Optical phenomena",
  "1182": "Semantic Web",
  "25096": "Compulsive hoarding",
  "718": "Drinking games",
  "14820": "Lists of best seller albums",
  "15660": "Cosmology",
  "242": ".NET framework",
  "24265": "Electronics",
  "7838": "Medical imaging",
  "14589": "Evidence",
  "5964": "Japanese folklore",
  "20129": "Time zones",
  "12437": "Warcraft",
  "18586": "Aquaculture",
  "15284": "Broadband",
  "5132": "Anomalous weather",
  "22980": "Data security",
  "3142": "Software development process",
  "26751": "African culture",
  "9304": "Academic publishing",
  "4044": "Viruses",
  "20624": "Nebula awards",
  "7759": "Microsoft lists",
  "16474": "Electronic design",
  "21093": "Judaism",
  "26341": "Military tactics",
  "11563": "Language games",
  "6203": "Widgets",
  "2525": "Human computer interaction",
  "509": "User interface techniques",
  "8915": "Cargo cults",
  "36070": "Flexible fuel vehicles",
  "419": "Web security exploits",
  "9330": "Binary trees",
  "10781": "Risk in finance",
  "6491": "URL",
  "2393": "Classical ciphers",
  "12189": "Enlargement of the European Union",
  "19880": "Executable file formats",
  "4886": "Software comparisons",
  "21990": "Epigenetics",
  "566": "Neo noir",
  "646": "Musical notation",
  "6550": "Risk",
  "10887": "Syntactic entities",
  "31082": "Articles lacking in text citations from (February 2008",
  "4289": "Stochastic processes",
  "13499": "Dance styles",
  "51": "Nutrition",
  "2128": "Software engineering",
  "11761": "Medical ethics",
  "16175": "Options",
  "21780": "History of video games",
  "9601": "Web 2.0 neologisms",
  "14287": "X86 architecture",
  "510": "Dice games",
  "1954": "Divination",
  "490": "Online social networking",
  "21535": "Peer to peer lending companies",
  "12307": "Cartographic projections",
  "1215": "Databases",
  "1797": "File sharing networks",
  "1922": "Programming paradigms",
  "10913": "Electronics manufacturing",
  "3825": "Internet memes",
  "135": "Cryptography",
  "22151": "Music industry",
  "5958": "Functional programming",
  "3025": "Programming evaluation",
  "12717": "Modems",
  "18428": "Pilates",
  "11228": "Windows CE",
  "1867": "Data management",
  "15607": "Birthdays",
  "9794": "Game design",
  "3384": "Graphics file formats",
  "463": "Cognitive biases",
  "1441": "Library and information science",
  "22485": "Transducers",
  "24758": "Abbreviations",
  "20874": "Baseball statistics",
  "31116": "Britpop",
  "12394": "Type theory",
  "867": "Game theory",
  "9617": "Poetry movements",
  "19056": "Integers",
  "21888": "Proxy servers",
  "24462": "Graph families",
  "22599": "Kawasaki motorcycles",
  "13751": "Radio modulation modes",
  "8082": "Beatboxing",
  "20084": "IBM Mainframe computer operating systems",
  "9205": "Gross pathology",
  "5732": "Economic history of the United States",
  "34592": "2007 in music",
  "5841": "Interpolation",
  "37003": "Terrorist incidents",
  "16626": "Free web server software",
  "7000": "Java programming language",
  "8059": "Christmas",
  "18981": "Personality",
  "18015": "Gate arrays",
  "6346": "Musical techniques",
  "517": "Unix software",
  "35928": "Canon PowerShot cameras",
  "15345": "Memory",
  "9196": "Axiom of choice",
  "12596": "Human physiology",
  "1297": "Film techniques",
  "6349": "Java platform",
  "12": "Visualization (graphic)",
  "1613": "Tissues",
  "473": "Business terms",
  "2369": "Internet culture",
  "12035": "Popular psychology",
  "2514": "Personality disorders",
  "9310": "Biodiesel",
  "72": "Programming bugs",
  "13077": "Linked lists",
  "28298": "Buddhism",
  "2406": "Politics of the United Kingdom",
  "36334": "Capsaicinoids",
  "33334": "1875 poems",
  "2334": "Computational models",
  "815": "Wikis",
  "6202": "Hypertext",
  "13790": "Procurement",
  "3307": "Information technology management",
  "4167": "Mexican culture",
  "17631": "Discrete mathematics",
  "19004": "Landscape architecture",
  "7890": "Pendulums",
  "28554": "20th century poems",
  "12366": "Musical composition",
  "4485": "Data types",
  "26671": "Fertility tracking",
  "3992": "Information retrieval",
  "5057": "German loanwords",
  "9465": "Software metrics",
  "2335": "Fractals",
  "31070": "Military exercises and wargames",
  "1895": "Anxiety disorders",
  "1193": "Intel x86 microprocessors",
  "2884": "Statistical terminology",
  "1206": "Multi sport competitions",
  "22202": "X servers",
  "14320": "Experimental design",
  "620": "3D graphics software",
  "10315": "Video formats",
  "8094": "VMware",
  "3077": "Video codecs",
  "19975": "Gambling",
  "1566": "Ophthalmology",
  "9610": "Checksum algorithms",
  "12402": "Macross",
  "19169": "English words",
  "106": "Spam filtering",
  "24031": "Directed graphs",
  "10355": "Disorders causing seizures",
  "11769": "Education in the United States",
  "10297": "Electric motors",
  "3895": "Art genres",
  "23478": "Demos",
  "10787": "Human based computation",
  "2740": "Database normalization",
  "2881": "Web hosting",
  "6546": "Chemical bonding",
  "9455": "Statistical charts and diagrams",
  "16397": "Corrective lenses",
  "9492": "Islam",
  "10754": "Wireless e mail",
  "12446": "Programming language classification",
  "8057": "Covariance and correlation",
  "3479": "Buddhist philosophical concepts",
  "23025": "Magic tricks",
  "745": "Software architecture",
  "21389": "Computational fluid dynamics",
  "5314": "Pedagogy",
  "4623": "Object oriented programming",
  "507": "Greek loanwords",
  "13372": "Fractal curves",
  "34778": "Cities and towns in Upper Austria",
  "10983": "Comics awards",
  "14056": "Nokia mobile phones",
  "21957": "Platonism",
  "1799": "Film and video technology",
  "6213": "Telecommunications",
  "4280": "Usenet",
  "18046": "Internal combustion engine",
  "3991": "Dance notation",
  "2445": "Discrimination",
  "3832": "Evolutionary biology",
  "2806": "Technical analysis",
  "18926": "Mystery Science Theater 3000",
  "4621": "Source code",
  "13797": "Warfare of the Medieval era",
  "2357": "Topology",
  "2065": "Application layer protocols",
  "9012": "Digital television",
  "3078": "Interactive television",
  "3240": "Esoteric programming languages",
  "3597": "Join algorithms",
  "1556": "Communication",
  "1298": "Botany",
  "1407": "Software design patterns",
  "909": "Photographic techniques",
  "294": "Quantum information science",
  "5018": "Japanese words and phrases",
  "11947": "Science fiction genres",
  "13571": "Body modification",
  "4925": "Viral diseases",
  "3233": "East Africa",
  "18823": "Dermatologic procedures and surgery",
  "4901": "Physical cosmology",
  "6111": "Vector calculus",
  "1680": "Sleep disorders",
  "1483": "Greek mythology",
  "12666": "OLAP",
  "2831": "Motivational theories",
  "36768": "Acting techniques",
  "18680": "Shades of violet",
  "28524": "Screenwriting software",
  "10013": "Periodic table",
  "26620": "Types of insurance",
  "10074": "Meetings",
  "5381": "Trees (structure)",
  "1603": "Software requirements",
  "7707": "Video game gameplay",
  "3179": "Software development philosophies",
  "2362": "Microprocessors",
  "508": "Telephone services",
  "10457": "Warfare by type",
  "5182": "Systems of formal logic",
  "23445": "Cultural policies of the European Union",
  "6244": "Internet slang",
  "17346": "Password authentication",
  "20626": "Public speaking",
  "12293": "Automation",
  "29617": "Underground Railroad",
  "259": "Marketing",
  "1440": "Information science",
  "1798": "World Wide Web",
  "1866": "Video editing software",
  "719": "Top level domains",
  "8502": "Auditory illusions",
  "15308": "Digital gold currencies",
  "21955": "Memory disorders",
  "4089": "Memory processes",
  "7431": "Marketing strategies and paradigms",
  "8676": "Line codes",
  "8256": "Board games",
  "5690": "Markup languages",
  "4233": "Attention deficit hyperactivity disorder",
  "12798": "Fusion power",
  "18809": "Gears",
  "16684": "Algorithms on strings",
  "14586": "Criminal law",
  "27004": "Free web analytics software",
  "90": "Latin legal phrases",
  "1630": "Optimization algorithms",
  "1593": "Atmospheric optical phenomena",
  "6431": "Logic puzzles",
  "28850": "Ampakines",
  "9853": "Ergonomics",
  "4196": "Product management",
  "21640": "BBC television documentaries on history",
  "6288": "Latin language",
  "712": "Social psychology",
  "20143": "Weather modification",
  "5108": "Cellular automaton rules",
  "505": "Taxonomy",
  "29886": "Software engineering costs",
  "6379": "Barcodes",
  "3249": "HTML",
  "11676": "Geneva Conventions"
}

    visul = tsne_visualize()
    cats_pro = visul.process_categories(cats)
    logger.debug(cats_pro)
    model = visul.load_model(filename="glove.840B.300d.txt_word2vec")
    cats_pro = visul.gen_cat_vecs(model,cats_pro)
    logger.debug(cats_pro)
    cats_pro = visul.use_tsne(cats_pro,list(cats.values()))
    logger.debug(cats_pro)
    cats_pro = visul.get_cosine_dist(cats_pro)
    logger.debug(cats_pro)


if __name__ == "__main__":
    main()
