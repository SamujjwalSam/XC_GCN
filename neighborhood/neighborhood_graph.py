# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Generates neighborhood graph based on label co-occurrence between samples.

__description__ : Generates neighborhood graph based on label co-occurrence between samples.
__project__     : XCGCN
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : Neighborhood_Graph
"""

import numpy as np
import networkx as nx
from os.path import join,exists
from queue import Queue  # Python 2.7 does not have this library
from collections import OrderedDict

from file_utils import File_Util
from logger import logger
from config import configuration as config
from config import platform as plat
from config import username as user


class Neighborhood_Graph:
    """ Class to generate neighborhood graph of categories. """

    def __init__(self,dataset_name: str = config["data"]["dataset_name"],graph_format: str = "graphml",top_k: int = 10,
                 graph_dir: str = config["paths"]["dataset_dir"][plat][user]):
        """

        :param dataset_name:
        :param graph_dir:
        :param graph_format:
        :param top_k:
        """
        super(Neighborhood_Graph,self).__init__()
        self.graph_dir = graph_dir
        self.dataset_name = dataset_name
        self.graph_format = graph_format
        self.top_k = top_k
        # self.load_doc_neighborhood_graph()

    def get_adj_matrix(self,G: nx.classes.graph.Graph = None,default_load: str = 'val',adj_format: str = "coo"):
        """ Returns the adjacency matrix in [adj_format].

        :param default_load:
        :param G: Networkx Graph
        :param adj_format: scipy sparse format name, e.g. {‘bsr’, ‘csr’, ‘csc’, ‘coo’, ‘lil’, ‘dia’, ‘dok’}.
        """
        if G is None:
            G = self.load_doc_neighborhood_graph(get_stats=False)
        return nx.to_scipy_sparse_matrix(G,format=adj_format)

    def invert_classes_dict(self,input_dict=None):
        """ Generates a new dict with categories to document ids map.

        :param input_dict:
        :return:
        """
        if input_dict is None: input_dict = self.sample2cats
        inverted_dict = OrderedDict()
        for doc,cats in input_dict.items():
            for cat in cats:
                inverted_dict[cat].append(doc)

        return inverted_dict

    def prepare_label_graph(self,cat2docs_map=None):
        """ Generates a dict of categories mapped to document and then creates the label neighborhood graph.

        :param cat2docs_map:
        """
        if cat2docs_map is None:
            cat2docs_map = self.invert_classes_dict(self.sample2cats)

        G_cats = self.create_neighborhood_graph(cat2docs_map)

        nx.relabel_nodes(G_cats,self.cat_id2text_map,copy=False)
        return G_cats

    def create_neighborhood_graph(self,nodes: list = None,sample2cats_map: dict = None,
                                  min_common=config["graph"]["min_common"]):
        """ Generates the neighborhood graph (of type category or document) as key
        and common items as values.

        Default: Generates the document graph; for label graph call 'prepare_label_graph()'.

        :param nodes: List of node ids to consider, others are ignored.
        :param min_common: Minimum number of common categories between two documents.
        :param sample2cats_map:
        :return:
        """
        if sample2cats_map is None: sample2cats_map = self.sample2cats
        if nodes is None: nodes = list(sample2cats_map.keys())
        G = nx.Graph()
        G.add_nodes_from(nodes)
        logger.debug(nx.info(G))
        for doc1,cats1 in sample2cats_map.items():
            for doc2,cats2 in sample2cats_map.items():
                if doc1 in nodes and doc2 in nodes:  ## Consider samples only within [nodes] list.
                    if doc1 != doc2:
                        cats_common = set(cats1).intersection(set(cats2))
                        if len(cats_common) >= min_common:
                            G.add_edge(doc1,doc2,edge_id=str(doc1) + '-' + str(doc2),common=repr(cats_common))

        return G

    def load_doc_neighborhood_graph(self,nodes,graph_path=None,get_stats: bool = config["graph"]["stats"]):
        """ Loads the graph file if found else creates neighborhood graph.

        :param nodes: List of node ids to consider.
        :param get_stats:
        :param graph_path: Full path to the graphml file.
        :return: Networkx graph, Adjecency matrix, stats related to the graph.
        """

        # if graph_path is None: graph_path = join(self.graph_dir,self.dataset_name,
        #                                          self.dataset_name + "_G_" + str(len(nodes)) + ".graphml")
        # if exists(graph_path):
        #     logger.info("Loading neighborhood graph from [{0}]".format(graph_path))
        #     Docs_G = nx.read_graphml(graph_path)
        # else:
        self.sample2cats = File_Util.load_json(join(self.graph_dir,self.dataset_name,self.dataset_name +
                                                    "_sample2cats"))
        # self.categories = File_Util.load_json(join(self.graph_dir,self.dataset_name,self.dataset_name +
        #                                            "_cats"))
        # self.cat_id2text_map = File_Util.load_json(join(self.graph_dir,self.dataset_name,self.dataset_name +
        #                                                 "_cat_id2text_map"))
        Docs_G = self.create_neighborhood_graph(nodes=nodes)
        logger.debug(nx.info(Docs_G))
            # logger.info("Saving neighborhood graph at [{0}]".format(graph_path))
            # nx.write_graphml(Docs_G,graph_path)
        # Docs_adj = nx.adjacency_matrix(Docs_G)
        if get_stats:
            Docs_G_stats = self.graph_stats(Docs_G)
            File_Util.save_json(Docs_G_stats,filename=self.dataset_name + "_G_stats",overwrite=True,
                                filepath=join(self.graph_dir,self.dataset_name))
            return Docs_G,Docs_G_stats
        return Docs_G

    @staticmethod
    def graph_stats(G):
        """ Generates and returns graph related statistics.

        :param G: Graph in Netwokx format.
        :return: dict
        """
        G_stats = OrderedDict()
        G_stats["info"] = nx.info(G)
        logger.debug("info: [{0}]".format(G_stats["info"]))
        G_stats["degree_sequence"] = sorted([d for n,d in G.degree()],reverse=True)
        # logger.debug("degree_sequence: {0}".format(G_stats["degree_sequence"]))
        G_stats["dmax"] = max(G_stats["degree_sequence"])
        logger.debug("dmax: [{0}]".format(G_stats["dmax"]))
        G_stats["dmin"] = min(G_stats["degree_sequence"])
        logger.debug("dmin: [{0}]".format(G_stats["dmin"]))
        G_stats["node_count"] = nx.number_of_nodes(G)
        # logger.debug("node_count: [{0}]".format(G_stats["node_count"]))
        G_stats["edge_count"] = nx.number_of_edges(G)
        # logger.debug("edge_count: [{0}]".format(G_stats["edge_count"]))
        G_stats["density"] = nx.density(G)
        logger.debug("density: [{0}]".format(G_stats["density"]))
        if nx.is_connected(G):
            G_stats["radius"] = nx.radius(G)
            logger.debug("radius: [{0}]".format(G_stats["radius"]))
            G_stats["diameter"] = nx.diameter(G)
            logger.debug("diameter: [{0}]".format(G_stats["diameter"]))
            G_stats["eccentricity"] = nx.eccentricity(G)
            logger.debug("eccentricity: [{0}]".format(G_stats["eccentricity"]))
            G_stats["center"] = nx.center(G)
            logger.debug("center: [{0}]".format(G_stats["center"]))
            G_stats["periphery"] = nx.periphery(G)
            logger.debug("periphery: [{0}]".format(G_stats["periphery"]))
        else:
            logger.warning("The graph in not connected.")
            G_comps = nx.connected_components(G)
            logger.debug([len(c) for c in sorted(G_comps,key=len,reverse=True)])

        return G_stats

    def find_single_labels(self):
        """ Finds the number of samples with only single label. """
        single_labels = []
        for i,t in enumerate(self.sample2cats):
            if len(t) == 1:
                single_labels.append(i)
        if single_labels:
            logger.debug(len(single_labels),'samples has only single category. These categories will not occur in the'
                                            'co-occurrence graph.')
        return len(single_labels)

    @staticmethod
    def plot_occurance(E,plot_name=config["graph"]["plot_name"],clear=True,log=False):
        """

        :param E:
        :param plot_name:
        :param clear:
        :param log:
        """
        from matplotlib import pyplot as plt

        plt.plot(E)
        plt.xlabel("Documents")
        if log:
            plt.yscale('log')
        plt.ylabel("Category co-occurance")
        plt.title("Documents degree distribution (sorted)")
        plt.savefig(plot_name)
        plt.show()
        if clear:
            plt.cla()

    def get_subgraph(self,V,E,dataset_name,level=config["graph"]["level"],root_node=config["graph"]["root_node"],
                     subgraph_count=config["graph"]["subgraph_count"],ignore_deg=config["graph"]["ignore_deg"]):
        """ Generates a subgraph of [level] hops starting from [root_node] node.

        # total_points: total number of samples.
        # feature_dm: number of features per sample.
        # number_of_labels: total number of categories.
        # X: feature matrix of dimension total_points * feature_dm.
        # sample2cats: list of size total_points. Each element of the list containing categories corresponding to one sample.
        # V: list of all categories (nodes).
        # E: dict of edge tuple(node_1,node_2) -> weight, eg. {(1, 4): 1, (2, 7): 3}.
        """
        # build a unweighted graph of all edges
        g = nx.Graph()
        g.add_edges_from(E.keys())

        # Below section will try to build a smaller subgraph from the actual graph for visualization
        subgraph_lists = []
        for sg in range(subgraph_count):
            if root_node is None:
                # select a random vertex to be the root
                np.random.shuffle(V)
                v = V[0]
            else:
                v = root_node

            # two files to write the graph and label information
            # Remove characters like \, /, <, >, :, *, |, ", ? from file names,
            # windows can not have file name with these characters
            label_info_filepath = 'samples/' + str(dataset_name) + '_Info[{}].txt'.format(
                str(int(v)) + '-' + File_Util.remove_special_chars(self.cat_id2text_map[v]))
            label_graph_filepath = 'samples/' + str(dataset_name) + '_G[{}].graphml'.format(
                str(int(v)) + '-' + File_Util.remove_special_chars(self.cat_id2text_map[v]))
            # label_graph_el = 'samples/'+str(dataset_name)+'_E[{}].el'.format(str(int(v)) + '-'
            # + self.cat_id2text_map[v]).replace(' ','_')

            logger.debug('Label:[' + self.cat_id2text_map[v] + ']')
            label_info_file = open(label_info_filepath,'w')
            label_info_file.write('Label:[' + self.cat_id2text_map[v] + ']' + "\n")

            # build the subgraph using bfs
            bfs_q = Queue()
            bfs_q.put(v)
            bfs_q.put(0)
            node_check = OrderedDict()
            ignored = []

            sub_g = nx.Graph()
            lvl = 0
            while not bfs_q.empty() and lvl <= level:
                v = bfs_q.get()
                if v == 0:
                    lvl += 1
                    bfs_q.put(0)
                    continue
                elif node_check.get(v,True):
                    node_check[v] = False
                    edges = list(g.edges(v))
                    # label_info_file.write('\nNumber of edges: ' + str(len(edges)) + ' for node: '
                    # + self.cat_id2text_map[v] + '[' + str(v) + ']' + '\n')
                    if ignore_deg is not None and len(edges) > ignore_deg:
                        # label_info_file.write('Ignoring: [' + self.cat_id2text_map[v] + '] \t\t\t degree: ['
                        # + str(len(edges)) + ']\n')
                        ignored.append("Ignoring: deg [" + self.cat_id2text_map[v] + "] = [" + str(len(edges)) + "]\n")
                        continue
                    for uv_tuple in edges:
                        edge = tuple(sorted(uv_tuple))
                        sub_g.add_edge(edge[0],edge[1],weight=E[edge])
                        bfs_q.put(uv_tuple[1])
                else:
                    continue

            # relabel the nodes to reflect textual label
            nx.relabel_nodes(sub_g,self.cat_id2text_map,copy=False)
            logger.debug('sub_g: [{0}]'.format(sub_g))

            label_info_file.write(str('\n'))
            # Writing some statistics about the subgraph
            label_info_file.write(str(nx.info(sub_g)) + '\n')
            label_info_file.write('density: ' + str(nx.density(sub_g)) + '\n')
            label_info_file.write('list of the frequency of each degree value [degree_histogram]: ' +
                                  str(nx.degree_histogram(sub_g)) + '\n')
            for nodes in ignored:
                label_info_file.write(str(nodes) + '\n')
            # subg_edgelist = nx.generate_edgelist(sub_g,label_graph_el)
            label_info_file.close()
            nx.write_graphml(sub_g,label_graph_filepath)

            subgraph_lists.append(sub_g)

            logger.info('Sub graph generated at: [{0}]'.format(label_graph_filepath))

            if root_node:
                logger.info("Root node provided, will generate only one graph file.")
                break

        return subgraph_lists


def main():
    """

    :param args:
    :return:
    """
    cls = Neighborhood_Graph()
    # cls.find_single_labels()
    Docs_G,G_docs_stats = cls.load_doc_neighborhood_graph(get_stats=True)
    Adj_docs = cls.get_adj_matrix(Docs_G)
    cls.plot_occurance(list(G_docs_stats["degree_sequence"]))
    logger.info("Adjacency Matrix: [{0}]".format(Adj_docs.shape))

    return


if __name__ == '__main__':
    main()
