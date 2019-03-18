
from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from graph import *
import walker
import codecs 

import warnings
warnings.filterwarnings('ignore')


def main(graph_file_name, path_file_name):
    g = Graph()
    print("Reading...")

    g.read_edgelist(filename=graph_file_name)

    walker_class = walker.BasicWalker(g, workers=8)
    sentences = walker_class.simulate_walks(
            num_walks=10, walk_length=10)
    fw = codecs.open(path_file_name, "w", "utf-8")
    for path in sentences:
        for i in range(0, len(path)):
            if i < len(path) - 1:
                fw.write(str(path[i] + " "))
            else:
                fw.write(str(path[i] + "\n"))
    fw.close()


if __name__ == "__main__":
    ratio_list = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    #ratio_list = [0.95]
    for ratio in ratio_list:
        train_file = './zhihu/train_graph_' + str(ratio) + '.txt'
        target_file = "./zhihu/node_sequences_10_10_train_" + str(ratio) + '.txt'

        main(train_file, target_file)

