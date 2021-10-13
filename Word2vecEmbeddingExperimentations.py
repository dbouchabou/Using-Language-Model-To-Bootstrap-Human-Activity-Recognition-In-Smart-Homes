#!/usr/bin/env python3

import argparse

#import tensorflow as tf
import numpy as np

from SmartHomeHARLib.datasets.casas import Dataset

# Casas datasets
from SmartHomeHARLib.datasets.casas import Aruba
from SmartHomeHARLib.datasets.casas import Milan
from SmartHomeHARLib.datasets.casas import Cairo

# Ordonez datasets
from SmartHomeHARLib.datasets.ordonez import HouseA
from SmartHomeHARLib.datasets.ordonez import HouseB

from experiments.embedding_to_train.word2vec.Word2VecExperiment import Word2VecExperiment

DEBUG_MODE = False
SEED = 7

# Fix the random seed for numpy
np.random.seed(SEED)

if __name__ == '__main__':

    # Set and parse the arguments list
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('--d', dest='data', action='store', default='', help='dataset name')

    args = p.parse_args()
    data = str(args.data)
    

    if data == "aruba":
        dataset = Aruba()
    elif data == "milan":
        dataset = Milan()
    elif data == "cairo":
        dataset = Cairo()
    elif data == "ordonezA":
        dataset = HouseA()
    elif data == "ordonezB":
        dataset = HouseB()
    else:
        print("UNKNOWED DATASET")

    print(dataset.name)

    # Parameters
    word2vec_experiment_parameters= {
        "name" : "pretrain_embedding",
        "encoding" : "basic_raw",
        "model_type" : "Word2Vec",
        "embedding_size" : 64,
        "window_size" : 20,
        "workers_number" : 32,
        "epoch_number" : 100
    }

    exp = Word2VecExperiment(dataset, word2vec_experiment_parameters)

    exp.DEBUG = DEBUG_MODE

    exp.start()

    # Save experiment config
    exp.save_config()
    
