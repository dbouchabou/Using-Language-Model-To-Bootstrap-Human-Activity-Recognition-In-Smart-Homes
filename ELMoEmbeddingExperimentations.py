#!/usr/bin/env python3

import argparse


import tensorflow as tf
import numpy as np

from SmartHomeHARLib.datasets.casas import Dataset

from SmartHomeHARLib.datasets.casas import Aruba
from SmartHomeHARLib.datasets.casas import Milan
from SmartHomeHARLib.datasets.casas import Cairo

from experiments.embedding_to_train.elmo.ELMoExperiment import ELMoExperiment


SEED = 7

# fix the random seed for tensorflow
tf.random.set_seed(SEED)
# fix the random seed for numpy
np.random.seed(SEED)

if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    strategy = tf.distribute.MirroredStrategy()

    # set and parse the arguments list
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
    else:
        dataset = Dataset(data, "../../datasets/original_datasets/CASAS/{}/data".format(data))

    print(dataset.name)


    ELMo_experiment_parameters= {
        "name" : "pretrain_embedding",
        "encoding" : "basic_raw",
        "model_type" : "ELMo",
        "embedding_size" : 64,
        "window_size" : 20,
        "epoch_number" : 400,
        "batch_size":512
    }

    with strategy.scope():

        exp = ELMoExperiment(dataset, ELMo_experiment_parameters)

        # exp.setDebugMode(True)
        # print(exp.DEBUG)
        exp.start()

    # save experiment config
    exp.save_config()
