# coding: utf-8
# !/usr/bin/env python3


import numpy as np
import pandas as pd
import umap.plot
import json

from SmartHomeHARLib.datasets.casas import Dataset
from SmartHomeHARLib.datasets.casas import Encoder
from SmartHomeHARLib.datasets.casas import Segmentator
from SmartHomeHARLib.embedding import ELMoEventEmbedder

import tensorflow as tf

data = "aruba"
window_size = 20
embedding_size = 64

filename = "model_4_basic_raw_{}_{}_{}_no_backward".format(data,window_size,embedding_size)


dataset = Dataset(data,  "../../datasets/original_datasets/CASAS/{}/data".format(data))


encodedDataset = Encoder(dataset)
#encodedDataset.eventEmbeddingRaw5()
encodedDataset.basic_raw()
#encodedDataset.basic_raw_time()

X = encodedDataset.X

#print(X)

sentence_tmp = []
#for row in X:
#    sentence_tmp.append(" ".join(row))

#sentence = " ".join(sentence_tmp)
sentence = " ".join(X)

print(sentence[0:100])


elmo_model = ELMoEventEmbedder(sentences = [sentence],
                               embedding_size = embedding_size, 
                               window_size = window_size,
                               nb_epoch = 400, 
                               batch_size = 512, 
                               verbose = True, 
                               residual = True,
                               step = 1
                              )


elmo_model.tokenize()

vocabulary = elmo_model.vocabulary

path = filename+"_elmo_dict_vocabulary.json"
with open(path,"w") as save_vocab_file:
        json.dump(vocabulary, save_vocab_file, indent = 4)

#elmo_model.prepare()
elmo_model.prepare_4()
print(elmo_model.forward_inputs.shape)
print(elmo_model.forward_outputs.shape)

#input("Press Enter to continue...")

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

with strategy.scope():

    elmo_model.compile()
    elmo_model.train(filename+"_elmo_model.h5")