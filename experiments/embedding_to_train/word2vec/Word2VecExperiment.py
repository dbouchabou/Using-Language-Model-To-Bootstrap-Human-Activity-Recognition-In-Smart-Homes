# coding: utf-8
# !/usr/bin/env python3

import os
import time
import numpy as np

from tqdm import tqdm

from tensorflow.keras.preprocessing.text import text_to_word_sequence

from SmartHomeHARLib.utils import Experiment
from SmartHomeHARLib.datasets.casas import Encoder
from SmartHomeHARLib.embedding import Word2VecEventEmbedder


class Word2VecExperiment(Experiment):

    def __init__(self, dataset, experiment_parameters):
        super().__init__(dataset, experiment_parameters)

        self.experiment_tag = "Dataset_{}_Encoding_{}_WindowsSize_{}_EmbeddingSize_{}_NbEpochs_{}".format(
            self.dataset.name, 
            self.experiment_parameters["encoding"],
            self.experiment_parameters["window_size"],
            self.experiment_parameters["embedding_size"],
            self.experiment_parameters["epoch_number"]
        )

        # Embedding
        self.w2v_dataset_encoder = Encoder(self.dataset)
        self.w2v_data_train = []
        self.w2v_model = None


    def prepare_data_for_w2v(self):

        sentence_tmp = []
        for row in self.w2v_dataset_encoder.X:
            sentence_tmp.append(" ".join(row))

        sentence = " ".join(sentence_tmp)

        tokens = [text_to_word_sequence(sentence, lower = False, filters = '')]

        self.w2v_data_train = tokens


    def prepare_dataset(self):

        with tqdm(total=2, desc='Prepare Dataset') as pbar:

            self.w2v_dataset_encoder.basic_raw()
            pbar.update(1)

            self.prepare_data_for_w2v()
            pbar.update(1)


    def start(self):

        # Star time of the experiment
        current_time = time.strftime("%Y_%m_%d_%H_%M_%S")

        self.experiment_result_path = os.path.join(self.experiment_parameters["name"], 
                                                    self.experiment_parameters["model_type"],
                                                    self.dataset.name,
                                                    "run_" + "_" + str(current_time) + "_" + self.experiment_tag
        )

        # create a folder with the model name
        # if the folder doesn't exist
        if not os.path.exists(self.experiment_result_path):
            os.makedirs(self.experiment_result_path)

        self.prepare_dataset()

        self.w2v_model = Word2VecEventEmbedder(self.w2v_data_train,
                                                embedding_size = self.experiment_parameters["embedding_size"], 
                                                window_size = self.experiment_parameters["window_size"], 
                                                workers_number = self.experiment_parameters["workers_number"],
                                                nb_epoch = self.experiment_parameters["epoch_number"]
        )
        
        print("Start Training...")
        
        self.w2v_model.train()
        
        print("Training Finish")

        w2v_name = "w2v_{}.emb".format(self.dataset.name)
        w2v_model_path = os.path.join(self.experiment_result_path, w2v_name)
        self.w2v_model.save_model(w2v_model_path)