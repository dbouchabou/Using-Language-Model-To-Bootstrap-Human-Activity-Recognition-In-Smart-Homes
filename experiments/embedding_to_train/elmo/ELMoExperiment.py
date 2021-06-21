# coding: utf-8
# !/usr/bin/env python3

import os
import time
import json

from progress.bar import *

from SmartHomeHARLib.tools import Experiment
from SmartHomeHARLib.tools.datasets.casas import Encoder
from SmartHomeHARLib.embedding import ELMoEventEmbedder


class ELMoExperiment(Experiment):

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
        self.elmo_dataset_encoder = Encoder(self.dataset)
        self.elmo_data_train = []
        self.elmo_model = None


    def prepare_data_for_elmo(self):

        X = self.elmo_dataset_encoder.X

        sentence = " ".join(X)

        self.elmo_data_train = [sentence]


    def prepare_dataset(self):

        bar = IncrementalBar('Prepare Dataset', max = 2)

        self.elmo_dataset_encoder.basic_raw()
        bar.next()

        self.prepare_data_for_elmo()
        bar.next()

        bar.finish()


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

        filename_base = "model_4_basic_raw_{}_{}_{}_no_backward".format(self.dataset.name,
                                                                    self.experiment_parameters["window_size"],
                                                                    self.experiment_parameters["embedding_size"]
                                                                    )

        self.prepare_dataset()

        self.elmo_model = ELMoEventEmbedder(sentences = self.w2v_data_train,
                               embedding_size = self.experiment_parameters["embedding_size"], 
                               window_size = self.experiment_parameters["window_size"],
                               nb_epoch = self.experiment_parameters["epoch_number"], 
                               batch_size = self.experiment_parameters["batch_size"], 
                               verbose = True, 
                               residual = True,
                               step = 1
                              )
        
        self.elmo_model.tokenize()

        self.elmo_model.prepare_4()

        self.elmo_model.compile()
        
        print("Start Training...")
        

        model_filename = filename_base+"_elmo_model.h5"
        final_model_path = os.path.join(self.experiment_result_path,model_filename)

        self.elmo_model.train(final_model_path)
        
        print("Training Finish")


        # Save the vocabulary dict
        vocabulary = self.elmo_model.vocabulary

        vocabulary_filename = filename_base+"_elmo_dict_vocabulary.json"
        final_vocabulary_path = os.path.join(self.experiment_result_path,vocabulary_filename)

        with open(final_vocabulary_path,"w") as save_vocab_file:
                json.dump(vocabulary, save_vocab_file, indent = 4)