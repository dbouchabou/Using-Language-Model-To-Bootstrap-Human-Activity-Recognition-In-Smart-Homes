# coding: utf-8
# !/usr/bin/env python3

import os
import time
import json
import numpy as np

from tqdm import tqdm

from SmartHomeHARLib.utils import Experiment
from SmartHomeHARLib.datasets.casas import Encoder
from SmartHomeHARLib.datasets.casas import Segmentator
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
        self.elmo_dataset_segmentator = None
        self.elmo_data_train = []
        self.elmo_model = None


    def prepare_data_for_elmo(self):

        X = self.elmo_dataset_segmentator.X
        
        sentences = []
        
        for activity in X:
            sentences.append(" ".join(activity))

        self.elmo_data_train = sentences

        if self.DEBUG:
            print("")
            print(len(sentences))
            print(sentences[0])

            input("Press Enter to continue...")


    def prepare_dataset(self):

        with tqdm(total=2, desc='Prepare Dataset') as pbar:

            self.elmo_dataset_encoder.basic_raw()
            pbar.update(1)

            self.elmo_dataset_segmentator = Segmentator(self.elmo_dataset_encoder)
            self.elmo_dataset_segmentator.explicitWindow()
            pbar.update(1)

            self.prepare_data_for_elmo()
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

        filename_base = "model_4_basic_raw_{}_{}_{}_backward".format(self.dataset.name,
                                                                    self.experiment_parameters["window_size"],
                                                                    self.experiment_parameters["embedding_size"]
                                                                    )

        self.prepare_dataset()

        self.elmo_model = ELMoEventEmbedder(sentences = self.elmo_data_train,
                               embedding_size = self.experiment_parameters["embedding_size"], 
                               window_size = self.experiment_parameters["window_size"],
                               nb_epoch = self.experiment_parameters["epoch_number"], 
                               batch_size = self.experiment_parameters["batch_size"]
                              )
        
        self.elmo_model.tokenize()

        if self.DEBUG:
            print("")
            print(self.elmo_model.vocabulary)

            input("Press Enter to continue...")

        self.elmo_model.prepare_4()

        if self.DEBUG:
            print("")
            print(self.elmo_model.forward_inputs.shape)
            print(self.elmo_model.backward_inputs.shape)
            print(self.elmo_model.forward_outputs.shape)
            print(self.elmo_model.forward_inputs[0])
            print(self.elmo_model.backward_inputs[0])
            print(self.elmo_model.forward_outputs[0])

            print(self.elmo_model.forward_inputs[1])
            print(self.elmo_model.backward_inputs[1])
            print(self.elmo_model.forward_outputs[1])

            print(np.sort(np.unique(self.elmo_model.forward_outputs)))

            input("Press Enter to continue...")

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