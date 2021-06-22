# coding: utf-8
# !/usr/bin/env python3

import os
import numpy as np

from progress.bar import *

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.activations import *

from .w2v_pre_train_base_experiments import Word2VecPreTrainBaseExperiment


class Word2vecLSTMExperiment(Word2VecPreTrainBaseExperiment):

    def __init__(self, dataset, experiment_parameters):
        super().__init__(dataset, experiment_parameters)


    def build_model_classifier(self, run_number=0):

        nb_timesteps = self.experiment_parameters["sequence_lenght"]
        nb_classes = len(self.dataset.activitiesList)
        emb_dim = self.w2v_model.embedding_size
        vocab_size = len(self.classifier_dataset_encoder.eventDict)
        vocab = list(self.classifier_dataset_encoder.eventDict.keys())
        output_dim = self.experiment_parameters["nb_units"]

        # build the model

        # create embedding

        embedding_weight = self.get_embedding_matrix(vocab)

        embedding_layer = Embedding(input_dim = vocab_size+1, output_dim = emb_dim, input_length = nb_timesteps, mask_zero = True, weights = [embedding_weight], trainable = False)

        # classifier
        feature_1 = Input(shape=((nb_timesteps,)))
        

        token_emb = embedding_layer (feature_1)

        lstm_1 = LSTM(output_dim)(token_emb)

        output_layer = Dense(nb_classes, activation='softmax')(lstm_1)

        self.classifier_model = Model(inputs=feature_1, outputs=output_layer, name="W2V_LSTM_Classifier")

        # ceate a picture of the model
        picture_name = self.classifier_model.name + "_" + self.experiment_tag + "_" + str(run_number) + ".png"
        picture_path = os.path.join(self.experiment_result_path, picture_name)

        plot_model(self.classifier_model, show_shapes = True, to_file = picture_path)
    

    def check_input_model(self, run_number = 0):

        X_val_input = None
        Y_val_input = None

        if self.DEBUG:
            print(self.classifier_data_X_train.shape)
            print(self.classifier_data_X_test.shape)
            if self.classifier_data_X_val != None:
                print(self.classifier_data_X_val.shape)
            else:
                print("None")
            input("Press Enter to continue...")

        # Check number size of exemples
        if len(self.classifier_data_X_train) < 2:
            data_X_train = self.classifier_data_X_train[0]
            data_Y_train = self.classifier_data_Y_train[0]
        else:
            data_X_train = self.classifier_data_X_train[run_number]
            data_Y_train = self.classifier_data_Y_train[run_number]

        if len(self.classifier_data_X_test) < 2:
            data_X_test = self.classifier_data_X_test[0]
            data_Y_test = self.classifier_data_Y_test[0]
        else:
            data_X_test = self.classifier_data_X_test[run_number]
            data_Y_test = self.classifier_data_Y_test[run_number]

        if self.classifier_data_X_val != None:
            if len(self.classifier_data_X_val) < 2:
                data_X_val = self.classifier_data_X_val[0]
                data_Y_val = self.classifier_data_Y_val[0]
            else:
                data_X_val = self.classifier_data_X_val[run_number]
                data_Y_val = self.classifier_data_Y_val[run_number]

        # Nb features depends on data shape
        if data_X_train.ndim > 2:
            nb_features = data_X_train.shape[2]
        else:
            nb_features = 1

        if self.DEBUG:
            print(len(data_X_train[:, :, 0]))
            print(len(data_X_train[:, :, 1]))
            print(len(data_X_train[:, :, 2]))
            print(data_X_train.shape)

        X_train_input = data_X_train
        X_test_input = data_X_test

        if self.classifier_data_X_val != None:
            X_val_input = data_X_val

        Y_train_input = data_Y_train
        Y_test_input = data_Y_test

        if self.classifier_data_X_val != None:
            Y_val_input = data_Y_val

        if self.DEBUG:
            print("Train {}:".format(np.array(X_train_input).shape))
            print("Test : {}".format(np.array(X_test_input).shape))

            if self.classifier_data_X_val != None:
                print("Val : {}".format(np.array(X_val_input).shape))
            else:
                print("Val : None")

            input("Press Enter to continue...")

        #print("Train {}:".format(np.array(X_train_input).shape))
        #print(X_train_input[0])
        #input("Press Enter to continue...")

        return X_train_input, Y_train_input, X_val_input, Y_val_input, X_test_input, Y_test_input, nb_features
