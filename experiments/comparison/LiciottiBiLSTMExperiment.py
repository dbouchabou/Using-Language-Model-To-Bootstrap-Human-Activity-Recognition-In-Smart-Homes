# coding: utf-8
# !/usr/bin/env python3

import os
import sys
import csv
import time
import json
import itertools
import pandas as pd
import numpy as np

from progress.bar import *

from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.activations import *
from tensorflow.keras.preprocessing.sequence import pad_sequences


from SmartHomeHARLib.utils import Experiment
from SmartHomeHARLib.utils import Evaluator

from SmartHomeHARLib.datasets.casas import Encoder
from SmartHomeHARLib.datasets.casas import Segmentator


class LiciottiBiLSTMExperiment(Experiment):

    def __init__(self, dataset, experiment_parameters):
        super().__init__(dataset, experiment_parameters)

        self.experiment_tag = "Dataset_{}_Encoding_{}_Segmentation_{}_Batch_{}_Patience_{}_SeqLenght_{}_EmbDim_{}_NbUnits_{}".format(
            self.dataset.name, self.experiment_parameters["encoding"],
            self.experiment_parameters["segmentation"],
            self.experiment_parameters["batch_size"],
            self.experiment_parameters["patience"],
            self.experiment_parameters["sequence_lenght"],
            self.experiment_parameters["emb_dim"],
            self.experiment_parameters["nb_units"]
            
        )

        # General
        self.global_classifier_accuracy = []
        self.global_classifier_balance_accuracy = []
        self.current_time = None

        # Classifier
        self.classifier_dataset_encoder = None
        self.classifier_segmentator = None

        self.classifier_model = None
        self.classifier_best_model_path = None
        self.classifier_data_X = []
        self.classifier_data_Y = []
        self.classifier_data_X_train = []
        self.classifier_data_Y_train = []
        self.classifier_data_X_test = []
        self.classifier_data_Y_test = []
        self.classifier_data_X_val = None
        self.classifier_data_Y_val = None

    
    def encode_dataset_for_classifier(self):
        self.classifier_dataset_encoder = Encoder(self.dataset)
        #self.classifier_dataset_encoder.basic()
        self.classifier_dataset_encoder.basic_raw_encoded()


    def segment_dataset_for_classifier(self):
        self.classifier_segmentator = Segmentator(self.classifier_dataset_encoder)
        self.classifier_segmentator.explicitWindow()


    def prepare_data_for_classifier(self):

        self.classifier_data_X = pad_sequences(self.classifier_segmentator.X, maxlen = self.experiment_parameters["sequence_lenght"], padding = 'post')
        self.classifier_data_Y = self.classifier_segmentator.Y


    def prepare_dataset(self):

        bar = IncrementalBar('Prepare Dataset', max=3)

        self.encode_dataset_for_classifier()
        bar.next()

        self.segment_dataset_for_classifier()
        bar.next()

        self.prepare_data_for_classifier()
        bar.next()

        bar.finish()


    def model_selection(self):

        bar = IncrementalBar('Dataset Spliting', max=2)

        kfold = StratifiedKFold(n_splits=self.experiment_parameters["nb_splits"], shuffle=True, random_state=self.experiment_parameters["seed"])

        k = 0
        for train, test in kfold.split(self.classifier_data_X, self.classifier_data_Y, groups=None):
            self.classifier_data_X_train.append(np.array(self.classifier_data_X)[train])
            self.classifier_data_Y_train.append(np.array(self.classifier_data_Y)[train])


            self.classifier_data_X_test.append(np.array(self.classifier_data_X)[test])
            self.classifier_data_Y_test.append(np.array(self.classifier_data_Y)[test])
        
        bar.next()

        self.classifier_data_X_train = np.array(self.classifier_data_X_train)
        self.classifier_data_Y_train = np.array(self.classifier_data_Y_train)

        self.classifier_data_X_test = np.array(self.classifier_data_X_test)
        self.classifier_data_Y_test = np.array(self.classifier_data_Y_test)
        bar.next()

        if self.DEBUG:
            print("")
            print(self.classifier_data_X_train.shape)
            print(self.classifier_data_Y_train.shape)
            print(self.classifier_data_X_test.shape)
            print(self.classifier_data_Y_test.shape)

            input("Press Enter to continue...")

        bar.finish()


    def build_model_classifier(self, run_number=0):

        nb_timesteps = self.experiment_parameters["sequence_lenght"]
        nb_classes = len(self.dataset.activitiesList)
        emb_dim = self.experiment_parameters["emb_dim"]
        vocab_size = len(self.classifier_dataset_encoder.eventDict)
        vocab = list(self.classifier_dataset_encoder.eventDict.keys())
        output_dim = self.experiment_parameters["nb_units"]

        if self.DEBUG:
            print("")
            print(vocab_size)

            input("Press Enter to continue...")

        # build the model

        # create embedding

        embedding_layer = Embedding(input_dim = vocab_size+1, output_dim = emb_dim, input_length = nb_timesteps, mask_zero = True)

        # classifier
        feature_1 = Input(shape=((nb_timesteps,)))

        embedding = embedding_layer (feature_1)

        lstm_1 = Bidirectional(LSTM(output_dim))(embedding)

        output_layer = Dense(nb_classes, activation='softmax')(lstm_1)

        self.classifier_model = Model(inputs=feature_1, outputs=output_layer, name="liciotti_Bi_LSTM")

        # ceate a picture of the model
        picture_name = self.classifier_model.name + "_" + self.experiment_tag + "_" + str(run_number) + ".png"
        picture_path = os.path.join(self.experiment_result_path, picture_name)

        plot_model(self.classifier_model, show_shapes = True, to_file = picture_path)


    def train(self, X_train_input, Y_train_input, X_val_input, Y_val_input, run_number=0):

        root_logdir = os.path.join(self.experiment_parameters["name"],
                                   "logs_{}_{}".format(self.experiment_parameters["name"], 
                                   self.dataset.name)
        )

        run_id = self.classifier_model.name + "_" + self.experiment_tag + "_" + str(self.current_time) + str(run_number)
        log_dir = os.path.join(root_logdir, run_id)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        best_model_name_saved = self.classifier_model.name + "_" + self.experiment_tag + "_BEST_" + str(run_number) + ".h5"
        self.classifier_best_model_path = os.path.join(self.experiment_result_path, best_model_name_saved)

        csv_name = self.classifier_model.name + "_" + self.experiment_tag + "_" + str(run_number) + ".csv"
        csv_path = os.path.join(self.experiment_result_path, csv_name)

        # create a callback for the tensorboard
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir)

        # callbacks
        csv_logger = CSVLogger(csv_path)

        # simple early stopping
        es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = self.experiment_parameters["patience"])
        mc = ModelCheckpoint(self.classifier_best_model_path, monitor = 'val_sparse_categorical_accuracy', mode = 'max', verbose = 1, save_best_only = True)

        # cbs = [csv_logger,tensorboard_cb,mc,es,cm_callback]
        cbs = [csv_logger, tensorboard_cb, mc, es]

        nb_classes = len(self.dataset.activitiesList)

        self.classifier_model.fit(X_train_input, 
                        Y_train_input, 
                        epochs = self.experiment_parameters["nb_epochs"],
                        batch_size=self.experiment_parameters["batch_size"], 
                        verbose=self.experiment_parameters["verbose"],
                        callbacks=cbs, 
                        validation_split=0.2, 
                        shuffle=True
        )


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
            print(len(data_X_train))
            print(len(data_X_train))
            print(len(data_X_train))
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

        return X_train_input, Y_train_input, X_val_input, Y_val_input, X_test_input, Y_test_input, nb_features


    def compile_model(self):
        self.classifier_model.compile(loss = 'sparse_categorical_crossentropy', 
                                        optimizer = tf.keras.optimizers.Adam(),
                                        metrics = ['sparse_categorical_accuracy']
        )

        # print summary
        print(self.classifier_model.summary())


    def evaluate(self, X_test_input, Y_test_input, run_number=0):

        if self.DEBUG:
            print("")
            print("EVALUATION")
            print(np.array(X_test_input).shape)
            print(np.array(Y_test_input).shape)
            print(self.classifier_best_model_path)
            input("Press Enter to continue...")

        evaluator = Evaluator(X_test_input, Y_test_input, model_path=self.classifier_best_model_path)

        nb_classes = len(self.dataset.activitiesList)

        evaluator.simpleEvaluation(self.experiment_parameters["batch_size"], Y_test_input=Y_test_input)
        self.global_classifier_accuracy.append(evaluator.ascore)

        evaluator.evaluate()

        listActivities = self.dataset.activitiesList
        indexLabels = list(self.classifier_dataset_encoder.actDict.values())
        evaluator.classificationReport(listActivities, indexLabels)
        # print(evaluator.report)

        report_name = self.classifier_model.name + "_repport_" + self.experiment_tag + "_" + str(run_number) + ".csv"
        report_path = os.path.join(self.experiment_result_path, report_name)
        evaluator.saveClassificationReport(report_path)

        evaluator.confusionMatrix()
        # print(evaluator.cm)

        confusion_name = self.classifier_model.name + "_confusion_matrix_" + self.experiment_tag + "_" + str(
            run_number) + ".csv"
        confusion_path = os.path.join(self.experiment_result_path, confusion_name)
        evaluator.saveConfusionMatrix(confusion_path)

        evaluator.balanceAccuracyCompute()
        self.global_classifier_balance_accuracy.append(evaluator.bscore)


    def start(self):

        # Star time of the experiment
        self.current_time = time.strftime("%Y_%m_%d_%H_%M_%S")

        self.experiment_result_path = os.path.join(self.experiment_parameters["name"], self.experiment_parameters["model_type"],
                                           "run_" + self.experiment_tag + "_" + str(self.current_time))

        # create a folder with the model name
        # if the folder doesn't exist
        if not os.path.exists(self.experiment_result_path):
            os.makedirs(self.experiment_result_path)

        self.prepare_dataset()

        # Split the dataset into train, val and test examples
        self.model_selection()

        nb_runs = len(self.classifier_data_X_train)

        if self.DEBUG:
            print("")
            print("NB RUN: {}".format(nb_runs))

        for run_number in range(nb_runs):
            # prepare input according to the model type
            X_train_input, Y_train_input, X_val_input, Y_val_input, X_test_input, Y_test_input, nb_features = self.check_input_model(
                run_number)

            self.build_model_classifier(nb_features)

            # compile the model
            self.compile_model()

            self.train(X_train_input, Y_train_input, X_val_input, Y_val_input, run_number)

            self.evaluate(X_test_input, Y_test_input, run_number)


    def __save_dict_to_json(self, where_to_save, dict_to_save):

        with open(where_to_save, "w") as json_dict_file:
            json.dump(dict_to_save, json_dict_file, indent = 4)


    def save_word_dict(self):

        word_dict_name = "wordDict.json"
        word_dict_path = os.path.join(self.experiment_result_path, word_dict_name)

        self.__save_dict_to_json(word_dict_path, self.classifier_dataset_encoder.eventDict)

    
    def save_activity_dict(self):

        activity_dict_name = "activityDict.json"
        activity_dict_path = os.path.join(self.experiment_result_path, activity_dict_name)

        self.__save_dict_to_json(activity_dict_path, self.classifier_dataset_encoder.actDict)
        

    def save_metrics(self):
        
        csv_name = "cv_scores" + self.classifier_model.name + "_" + self.experiment_tag + "_" + str(self.current_time) + ".csv"
        csv_path = os.path.join(self.experiment_result_path, csv_name)

        with open(csv_path, "w") as output:
            writer = csv.writer(output, lineterminator='\n')

            writer.writerow(["accuracy score :"])
            for val in self.global_classifier_accuracy:
                writer.writerow([val * 100])
            writer.writerow([])
            writer.writerow([np.mean(self.global_classifier_accuracy) * 100])
            writer.writerow([np.std(self.global_classifier_accuracy)])

            writer.writerow([])
            writer.writerow(["balanced accuracy score :"])

            for val2 in self.global_classifier_balance_accuracy:
                writer.writerow([val2 * 100])
            writer.writerow([])
            writer.writerow([np.mean(self.global_classifier_balance_accuracy) * 100])
            writer.writerow([np.std(self.global_classifier_balance_accuracy)])
