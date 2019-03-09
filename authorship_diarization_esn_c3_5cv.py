#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPTextClassifier.py
# Description : Echo State Network for text classification.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing NLP Project.
# The Reservoir Computing Memory Project is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import torch
from torch.autograd import Variable
import echotorch.nn as etnn
from tools import argument_parsing, dataset, functions, features
import math
import matplotlib.pyplot as plt
import copy
from scipy import signal
import os


####################################################
# Main function
####################################################


# Compute f1 score
def compute_f1_score(tp, fp, fn):
    """
    Compute f1 score
    :param tp:
    :param fp:
    :param fn:
    :return:
    """
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2.0 * ((precision * recall) / (precision + recall))
# end compute_f1_score


# Compute F1
def compute_average_f1(confusion_matrix):
    """
    Compute F1
    :param confusion_matrix:
    :return:
    """
    pos_f1_score = compute_f1_score(confusion_matrix[1, 1], confusion_matrix[0, 1], confusion_matrix[1, 0])
    return pos_f1_score
# end compute_f1


# Compute accuracy
def compute_accuracy(confusion_matrix):
    """
    Compute accuracy
    :param confusion_matrix:
    :return:
    """
    return confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
# end compute_accuracy


# Create ESM models
def create_esn_models(is_cuda):
    """
    Create ESN models.
    :return:
    """
    # ESN cell with WV
    esn_c3 = etnn.LiESN(
        input_dim=60,
        hidden_dim=reservoir_size,
        output_dim=1,
        spectral_radius=spectral_radius,
        sparsity=input_sparsity,
        input_scaling=input_scaling,
        w_sparsity=w_sparsity,
        learning_algo='inv',
        leaky_rate=0.001,
        feedbacks=args.feedbacks,
        seed=1 if args.keep_w else None
    )
    if is_cuda:
        esn_c3.cuda()
    # end if

    return esn_c3
# end create_esn_models


####################################################
# Main function
####################################################

# Parse args
args, use_cuda, param_space, xp = argument_parsing.parser_esn_training()

# Load from directory
sfgram_dataset_wv, sfgram_loader_train_wv, sfgram_loader_dev_wv, sfgram_loader_test_wv = dataset.load_dataset(args.author, 'wv', remove_authors=args.remove_authors)
sfgram_dataset_c3, sfgram_loader_train_c3, sfgram_loader_dev_c3, sfgram_loader_test_c3 = dataset.load_dataset(args.author, 'c3', remove_authors=args.remove_authors)

# Print authors
xp.write(u"Author : {}".format(sfgram_dataset_wv.author), log_level=0)
xp.write(u"Texts : {}".format(len(sfgram_dataset_wv.texts)), log_level=0)

# W index
w_index = 0

# Last space
last_space = dict()

# Threshold
n_threshold = 200

# Iterate
for space in param_space:
    # Params
    reservoir_size, w_sparsity, leak_rate, input_scaling, \
    input_sparsity, spectral_radius, feature, aggregation, \
    state_gram, feedbacks_sparsity, lang, embedding, \
    ridge_param, washout = functions.get_params(space)

    # Set experience state
    xp.set_state(space)

    # Average sample
    average_sample = np.array([])

    # For each sample
    for n in range(args.n_samples):
        # Set sample
        xp.set_sample_state(n)

        # ESN cell
        esn_c3 = create_esn_models(use_cuda)

        # For each batch
        for k in range(5):
            # Set fold state
            xp.set_fold_state(k)

            # Set fold for WV
            sfgram_loader_train_wv.dataset.set_fold(k)
            sfgram_loader_dev_wv.dataset.set_fold(k)
            sfgram_loader_test_wv.dataset.set_fold(k)

            # Set fold for C3
            sfgram_loader_train_c3.dataset.set_fold(k)
            sfgram_loader_dev_c3.dataset.set_fold(k)
            sfgram_loader_test_c3.dataset.set_fold(k)

            # F1 per threshold
            f1_dev_scores = torch.zeros(n_threshold)
            thresholds = torch.linspace(0.0, 4.0, n_threshold)

            # Choose the right transformer
            sfgram_dataset_wv.transform = features.create_transformer('wv', embedding, args.embedding_path, lang)
            sfgram_dataset_c3.transform = features.create_transformer('c3', embedding, args.embedding_path, lang)

            # Train C3 models
            for i, data in enumerate(sfgram_loader_train_c3):
                # Inputs and labels
                inputs, labels = data

                # To variable
                inputs, labels = Variable(inputs), Variable(labels)
                if use_cuda: inputs, labels = inputs.cuda(), labels.cuda()

                # Accumulate xTx and xTy
                esn_c3(inputs, labels)
            # end for

            # Train
            esn_c3.finalize()

            # WV ouputs
            wv_lengths = dict()
            labels_list = list()

            # For each folds (WV)
            for i, data in enumerate(sfgram_loader_dev_wv):
                # Inputs and labels
                inputs, labels = data

                # Append
                labels_list.append(labels)

                # Length
                wv_lengths[i] = inputs.size(1)

                # Add to y and ^y
                if i == 0:
                    total_dev_labels = labels
                else:
                    total_dev_labels = torch.cat((total_dev_labels, labels), dim=1)
                # end if
            # end for

            # For each folds (C3)
            for i, data in enumerate(sfgram_loader_dev_c3):
                # Inputs and labels
                inputs, labels = data

                # To variable
                inputs, labels = Variable(inputs), Variable(labels)
                if use_cuda: inputs, labels = inputs.cuda(), labels.cuda()

                # Predict
                y_predicted = esn_c3(inputs)
                c3_length = y_predicted.size(1)

                # Downsample character predictions
                y_subsampled = torch.from_numpy(signal.resample(y_predicted.numpy(), wv_lengths[i], axis=1)).type(torch.FloatTensor)

                # Add to y and ^y
                if i == 0:
                    total_dev_predicted = y_subsampled
                else:
                    total_dev_predicted = torch.cat((total_dev_predicted, y_subsampled), dim=1)
                # end if
            # end for

            # Between 0 and 1
            total_dev_predicted -= torch.mean(total_dev_predicted)
            total_dev_predicted /= torch.std(total_dev_predicted)

            # For each threshold
            for j, threshold in enumerate(thresholds):
                # Confusion matrix
                confusion_matrix = torch.zeros((2, 2))

                # Above threshold => 1.0
                predicted_labels = total_dev_predicted >= threshold
                truth_labels = total_dev_labels == 1.0

                try:
                    tp_fp = float(torch.sum(predicted_labels))
                    tp_fn = float(torch.sum(truth_labels))
                    tp = float(torch.sum(total_dev_labels[predicted_labels]))

                    # Precision and recall
                    precision = tp / tp_fp
                    recall = tp / tp_fn

                    # Compute F1
                    f1_dev_scores[j] = 2.0 * ((precision * recall) / (precision + recall))
                except ZeroDivisionError:
                    f1_dev_scores[j] = 0.0
                # end try
            # end for

            # Best f1 score
            best_dev_f1_score = torch.max(f1_dev_scores)
            best_dev_threshold = thresholds[torch.argmax(f1_dev_scores)]
            print(u"Best dev threshold : {} with {}".format(best_dev_threshold, best_dev_f1_score))

            # For each folds (WV)
            for i, data in enumerate(sfgram_loader_test_wv):
                # Inputs and labels
                inputs, labels = data

                # Append
                labels_list.append(labels)

                # Length
                wv_lengths[i] = inputs.size(1)

                # Add to y and ^y
                if i == 0:
                    total_test_labels = labels
                else:
                    total_test_labels = torch.cat((total_test_labels, labels), dim=1)
                # end if
            # end for

            # For each folds (C3)
            for i, data in enumerate(sfgram_loader_test_c3):
                # Inputs and labels
                inputs, labels = data

                # To variable
                inputs, labels = Variable(inputs), Variable(labels)
                if use_cuda: inputs, labels = inputs.cuda(), labels.cuda()

                # Predict
                y_predicted = esn_c3(inputs)
                c3_length = y_predicted.size(1)

                # Downsample character predictions
                y_subsampled = torch.from_numpy(signal.resample(y_predicted.numpy(), wv_lengths[i], axis=1)).type(torch.FloatTensor)

                # Plot
                plt.plot(labels[0].numpy(), 'r')
                plt.plot(y_predicted[0].numpy(), 'g')
                plt.title(sfgram_dataset_c3.last_text)
                plt.savefig(os.path.join("images_{}".format(args.author), "plot.c3.{}.{}.jpg".format(k, i)))
                plt.close()

                # Save predictions
                with open(os.path.join("saved_outputs_{}".format(args.author), "predictions.c3.{}.{}.txt".format(k, i)), 'w') as f:
                    y_predicted_to_save = y_predicted - torch.mean(y_predicted)
                    y_predicted_to_save /= torch.std(y_predicted_to_save)
                    # For each timestep
                    for t in range(y_predicted.size(1)):
                        f.write("({}, {})\n".format(t, float(y_predicted_to_save[0, t, 0])))
                    # end for
                # end with

                # Save labels
                with open(os.path.join("saved_outputs_{}".format(args.author), "labels.c3.{}.{}.txt".format(k, i)), 'w') as f:
                    # For each timestep
                    for t in range(labels.size(1)):
                        f.write("({}, {})\n".format(t, float(labels[0, t, 0])))
                    # end for
                # end with

                # Add to y and ^y
                if i == 0:
                    total_test_predicted = y_subsampled
                else:
                    total_test_predicted = torch.cat((total_test_predicted, y_subsampled), dim=1)
                # end if
            # end for

            # Between 0 and 1
            total_test_predicted -= torch.mean(total_test_predicted)
            total_test_predicted /= torch.std(total_test_predicted)

            # Confusion matrix
            confusion_matrix = torch.zeros((2, 2))

            # Above threshold => 1.0
            predicted_labels = total_test_predicted >= best_dev_threshold
            truth_labels = total_test_labels == 1.0

            try:
                tp_fp = float(torch.sum(predicted_labels))
                tp_fn = float(torch.sum(truth_labels))
                tp = float(torch.sum(total_test_labels[predicted_labels]))

                # Precision and recall
                precision = tp / tp_fp
                recall = tp / tp_fn

                # Compute F1
                f1_test_score = 2.0 * ((precision * recall) / (precision + recall))
            except ZeroDivisionError:
                f1_test_score = 0.0
            # end try

            # Save result
            xp.add_result(f1_test_score)
        # end for

        # Delete classifier
        del esn_c3

        # W index
        w_index += 1
    # end for samples

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()
