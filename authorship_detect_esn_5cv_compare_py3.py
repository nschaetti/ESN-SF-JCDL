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
import echotorch.utils.matrix_generation
from tools import argument_parsing, dataset, functions, features
import matplotlib.pyplot as plt
import os
import torchlanguage
import sys


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


####################################################
# Main function
####################################################

# Parse args
args, use_cuda, param_space, xp = argument_parsing.parser_esn_training()

# Load from directory
if args.inverse_dev_test:
    sfgram_dataset, sfgram_loader_train, sfgram_loader_test, sfgram_loader_dev = dataset.load_dataset(
        args.author,
        args.transformer[0][0][0],
        remove_authors=args.remove_authors
    )
else:
    sfgram_dataset, sfgram_loader_train, sfgram_loader_dev, sfgram_loader_test = dataset.load_dataset(
        args.author,
        args.transformer[0][0][0],
        remove_authors=args.remove_authors
    )
# end if

# Print authors
xp.write(u"Author : {}".format(sfgram_dataset.author), log_level=0)
xp.write(u"Texts : {}".format(len(sfgram_dataset.texts)), log_level=0)

# W index
w_index = 0

# Initialise random number generator
torch.random.manual_seed(args.seed)
np.random.seed(args.seed)

# Last space
last_space = dict()

# Iterate
for space in param_space:
    # Params
    reservoir_size, w_sparsity, leak_rate, input_scaling, \
    input_sparsity, spectral_radius, feature, aggregation, \
    state_gram, feedbacks_sparsity, lang, embedding, \
    ridge_param, washout = functions.get_params(space)

    # Choose the right transformer
    sfgram_dataset.transform = features.create_transformer(feature, embedding, args.embedding_path, lang)

    # Set experience state
    xp.set_state(space)

    # Average sample
    average_sample = np.array([])

    # For each sample
    for n in range(args.n_samples):
        # Set sample
        xp.set_sample_state(n)

        # Internal matrix
        w_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
            connectivity=1.0 - w_sparsity,
            spetral_radius=spectral_radius
        )

        # Input weights
        win_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
            connectivity=1.0 - input_sparsity,
            scale=input_scaling,
            apply_spectral_radius=False,
        )

        # Bias vector
        wbias_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
            connectivity=1.0,
            scale=0,
            apply_spectral_radius=False
        )

        # ESN cell
        esn = etnn.LiESN(
            input_dim=sfgram_dataset.transform.input_dim,
            output_dim=1,
            hidden_dim=reservoir_size,
            leaky_rate=leak_rate,
            ridge_param=ridge_param,
            w_generator=w_generator,
            win_generator=win_generator,
            wbias_generator=wbias_generator
        )
        if use_cuda:
            esn.cuda()
        # end if

        # For each batch
        for k in range(5):
            # Choose fold
            xp.set_fold_state(k)
            sfgram_loader_train.dataset.set_fold(k)
            sfgram_loader_test.dataset.set_fold(k)

            # Choose the right transformer
            transformer = features.create_transformer(
                feature,
                embedding,
                args.embedding_path,
                lang
            )

            # Set transformer
            sfgram_dataset.transform = transformer

            # For each folds
            for i, data in enumerate(sfgram_loader_train):
                # Inputs and labels
                inputs, labels = data

                # To variable
                inputs, labels = Variable(inputs), Variable(labels)
                if use_cuda: inputs, labels = inputs.cuda(), labels.cuda()

                # Accumulate xTx and xTy
                esn(inputs, labels)
            # end for

            # Train
            esn.finalize()

            # Truth vector
            truth_dev_tensor = torch.zeros(len(sfgram_loader_dev))
            n_threshold = 200
            thresholds = torch.linspace(0.0, 12.0, n_threshold)
            prediction_dev_tensor = torch.zeros(n_threshold, len(sfgram_loader_dev))
            f1_dev_scores = torch.zeros(n_threshold)

            # For each folds
            for i, data in enumerate(sfgram_loader_dev):
                # Inputs and labels
                inputs, labels = data

                # To variable
                inputs, labels = Variable(inputs), Variable(labels)
                if use_cuda: inputs, labels = inputs.cuda(), labels.cuda()

                # Predict
                y_predicted = esn(inputs)

                # Between 0 and 1
                y_predicted -= 0.03
                y_predicted /= 0.10

                # Truth tensor
                if torch.sum(labels) > 0.0:
                    truth_dev_tensor[i] = 1.0
                # end if

                # For each threshold
                for j, threshold in enumerate(thresholds):
                    # Above threshold => 1.0
                    predicted_labels = y_predicted >= threshold

                    # If thereis interest point
                    if torch.sum(predicted_labels) > 0.0:
                        # Set as detected
                        prediction_dev_tensor[j, i] = 1.0
                    # end if
                # end for
            # end for

            # For each threshold
            threshold_to_score = list()
            for j, threshold in enumerate(thresholds):
                # Above threshold => 1.0
                predicted_labels = prediction_dev_tensor[j] == 1.0

                try:
                    tp_fp = float(torch.sum(predicted_labels))
                    tp_fn = float(torch.sum(truth_dev_tensor))
                    tp = float(torch.sum(truth_dev_tensor[predicted_labels]))

                    # Precision and recall
                    precision = tp / tp_fp
                    recall = tp / tp_fn

                    # Compute F1
                    f1_dev_scores[j] = 2.0 * ((precision * recall) / (precision + recall))
                    threshold_to_score.append((threshold, f1_dev_scores[j]))
                except ZeroDivisionError:
                    f1_dev_scores[j] = 0.0
                    threshold_to_score.append((threshold, 0.0))
                # end try
            # end for

            # Best f1 score
            best_dev_f1_score = torch.max(f1_dev_scores)
            best_dev_threshold = thresholds[torch.argmax(f1_dev_scores)]
            threshold_to_score = sorted(threshold_to_score, key=lambda x: x[0])
            # print(threshold_to_score)
            print(u"Best validation threshold : {} with {}".format(best_dev_threshold, best_dev_f1_score))

            # Truth vector
            truth_test_tensor = torch.zeros(len(sfgram_loader_test))
            prediction_test_tensor = torch.zeros((n_threshold, len(sfgram_loader_test)))
            f1_test_scores = torch.zeros(n_threshold)

            # For each folds
            for i, data in enumerate(sfgram_loader_test):
                # print(u"Test: {}".format(i))
                # Inputs and labels
                inputs, labels = data

                # To variable
                inputs, labels = Variable(inputs), Variable(labels)
                if use_cuda: inputs, labels = inputs.cuda(), labels.cuda()

                # Predict
                y_predicted = esn(inputs)

                # Between 0 and 1
                y_predicted -= 0.03
                y_predicted /= 0.10
                # print(u"Min : {}".format(torch.min(y_predicted)))
                # print(u"Max : {}".format(torch.max(y_predicted)))
                # Truth tensor
                if torch.sum(labels) > 0.0:
                    truth_test_tensor[i] = 1.0
                # end if

                # For each threshold
                for j, threshold in enumerate(thresholds):
                    # Above threshold => 1.0
                    predicted_labels = y_predicted >= threshold

                    # If thereis interest point
                    if torch.sum(predicted_labels) > 0.0:
                        # Set as detected
                        prediction_test_tensor[j, i] = 1.0
                    # end if
                # end for
            # end for

            # For each threshold
            threshold_to_score = list()
            for j, threshold in enumerate(thresholds):
                # Above threshold => 1.0
                predicted_labels = prediction_test_tensor[j] == 1.0

                try:
                    tp_fp = float(torch.sum(predicted_labels))
                    tp_fn = float(torch.sum(truth_test_tensor))
                    tp = float(torch.sum(truth_test_tensor[predicted_labels]))

                    # Precision and recall
                    precision = tp / tp_fp
                    recall = tp / tp_fn

                    # Compute F1
                    f1_test_scores[j] = 2.0 * ((precision * recall) / (precision + recall))
                    threshold_to_score.append((threshold, f1_test_scores[j]))
                except ZeroDivisionError:
                    f1_test_scores[j] = 0.0
                    threshold_to_score.append((threshold, 0.0))
                # end try
            # end for

            # Best f1 score
            best_test_f1_score = torch.max(f1_test_scores)
            best_test_threshold = thresholds[torch.argmax(f1_test_scores)]
            threshold_to_score = sorted(threshold_to_score, key=lambda x: x[0])
            # print(threshold_to_score)
            print("Best test threshold : {} with {}".format(best_test_threshold, best_test_f1_score))
            print("Best test F1 with dev threshold : {}".format(threshold_to_score[best_dev_threshold]))

            # Save result
            xp.add_result(threshold_to_score[best_dev_threshold])
        # end for

        # Delete classifier
        del esn

        # W index
        w_index += 1
    # end for samples

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()
