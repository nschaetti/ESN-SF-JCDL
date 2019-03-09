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
import matplotlib.pyplot as plt
import os
import torchlanguage


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
sfgram_dataset, sfgram_loader_train, sfgram_loader_test = dataset.load_dataset(args.author, args.transformer[0][0][0], remove_authors=args.remove_authors)

# Print authors
xp.write(u"Author : {}".format(sfgram_dataset.author), log_level=0)
xp.write(u"Texts : {}".format(len(sfgram_dataset.texts)), log_level=0)

# W index
w_index = 0

# Last space
last_space = dict()

# Load novels
novels_dataset = torchlanguage.datasets.FileDirectory(root='./novels', save_transform=True)
novels_loader = torch.utils.data.DataLoader(novels_dataset, batch_size=1, shuffle=False)

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

        # ESN cell
        esn = etnn.LiESN(
            input_dim=sfgram_dataset.transform.input_dim,
            hidden_dim=reservoir_size,
            output_dim=1,
            spectral_radius=spectral_radius,
            sparsity=input_sparsity,
            input_scaling=input_scaling,
            w_sparsity=w_sparsity,
            learning_algo='inv',
            leaky_rate=leak_rate,
            feedbacks=args.feedbacks,
            seed=1 if args.keep_w else None,
            ridge_param=ridge_param
        )
        if use_cuda:
            esn.cuda()
        # end if

        # For each batch
        for k in range(10):
            # Choose fold
            xp.set_fold_state(k)
            sfgram_loader_train.dataset.set_fold(k)
            sfgram_loader_test.dataset.set_fold(k)

            # F1 per threshold
            f1_train_scores = torch.zeros(100)
            f1_scores = torch.zeros(100)
            thresholds = torch.linspace(0.0, 1.0, 100)

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
                print(i)
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

            # Success and total
            success = 0.0
            total = 0.0

            # For each folds
            for i, data in enumerate(sfgram_loader_test):
                # Inputs and labels
                inputs, labels = data

                # To variable
                inputs, labels = Variable(inputs), Variable(labels)
                if use_cuda: inputs, labels = inputs.cuda(), labels.cuda()

                # Predict
                y_predicted = esn(inputs)

                # Plot
                plt.plot(labels[0].numpy(), 'r')
                plt.plot(y_predicted[0].numpy(), 'g')
                plt.title(sfgram_dataset.last_text)
                plt.show()

                # Add to y and ^y
                if i == 0:
                    total_predicted = y_predicted
                    total_labels = labels
                else:
                    total_predicted = torch.cat((total_predicted, y_predicted), dim=1)
                    total_labels = torch.cat((total_labels, labels), dim=1)
                # end if

                # Total
                total += 1.0
            # end for

            # Outputs stats
            print(u"Min : {}".format(torch.min(total_predicted)))
            print(u"Max : {}".format(torch.max(total_predicted)))
            print(u"Mean : {}".format(torch.mean(total_predicted)))
            print(u"Std : {}".format(torch.std(total_predicted)))

            # Save result
            xp.add_result(0)
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
