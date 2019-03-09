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
sfgram_dataset, sfgram_loader_train, sfgram_loader_dev, sfgram_loader_test = dataset.load_dataset(args.author, args.transformer[0][0][0])

# Print authors
xp.write(u"Author : {}".format(sfgram_dataset.author), log_level=0)
xp.write(u"Texts : {}".format(len(sfgram_dataset.texts)), log_level=0)

# W index
w_index = 0

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

    # Choose fold
    xp.set_fold_state(0)
    sfgram_loader_train.dataset.set_fold(0)
    sfgram_loader_test.dataset.set_fold(0)

    # Counts
    author_counts = 0.0
    total_counts = 0.0
    doc_count = 0.0
    total_doc = 0.0

    # For each folds
    for i, data in enumerate(sfgram_loader_train):
        # Inputs and labels
        inputs, labels = data

        # To variable
        inputs, labels = Variable(inputs), Variable(labels)
        if use_cuda: inputs, labels = inputs.cuda(), labels.cuda()

        # Add
        total_counts += int(inputs.size(1))
        author_counts += int(torch.sum(labels))

        # Doc
        if int(torch.sum(labels)) != 0:
            doc_count += 1.0
        # end if

        total_doc += 1.0
    # end for

    # For each folds
    for i, data in enumerate(sfgram_loader_dev):
        # Inputs and labels
        inputs, labels = data

        # To variable
        inputs, labels = Variable(inputs), Variable(labels)
        if use_cuda: inputs, labels = inputs.cuda(), labels.cuda()

        # Add
        total_counts += int(inputs.size(1))
        author_counts += int(torch.sum(labels))

        # Doc
        if int(torch.sum(labels)) != 0:
            doc_count += 1.0
        # end if

        total_doc += 1.0
    # end for

    # For each folds
    for i, data in enumerate(sfgram_loader_test):
        # Inputs and labels
        inputs, labels = data

        # To variable
        inputs, labels = Variable(inputs), Variable(labels)
        if use_cuda: inputs, labels = inputs.cuda(), labels.cuda()

        # Add
        total_counts += int(inputs.size(1))
        author_counts += int(torch.sum(labels))

        # Doc
        if int(torch.sum(labels)) != 0:
            doc_count += 1.0
        # end if

        total_doc += 1.0
    # end for

    # Stats
    print(u"Total docs : {}".format(total_doc))
    print(u"Author doc : {}".format(doc_count))
    print(u"Total words : {}".format(total_counts))
    print(u"Author words : {}".format(author_counts))
    print(u"Ratio of DS : {}".format(author_counts / total_counts))
# end for

# Save experiment results
xp.save()
