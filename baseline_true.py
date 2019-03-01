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
import torch.utils.data
from torch.autograd import Variable
import echotorch.nn as etnn
import echotorch.utils
from tools import argument_parsing, dataset, functions, features
import matplotlib.pyplot as plt
import nsNLP
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

####################################################
# Main
####################################################


# Argument builder
args = nsNLP.tools.ArgumentBuilder(desc=u"Argument test")

# Dataset arguments
args.add_argument(command="--k", name="k", type=int, help="K-Fold Cross Validation", extended=False, default=10)
args.add_argument(command="--author", name="author", type=str, help="Author to test", extended=False, default=None)

# Experiment output parameters
args.add_argument(command="--name", name="name", type=str, help="Experiment's name", extended=False, required=True)
args.add_argument(command="--description", name="description", type=str, help="Experiment's description",
                  extended=False, required=True)
args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                  extended=False)
args.add_argument(command="--verbose", name="verbose", type=int, help="Verbose level", default=2, extended=False)

# Parse arguments
args.parse()

# Load from directory
sfgram_dataset, sfgram_loader_train, sfgram_loader_test = dataset.load_dataset(args.author, '')

# Experiment
xp = nsNLP.tools.ResultManager\
(
    args.output,
    args.name,
    args.description,
    args.get_space(),
    1,
    args.k,
    verbose=args.verbose
)

# Average
average_k_fold = np.array([])

# Print authors
xp.write(u"Author : {}".format(sfgram_dataset.author), log_level=0)
xp.write(u"Texts : {}".format(len(sfgram_dataset.texts)), log_level=0)

# For each batch
for k in range(10):
    # Choose fold
    xp.set_fold_state(k)
    sfgram_loader_train.dataset.set_fold(k)
    sfgram_loader_test.dataset.set_fold(k)

    # Choose the right transformer
    sfgram_dataset.transform = None

    # Prediction for each threshold
    truth_vector = np.zeros((len(sfgram_loader_test)), dtype=np.int32)
    prediction_vector = np.ones((len(sfgram_loader_test)), dtype=np.int32)

    # Get test data for this fold
    for i, data in enumerate(sfgram_loader_test):
        # Sample
        inputs, label = data

        # Present or not
        for j in label:
            if int(j) == 1:
                truth_vector[i] = 1
            # end if
        # end for
    # end for

    try:
        # F1 score
        tp_fp = float(np.sum(prediction_vector))
        tp_fn = float(np.sum(truth_vector))
        mask = prediction_vector == 1
        tp = float(np.sum(truth_vector[mask]))

        # Precision and recall
        precision = tp / tp_fp
        recall = tp / tp_fn

        # Compute F1
        f1_score = 2.0 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f1_score = 0.0
    # end try

    # Print success rate
    xp.add_result(f1_score)
# end for

xp.save()