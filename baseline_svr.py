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
from sklearn import ensemble
from sklearn.svm import LinearSVR

####################################################
# Main
####################################################


# Argument builder
args = nsNLP.tools.ArgumentBuilder(desc=u"Argument test")

# Dataset arguments
args.add_argument(command="--k", name="k", type=int, help="K-Fold Cross Validation", extended=False, default=10)
args.add_argument(command="--ngram", name="ngram", type=int, help="Ngram", extended=False, default=1)
args.add_argument(command="--analyzer", name="analyzer", type=str, help="word, char, char_wb", extended=False, default='word')
args.add_argument(command="--mfw", name="mfw", type=int, help="mfw", extended=False, default=None)
args.add_argument(command="--author", name="author", type=str, help="Author to test", extended=False, default=None)
args.add_argument(command="--nestimators", name="nestimators", type=int, help="Nb. estimators", extended=False, default=10)

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

    # Samples and classes
    samples = list()
    classes = list()

    # Count vector
    count_vec = CountVectorizer(ngram_range=(1, args.ngram))

    # TF-IDF transformer
    tf_transformer = TfidfTransformer(use_idf=True)

    # Classifier
    classifier = LinearSVR(random_state=42)

    # Pipleline
    text_clf = Pipeline([('vec', count_vec),
                         ('tfidf', tf_transformer),
                         ('clf', classifier)])

    # Choose the right transformer
    sfgram_dataset.transform = None

    # Get training data for this fold
    for i, data in enumerate(sfgram_loader_train):
        # Sample
        inputs, label = data

        # Present or not
        author_label = 0.0
        for j in label:
            if int(j) == 1:
                author_label = 1.0
            # end if
        # end for

        # Clean inputs
        inputs = inputs[0][0]
        inputs = inputs.replace(u"SFGRAM_START_ASIMOV", u"")
        inputs = inputs.replace(u"SFGRAM_STOP_ASIMOV", u"")
        inputs = inputs.replace(u"SFGRAM_START_DICK", u"")
        inputs = inputs.replace(u"SFGRAM_STOP_DICK", u"")
        inputs = inputs.replace(u"SFGRAM_START_SILVERBERG", u"")
        inputs = inputs.replace(u"SFGRAM_STOP_SILVERBERG", u"")
        inputs = inputs.lower()
        inputs = inputs.replace(u"isaac", u"")
        inputs = inputs.replace(u"asimov", u"")
        inputs = inputs.replace(u"philip", u"")
        inputs = inputs.replace(u"dick", u"")
        inputs = inputs.replace(u"robert", u"")
        inputs = inputs.replace(u"silverberg", u"")

        # Add
        samples.append(inputs)
        classes.append(author_label)
    # end for

    # Train
    text_clf.fit(samples, classes)

    # Prediction for each threshold
    n_threshold = 100
    thresholds = torch.linspace(0.0, 1.0, n_threshold)
    truth_vector = np.zeros((len(sfgram_loader_test)), dtype=np.int32)
    prediction_vector = np.zeros((n_threshold, len(sfgram_loader_test)), dtype=np.int32)
    f1_scores = torch.zeros(n_threshold)

    # Get test data for this fold
    for i, data in enumerate(sfgram_loader_test):
        # Sample
        inputs, label = data

        # Present or not
        author_label = u"F"
        for j in label:
            if int(j) == 1:
                author_label = u"T"
                truth_vector[i] = 1
            # end if
        # end for

        # Clean inputs
        inputs = inputs[0][0]
        inputs = inputs.replace(u"SFGRAM_START_ASIMOV", u"")
        inputs = inputs.replace(u"SFGRAM_STOP_ASIMOV", u"")
        inputs = inputs.replace(u"SFGRAM_START_DICK", u"")
        inputs = inputs.replace(u"SFGRAM_STOP_DICK", u"")
        inputs = inputs.replace(u"SFGRAM_START_SILVERBERG", u"")
        inputs = inputs.replace(u"SFGRAM_STOP_SILVERBERG", u"")
        inputs = inputs.lower()
        inputs = inputs.replace(u"isaac", u"")
        inputs = inputs.replace(u"asimov", u"")
        inputs = inputs.replace(u"philip", u"")
        inputs = inputs.replace(u"dick", u"")
        inputs = inputs.replace(u"robert", u"")
        inputs = inputs.replace(u"silverberg", u"")

        # Predict
        prediction = text_clf.predict([inputs])[0]

        # For each threshold
        for j, threshold in enumerate(thresholds):
            # Above threshold
            if prediction >= threshold:
                # Set as detected
                prediction_vector[j, i] = 1.0
            # end if
        # end for
    # end for

    # For each threshold
    for j, threshold in enumerate(thresholds):
        try:
            # F1 score
            tp_fp = float(np.sum(prediction_vector[j, :]))
            tp_fn = float(np.sum(truth_vector))
            mask = prediction_vector[j, :] == 1
            tp = float(np.sum(truth_vector[mask]))

            # Precision and recall
            precision = tp / tp_fp
            recall = tp / tp_fn

            # Compute F1
            f1_scores[j] = 2.0 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f1_scores[j] = 0.0
        # end try
    # end for

    # Best f1 score
    best_f1_score = torch.max(f1_scores)
    best_threshold = thresholds[torch.argmax(f1_scores)]

    # Print success rate
    xp.add_result(best_f1_score)
# end for

xp.save()