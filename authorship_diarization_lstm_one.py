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

import nsNLP
import numpy as np
from tools.functions import create_tokenizer, create_converter
import os
import matplotlib.pyplot as plt
import math
from data.DiarizationDataset import DiarizationDataset


ALPHABET = u""


####################################################
# Functions
####################################################


# Converter in
def converter_in(converters_desc, converter):
    """
    Is the converter in the desc
    :param converters_desc:
    :param converter:
    :return:
    """
    for converter_desc in converters_desc:
        if converter in converter_desc:
            return True
        # end if
    # end for
    return False
# end converter_in


# Create directory
def create_directories(output_directory, xp_name):
    """
    Create image directory
    :return:
    """
    # Directories
    image_directory = os.path.join(output_directory, xp_name, "images")
    texts_directory = os.path.join(output_directory, xp_name, "texts")

    # Create if does not exists
    if not os.path.exists(image_directory):
        os.mkdir(image_directory)
    # end if

    # Create if does not exists
    if not os.path.exists(texts_directory):
        os.mkdir(texts_directory)
    # end if

    return image_directory, texts_directory
# end create_directories


####################################################
# Main function
####################################################

# Main function
if __name__ == "__main__":
    # Argument builder
    args = nsNLP.tools.ArgumentBuilder(desc=u"Argument test")

    # Dataset arguments
    args.add_argument(command="--dataset", name="dataset", type=str,
                      help="JSON file with the file description for each authors", required=True, extended=False)

    # Author
    args.add_argument(command="--author", name="author1", type=str, help="First author to learn",
                      required=True, extended=False)

    # LSTM arguments
    args.add_argument(command="--hidden-size", name="hidden_size", type=int, default=100, extended=True)
    args.add_argument(command="--embedding-dim", name="embedding_dim", type=int, default=300, extended=True)
    args.add_argument(command="--learning-rate", name="learning_rate", type=float, default=0.1, extended=True)
    args.add_argument(command="--converters", name="converters", type=str,
                      help="The text converters to use (fw, pos, tag, wv, trained)", default='oh', extended=True)
    args.add_argument(command="--pca-path", name="pca_path", type=str, help="PCA model to load", default=None,
                      extended=False)
    args.add_argument(command="--aggregation", name="aggregation", type=str, help="Output aggregation method",
                      extended=True, default="average")

    # Tokenizer and clustering parameters
    args.add_argument(command="--tokenizer", name="tokenizer", type=str,
                      help="Which tokenizer to use (spacy, nltk, spacy-tokens)", default='nltk', extended=False)
    args.add_argument(command="--lang", name="lang", type=str, help="Tokenizer language parameters", default='en',
                      extended=False)

    # Experiment output parameters
    args.add_argument(command="--name", name="name", type=str, help="Experiment's name", extended=False, required=True)
    args.add_argument(command="--description", name="description", type=str, help="Experiment's description",
                      extended=False, required=True)
    args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                      extended=False)
    args.add_argument(command="--eval", name="eval", type=str, help="Evaluation measure", default='bsquared',
                      extended=False)
    args.add_argument(command="--n-samples", name="n_samples", type=int, help="Number of different reservoir to test",
                      default=1, extended=False)
    args.add_argument(command="--verbose", name="verbose", type=int, help="Verbose level", default=2, extended=False)

    # Parse arguments
    args.parse()

    # Corpus
    sfgram = DiarizationDataset(source_directory=args.dataset, author=args.author)

    # Parameter space
    param_space = nsNLP.tools.ParameterSpace(args.get_space())

    # Experiment
    xp = nsNLP.tools.ResultManager\
    (
        args.output,
        args.name,
        args.description,
        args.get_space(),
        args.n_samples,
        1,
        verbose=args.verbose
    )

    # Create image directory
    image_directory, texts_directory = create_directories(args.output, args.name)

    # Print authors
    xp.write(u"Authors : {}".format(sfgram.authors, log_level=0))
    xp.write(u"Texts : {}".format(len(sfgram.files), log_level=0))

    # Last space
    last_space = dict()

    # Iterate
    for space in param_space:
        # Params
        hidden_size = space['hidden_size']
        embedding_dim = space['embedding_dim']
        learning_rate = space['learning_rate']
        converter_desc = space['converters']
        aggregation = space['aggregation'][0][0]

        # Choose the right tokenizer
        if converter_in(converter_desc, "wv") or \
                converter_in(converter_desc, "pos") or \
                converter_in(converter_desc, "tag"):
            tokenizer = create_tokenizer("spacy_wv")
        else:
            tokenizer = create_tokenizer("nltk")
        # end if

        # Set experience state
        xp.set_state(space)

        # Average sample
        average_sample = np.array([])

        # For each sample
        for n in range(args.n_samples):
            # Set sample
            xp.set_sample_state(n)

            # Description
            desc_info = u"{}-{}".format(space, n)

            # Create LSTM text analyser
            classifier = nsNLP.lstm_models.LSTMTextAnalyser\
            (
                classes=sfgram.authors_tags,
                hidden_size=hidden_size,
                converter=create_converter(converter_desc, args.pca_path, 0, False, ALPHABET),
                embedding_dim=embedding_dim,
                learning_rate=learning_rate,
                aggregation=aggregation
            )

            # For each folds
            for (training_set, training_y, test_set, test_y) in sfgram:
                # For each magazine in training set
                for index, text in enumerate(training_set):
                    # Add
                    classifier.train(tokenizer(text), training_y[index])
                # end for

                # Train
                classifier.finalize(verbose=False)

                # Success and total
                success = 0.0
                total = 0.0

                # For each magazine in test set
                for index, text in enumerate(test_set):
                    # Targets
                    y = test_y[index]

                    # Classify the text
                    predicted_author, prediction_probs = classifier.predict(tokenizer(text))

                    # Output predictions
                    outputs = classifier.outputs

                    # Image
                    for index2, output in enumerate(outputs):
                        if output == y[index2]:
                            success += 1.0
                        # end if
                        total += 1.0
                    # end for
                # end for

                # Save result
                xp.add_result(success / total)
            # end for

            # Delete classifier
            del classifier
        # end for samples

        # Last space
        last_space = space
    # end for

    # Save experiment results
    xp.save()
# end if
