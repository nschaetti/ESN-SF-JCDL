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

# Imports
import spacy
from nltk.tokenize import word_tokenize
import json
import os
import math
import random
import codecs
import re


# Object to read data set
class DiarizationDataset(object):
    """
    Object to read a diarization data set
    """

    # Properties
    _source_directory = u""
    _authors = {}
    _magazines = {}
    _files = []
    _converter = "spacy"
    _k = 10
    _n_samples = 0
    _pos = 0
    _id2name = {}
    _name2id = {}
    _int2id = {}
    _id2int = {}
    _tokenizer = None
    _labels = {}
    _authors_tags = [u"NONE"]

    # Constructor
    def __init__(self, source_directory, converter="spacy", k=10, shuffle=False, spacy_model="en_core_web_lg",
                 author=None):
        """
        Constructor
        :param source_directory:
        """
        # Properties
        self._source_directory = source_directory
        self._authors_file = os.path.join(source_directory, "authors.json")
        self._magazines_file = os.path.join(source_directory, "magazines.json")
        self._k = 10
        self._author = author

        # Tokenizer
        if converter == "spacy":
            self._tokenizer = spacy.load(spacy_model)
        # end if

        # Load data set
        self._load()

        # Shuffle
        if shuffle:
            self._files = random.shuffle(self._files)
        # end if

        # Get folds
        self._folds = self._compute_folds(k)
    # end __init__

    #####################################
    # Properties
    #####################################

    # Authors
    @property
    def authors(self):
        """
        Authors
        :return:
        """
        return self._authors
    # end authors

    # Magazines
    @property
    def magazines(self):
        """
        Magazines
        :return:
        """
        return self._magazines
    # end magazines

    # Files
    @property
    def files(self):
        """
        Files
        :return:
        """
        return self._files
    # end files

    # Folds
    @property
    def folds(self):
        """
        Folds
        :return:
        """
        return self._folds
    # end folds

    # Authors tags
    @property
    def authors_tags(self):
        """
        Authors tags
        :return:
        """
        return self._authors_tags
    # end authors_tags

    #####################################
    # Private
    #####################################

    # ID to name
    def _id_to_name(self, aid):
        """
        Get name from ID
        :return:
        """
        return self._id2name[aid]
    # end _id_to_name

    # ID to int
    def _id_to_int(self, aid):
        """
        Get int from ID
        :param aid:
        :return:
        """
        return self._id2int[aid]
    # end _id_to_int

    # Compute folds
    def _compute_folds(self, k=10):
        """
        Compute folds
        :param k:
        :return:
        """
        # Folds
        folds_size = []
        folds = []

        # Divider and reste
        divider = int(math.floor(self._n_samples / k))
        reste = self._n_samples % k
        non_reste = k - reste

        # Smaller
        folds_size += [divider]*non_reste
        folds_size += [divider+1]*reste

        # Folds
        pos = 0
        for i in range(k):
            folds.append(self._files[pos: pos+folds_size[i]])
            pos += folds_size[i]
        # end for

        return folds
    # end _folds

    # Load dataset
    def _load(self):
        """
        Load dataset
        :return:
        """
        # Load authors and magazines
        self._authors = json.load(open(self._authors_file, 'r'))
        self._magazines = json.load(open(self._magazines_file, 'r'))

        # Author's info
        author_id = 1
        for author_name in self._authors.keys():
            author = self._authors[author_name]
            self._id2name[author['id']] = author_name
            self._id2int[author['id']] = author_id
            self._name2id[author_name] = author['id']
            self._int2id[author_id] = author['id']
            self._authors_tags.append(author['id'])
        # end for

        # Get each file path
        for file_path in os.listdir(self._source_directory):
            if u".txt" in file_path:
                sample_path = os.path.join(self._source_directory, file_path)
                self._files.append(sample_path)
                self._labels[sample_path] = self._get_y(codecs.open(sample_path, 'r', encoding='utf-8').read())
            # end if
        # end for

        # Samples
        self._n_samples = len(self._files)
    # end _load

    # Get tokens
    def _get_tokens(self, text):
        """
        Get tokens
        :param text:
        :return:
        """
        if self._converter == "spacy":
            tokens = list()
            for token in self._tokenizer(text):
                tokens.append(token.text)
            # end for
            return tokens
        else:
            return word_tokenize(text)
        # end if
    # end _get_tokens

    # Get text's targets
    def _get_y(self, text):
        """
        Get text's targets
        :param text:
        :return:
        """
        # Tokens
        tokens = self._get_tokens(text)
        y = list()

        # State
        author_id = u"NONE"
        in_author_state = False

        # For each tokens
        for token in tokens:
            # In author state
            if u"SFGRAM_START_" in token:
                if in_author_state:
                    print(u"Error, already in author state!")
                    exit()
                # end if
                in_author_state = True
                # author_id = self._id_to_int(token[13:])
                author_id = token[13:]
            elif u"SFGRAM_STOP_" in token:
                in_author_state = False
                author_id = u"NONE"
            else:
                # New entry
                if self._author is None or author_id == self._author:
                    y.append(author_id)
                else:
                    y.append(u"NONE")
                # end if
            # end if
        # end for

        return y
    # end _get_y

    # Remove tags
    def _remove_tags(self, text):
        """
        Remove tags
        :param text:
        :return:
        """
        # Regexes
        tag_regex = [u"SFGRAM_START_[A-Z]+", u"SFGRAM_STOP_[A-Z]+"]

        # Delete tags
        for regex in tag_regex:
            text = re.sub(regex, u'', text)
        # end for

        return text
    # end _remove_tags

    ######################################
    # Override
    ######################################

    # Iterator
    def __iter__(self):
        """
        Iterator
        :return:
        """
        return self
    # end __iter__

    # Next
    def next(self):
        """
        Next
        :return:
        """
        # Stop
        if self._pos >= self._k:
            self._pos = 0
            raise StopIteration()
        # end if

        # Training set
        training_set = list(self._files)

        # Test set
        test_set = list(self._folds[self._pos])

        # Remove test set
        for text_file in test_set:
            training_set.remove(text_file)
        # end for

        # Text sets
        training_texts = list()
        test_texts = list()

        # Ouput sets
        training_ys = list()
        test_ys = list()

        # Load training set
        for text_file in training_set:
            t = codecs.open(text_file, 'r', encoding='utf-8').read()
            training_texts.append(self._remove_tags(t))
            training_ys.append(self._labels[text_file])
        # end for

        # Load test set
        for text_file in test_set:
            t = codecs.open(text_file, 'r', encoding='utf-8').read()
            test_texts.append(self._remove_tags(t))
            test_ys.append(self._labels[text_file])
        # end for

        # Inc
        self._pos += 1

        return training_texts, training_ys, test_texts, test_ys
    # end next

# end DiarizationDataset
