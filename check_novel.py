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
from tools.functions import create_tokenizer
import os
import argparse
from tools.cleaning import remove_underline, remove_line_breaks


####################################################
# Functions
####################################################


####################################################
# Main function
####################################################

# Main function
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description=u"ESN-SF-JCDL - Create dataset")

    # Argument
    parser.add_argument("--dataset", type=str, help="Input directory")
    parser.add_argument("--novel", type=str, help="Novel file")
    args = parser.parse_args()

    # Corpus
    sfgram = nsNLP.data.Corpus(args.dataset)

    # Author list
    authors = sfgram.get_authors()

    # Tokenizer
    tokenizer = create_tokenizer("spacy_wv")

    # Get text list
    for author in authors:
        for text in author.get_texts():
            if args.novel in text.get_path():
                #print(tokenizer(text.get_text()))
                for word in tokenizer(text.get_text()):
                    print(word.text)
                # end for
                exit()
            # end if
        # end for
    # end for
# end if
