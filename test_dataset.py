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
# Foobar is distributed in the hope that it will be useful,jjjjj
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
import matplotlib.pyplot as plt
import math
import argparse
from data.DiarizationDataset import DiarizationDataset


####################################################
# Functions
####################################################


####################################################
# Main function
####################################################

# Main function
if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(prog=u"Test data set")
    parser.add_argument("--path", type=str, help="Data set path", required=True)
    args = parser.parse_args()

    # Load
    sfgram = DiarizationDataset(source_directory=args.path)

    # For each folds
    for (training_set, training_y, test_set, test_y) in sfgram:
        print(training_set[0].encode('ascii', errors='ignore'))
        print(training_y[0])
        exit()
    # end for
# end if
