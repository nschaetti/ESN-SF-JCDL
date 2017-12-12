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
import codecs
from tools.cleaning import remove_underline, remove_line_breaks, remove_multiple_line, remove_pagination, remove_magazine_title


####################################################
# Functions
####################################################


####################################################
# Main function
####################################################

# Main function
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description=u"ESN-SF-JCDL - Create author diarization dataset")

    # Argument
    parser.add_argument("--dataset", type=str, help="Input directory")
    args = parser.parse_args()

    # Get text list
    for filename in os.listdir(args.dataset):
        # File path
        file_path = os.path.join(args.dataset, filename)

        # Log
        print(u"Cleaning {}".format(file_path))

        # Open file for reading
        input_file = codecs.open(file_path, 'r', encoding='utf-8')

        # Clean text
        document_text = input_file.read()
        document_text = remove_underline(document_text)
        document_text = remove_line_breaks(document_text)
        document_text = remove_pagination(document_text)
        document_text = remove_magazine_title(document_text)

        # Remove double space
        for i in range(50):
            pass
            document_text = remove_multiple_line(document_text)
            document_text = document_text.replace(u"  ", u" ")
        # end for

        # Save
        output_file = codecs.open(file_path, 'w', encoding='utf-8')
        output_file.write(document_text)
    # end for



# end if
