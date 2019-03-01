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

import argparse
import os
import spacy
import codecs


parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--position", type=int)
args = parser.parse_args()

f = codecs.open(args.file, 'r', encoding='utf-8')

nlp = spacy.load('en_vectors_web_lg')
doc = nlp(f.read())

positions = range(args.position-20, args.position+1)

for i, token in enumerate(doc):
    if i in positions:
        print(token.text)
    # end if
# end for
