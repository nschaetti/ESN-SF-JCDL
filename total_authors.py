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
import json


parser = argparse.ArgumentParser()
parser.add_argument("--magazines", type=str)
parser.add_argument("--sfgram", type=str)
args = parser.parse_args()

# Load JSON magazines and sfgram
magazine_data = json.load(open(args.magazines))
sfgram_data = json.load(open(args.sfgram))

# Author count
author_count = 0.0

# For each magazine
for m in magazine_data.keys():
    # Find magazine in sfgram
    for b in sfgram_data['books']:
        if b['title'] == m:
            author_count += len(b['authors'])
        # end if
    # end for
# end for

print(u"There is {} authors".format(author_count))
