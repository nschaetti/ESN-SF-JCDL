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


parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str)
args = parser.parse_args()

# Timeseries
timeserie = list()

# For each file
for dir_file in os.listdir(args.dir):
    if "predictions" in dir_file:
        # Open files
        f = open(os.path.join(args.dir, dir_file), 'r')
        lines = f.readlines()

        # For each line
        for line in lines:
            # Data
            data = line[1:-2].split(", ")

            # Timestep and
            timeserie.append((dir_file, int(data[0]), float(data[1])))
        # end for

        # Sort by value
        timeserie.sort(key=lambda tup: tup[2], reverse=True)

        # Close
        f.close()
    # end if
# end for

# Show first 100
for i in range(100):
    print(timeserie[i])
# end for
