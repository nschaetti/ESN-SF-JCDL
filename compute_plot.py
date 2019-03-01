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


parser = argparse.ArgumentParser()
parser.add_argument("--labels", type=str)
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--points", type=str)
parser.add_argument("--threshold", type=float)
parser.add_argument("--step", type=int)
args = parser.parse_args()

# Open files
f = open(args.input, 'r')
lines = f.readlines()
o = open(args.output, 'w')
p = open(args.points, 'w')
l = open(args.labels, 'r')

# Timeseries
timeserie = list()
downsampled = list()

# For each line
for line in lines:
    # Data
    data = line[1:-2].split(", ")

    # Timestep and
    timeserie.append((int(data[0]), float(data[1])))
# end for

# For each entry
sum = 0.0
timestep = 0
for i, (t, s) in enumerate(timeserie):
    if i % args.step == 0 and i != 0:
        averaged_value = sum / float(args.step)
        o.write("({}, {})\n".format(timestep, averaged_value))
        downsampled.append((timestep, averaged_value))
        sum = s
        timestep += 1
    else:
        sum += s
    # end if
# end for

# For each entry
sum = 0.0
timestep = 0
for i, (t, s) in enumerate(downsampled):
    if s >= args.threshold:
        p.write("({}, {})\n".format(t, s))
    # end if
# end for

# For each line in labels
lines = l.readlines()
labels = list()
for line in lines:
    # Data
    data = line[1:-2].split(", ")

    # Timestep and
    labels.append((int(data[0]), float(data[1])))
# end for

# For each entry
change = False
last_label = 0.0
timestep = 0
for i, (t, s) in enumerate(labels):
    if i % args.step == 0 and i != 0:
        if change:
            print("change at {}".format(timestep))
        # end if
        timestep += 1
        change = False
    # end if

    # Change
    if s != last_label:
        change = True
    # end if

    last_label = s
# end for

l.close()
f.close()
o.close()
p.close()