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
import argparse


# Args
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str)
parser.add_argument("--magazine", type=str)
parser.add_argument("--author", type=str)
args = parser.parse_args()

# For each file
for f in os.listdir(args.dir):
    file_path = os.path.join(args.dir, f)
    file_name = f[:-2]
    if f[-2:] == ".p":
        if args.magazine in f:
            os.rename(file_path, os.path.join(args.dir, file_name + "." + args.author + ".p"))
            print(u"From {} to {}".format(file_path, os.path.join(args.dir, file_name + "." + args.author + ".p")))
        # end if
    # end if
# end for
