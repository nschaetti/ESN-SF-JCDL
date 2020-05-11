#!/usr/bin/env bash

# First sample
python3 authorship_detect_esn_5cv_compare_py3.py --seed 1 --reservoir-size 1000 --leak-rate 0.001 --transformer c3 --keep-w --name "AD ESN C3 ASIMOV REDONE, sample 1" --description "Authorship Detection C3 1000 neurons ASIMOV, redone" --output outputs/ --embedding-path /home/schaetti/Projets/TURING/Recherches/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c3_cx2_d60.p --verbose 4 --author ASIMOV
python3 authorship_detect_esn_5cv_compare_py3.py --seed 1 --inverse-dev-test --reservoir-size 1000 --leak-rate 0.001 --transformer c3 --keep-w --name "AD ESN C3 ASIMOV REDONE INVERSE, sample 1" --description "Authorship Detection C3 1000 neurons ASIMOV, redone, inverse" --output outputs/ --embedding-path /home/schaetti/Projets/TURING/Recherches/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c3_cx2_d60.p --verbose 4 --author ASIMOV
