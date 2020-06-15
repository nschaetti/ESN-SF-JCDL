#!/usr/bin/env bash

# First sample
python3 authorship_diarization_esn_5cv_compare_py3.py --seed 1 --reservoir-size 1000 --leak-rate 0.01 --transformer wv --keep-w --name "AV ESN WV ASIMOV REDONE, sample 1" --description "Authorship Verification WV 1000 neurons ASIMOV, redone" --output outputs/ --verbose 4 --author ASIMOV
python3 authorship_diarization_esn_5cv_compare_py3.py --seed 1 --inverse-dev-test --reservoir-size 1000 --leak-rate 0.01 --transformer wv --keep-w --name "AV ESN WV ASIMOV REDONE INVERSE, sample 1" --description "Authorship Verification WV 1000 neurons ASIMOV, redone, inverse" --output outputs/ --verbose 4 --author ASIMOV

