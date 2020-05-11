#!/usr/bin/env bash

# First sample
python3 authorship_detect_esn_5cv_compare_py3.py --seed 1 --reservoir-size 1000 --leak-rate 0.01 --transformer wv --keep-w --name "AD ESN WV ASIMOV REDONE, sample 1" --description "Authorship Detection WV 1000 neurons ASIMOV, redone" --output outputs/ --verbose 4 --author ASIMOV
python3 authorship_detect_esn_5cv_compare_py3.py --seed 1 --inverse-dev-test --reservoir-size 1000 --leak-rate 0.01 --transformer wv --keep-w --name "AD ESN WV ASIMOV REDONE INVERSE, sample 1" --description "Authorship Detection WV 1000 neurons ASIMOV, redone, inverse" --output outputs/ --verbose 4 --author ASIMOV

# Second sample
# python3 authorship_detect_esn_5cv_compare_py3.py --seed 2 --reservoir-size 1000 --leak-rate 0.01 --transformer wv --keep-w --name "AD ESN WV ASIMOV REDONE, sample 2" --description "Authorship Detection WV 1000 neurons ASIMOV, redone" --output outputs/ --verbose 4 --author ASIMOV
# python3 authorship_detect_esn_5cv_compare_py3.py --seed 2 --inverse-dev-test --reservoir-size 1000 --leak-rate 0.01 --transformer wv --keep-w --name "AD ESN WV ASIMOV REDONE INVERSE, sample 2" --description "Authorship Detection WV 1000 neurons ASIMOV, redone, inverse" --output outputs/ --verbose 4 --author ASIMOV
