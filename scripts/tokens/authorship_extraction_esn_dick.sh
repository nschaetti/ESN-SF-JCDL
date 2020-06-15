#!/usr/bin/env bash

python3 authorship_diarization_esn_5cv_compare_py3.py --seed 1 --reservoir-size 1000 --leak-rate 0.01 --transformer wv --keep-w --name "AV ESN WV DICK REDONE samples 1" --description "Authorship Verification WV 1000 neurons DICK, redone, sample 1" --output outputs/ --verbose 4 --author DICK
python3 authorship_diarization_esn_5cv_compare_py3.py --seed 1 --inverse-dev-test --reservoir-size 1000 --leak-rate 0.01 --transformer wv --keep-w --name "AV ESN WV DICK REDONE INVERSE sample 1" --description "Authorship Verification WV 1000 neurons DICK, redone, inverse, sample 1" --output outputs/ --verbose 4 --author DICK

