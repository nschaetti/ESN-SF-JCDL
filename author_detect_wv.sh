#!/bin/bash

python authorship_detect_esn_5cv_compare.py --reservoir-size 1000 --leak-rate 0.01 --transformer wv --keep-w --name "AD ESN WV DICK" --description "Authorship detection WV 1000 neurons DICK" --output ./outputs/ --verbose 4 --author DICK

python authorship_detect_esn_5cv_compare.py --reservoir-size 1000 --leak-rate 0.01 --transformer wv --keep-w --name "AD ESN WV ASIMOV" --description "Authorship detection WV 1000 neurons ASIMOV" --output ./outputs/ --verbose 4 --author ASIMOV

python authorship_detect_esn_5cv_compare.py --reservoir-size 1000 --leak-rate 0.01 --transformer wv --keep-w --name "AD ESN WV SILVERBERG" --description "Authorship detection WV 1000 neurons SILVERBERG" --output ./outputs/ --verbose 4 --author SILVERBERG

