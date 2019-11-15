#!/bin/bash

set -e

make cleanall

python3 ./set.py

radmc3d mctherm setthreads 16

python2 ./plot.py

python3 ./read_fits.py


