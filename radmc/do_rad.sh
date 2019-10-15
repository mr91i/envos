#!/bin/bash

set -e

make cleanall

./set.py

radmc3d mctherm setthreads 4

./plot.py




