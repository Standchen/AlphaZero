#!/bin/sh
python3 -u ./training.py 2>&1 | tee -a log.txt
