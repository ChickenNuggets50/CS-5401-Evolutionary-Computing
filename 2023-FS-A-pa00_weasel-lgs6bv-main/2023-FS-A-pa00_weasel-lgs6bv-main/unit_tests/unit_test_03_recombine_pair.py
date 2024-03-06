#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Your code tests start here:
# To debug in pudb3
# Highlight the line of code below below
# Type 't' to jump 'to' it
# Type 's' to 'step' deeper
# Type 'n' to 'next' over
# Type 'f' or 'r' to finish/return a function call and go back to caller
from weasel import initialize_individual
from weasel import recombine_pair

import random

random.seed(42)
i1 = initialize_individual("EC is great", 10)
i2 = initialize_individual("Great is EC", 10)
assert recombine_pair(i1, i2) == [
    {"genome": "EC is greaC", "fitness": 0},
    {"genome": "Great is Et", "fitness": 0},
]
assert recombine_pair(i1, i2) == [
    {"genome": "Ereat is EC", "fitness": 0},
    {"genome": "GC is great", "fitness": 0},
]
assert recombine_pair(i1, i2) == [
    {"genome": "Great is EC", "fitness": 0},
    {"genome": "EC is great", "fitness": 0},
]
assert recombine_pair(i1, i2) == [
    {"genome": "EC it is EC", "fitness": 0},
    {"genome": "Greas great", "fitness": 0},
]
