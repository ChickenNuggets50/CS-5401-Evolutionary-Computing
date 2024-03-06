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
from weasel import recombine_group

import random

random.seed(42)
i1 = initialize_individual("EC is great", 10)
i2 = initialize_individual("Great is EC", 10)
pop = [i1, i2]
assert recombine_group(pop, 0.8) == [
    {"genome": "Great is EC", "fitness": 0},
    {"genome": "EC is great", "fitness": 0},
]
assert recombine_group(pop, 0.8) == [
    {"genome": "EC at is EC", "fitness": 0},
    {"genome": "Greis great", "fitness": 0},
]
assert recombine_group(pop, 0.8) == [
    {"genome": "Ereat is EC", "fitness": 0},
    {"genome": "GC is great", "fitness": 0},
]
assert recombine_group(pop, 0.8) == [
    {"genome": "EC is gr EC", "fitness": 0},
    {"genome": "Great iseat", "fitness": 0},
]
