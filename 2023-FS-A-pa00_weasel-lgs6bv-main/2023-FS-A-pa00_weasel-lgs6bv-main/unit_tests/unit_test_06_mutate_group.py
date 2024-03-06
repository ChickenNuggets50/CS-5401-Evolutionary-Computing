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
from weasel import mutate_group

import random

random.seed(42)
i1 = initialize_individual("EC is great", 10)
i2 = initialize_individual("Great is EC", 10)
pop = [i1, i2]
assert mutate_group(pop, 0.3) == [
    {"genome": "Eriis BfGat", "fitness": 0},
    {"genome": "Greaa BsWEg", "fitness": 0},
]
assert mutate_group(pop, 0.3) == [
    {"genome": "ECcis great", "fitness": 0},
    {"genome": "Groat is Ew", "fitness": 0},
]
assert mutate_group(pop, 0.3) == [
    {"genome": "rC OUDgreat", "fitness": 0},
    {"genome": "GrcatKis EC", "fitness": 0},
]
assert mutate_group(pop, 0.3) == [
    {"genome": "Ep is greFd", "fitness": 0},
    {"genome": "Gkeay iJ UI", "fitness": 0},
]
