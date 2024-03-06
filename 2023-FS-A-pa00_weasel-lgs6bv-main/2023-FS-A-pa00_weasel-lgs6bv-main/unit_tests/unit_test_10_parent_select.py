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
from weasel import parent_select

import random

random.seed(42)
i1 = initialize_individual("gene", 5)
i2 = initialize_individual("meme", 6)
i3 = initialize_individual("heme", 4)
pop = [i1, i2, i3]
assert parent_select(pop, 2) == [
    {"genome": "meme", "fitness": 6},
    {"genome": "gene", "fitness": 5},
]
assert parent_select(pop, 2) == [
    {"genome": "gene", "fitness": 5},
    {"genome": "gene", "fitness": 5},
]
assert parent_select(pop, 2) == [
    {"genome": "heme", "fitness": 4},
    {"genome": "meme", "fitness": 6},
]
assert parent_select(pop, 2) == [
    {"genome": "heme", "fitness": 4},
    {"genome": "gene", "fitness": 5},
]
