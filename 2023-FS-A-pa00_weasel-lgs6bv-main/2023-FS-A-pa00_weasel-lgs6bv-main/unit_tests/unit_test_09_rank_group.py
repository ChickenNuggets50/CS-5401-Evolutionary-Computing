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
from weasel import rank_group

import random

random.seed(42)
i1 = initialize_individual("Zhis glass it isabeesting!", 2)
i2 = initialize_individual("This class is motivating!!", 6)
objective = "This class is captivating!"
pop = [i1, i2]
rank_group(pop)
assert pop == [
    {"genome": "This class is motivating!!", "fitness": 6},
    {"genome": "Zhis glass it isabeesting!", "fitness": 2},
]
