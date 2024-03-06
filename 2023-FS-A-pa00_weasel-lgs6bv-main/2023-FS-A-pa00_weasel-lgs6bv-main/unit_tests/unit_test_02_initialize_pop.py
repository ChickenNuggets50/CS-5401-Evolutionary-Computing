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
from weasel import initialize_pop

import random

random.seed(42)
assert initialize_pop("EC is easy", 2) == [
    {"genome": "OhbVrpoiVg", "fitness": 0},
    {"genome": "RVIfLBcbfn", "fitness": 0},
]
assert initialize_pop("EC is easy", 2) == [
    {"genome": "oGMbJmTPSI", "fitness": 0},
    {"genome": "AoCLrZaWZk", "fitness": 0},
]
assert initialize_pop("EC is easy", 2) == [
    {"genome": "SBvrjnWvgf", "fitness": 0},
    {"genome": "ygwwMqZcUD", "fitness": 0},
]
assert initialize_pop("EC is easy", 2) == [
    {"genome": "IhyfJsONxK", "fitness": 0},
    {"genome": "mTecQoXsfo", "fitness": 0},
]
