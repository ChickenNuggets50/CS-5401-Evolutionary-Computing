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
from weasel import mutate_individual

import random

random.seed(42)
ind = initialize_individual("EC is fun", 10)
assert mutate_individual(ind, 0.3) == {"genome": "Eriis BfG", "fitness": 0}
assert mutate_individual(ind, 0.3) == {"genome": "EC is auB", "fitness": 0}
assert mutate_individual(ind, 0.3) == {"genome": "EW gs cun", "fitness": 0}
assert mutate_individual(ind, 0.3) == {"genome": "EC is fuo", "fitness": 0}
