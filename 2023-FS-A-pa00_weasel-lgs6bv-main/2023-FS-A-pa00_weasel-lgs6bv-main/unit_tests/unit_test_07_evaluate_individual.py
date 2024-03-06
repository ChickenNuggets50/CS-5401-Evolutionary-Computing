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
from weasel import evaluate_individual

i1 = initialize_individual("This assignment is hard!", 0)
i2 = initialize_individual("This assignment is good!", 0)
objective = "This assignment is easy!"
evaluate_individual(objective=objective, individual=i1)
evaluate_individual(objective=objective, individual=i2)
assert i1 == {"genome": "This assignment is hard!", "fitness": 21}
assert i2 == {"genome": "This assignment is good!", "fitness": 20}
