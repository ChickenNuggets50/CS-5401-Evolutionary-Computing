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

assert initialize_individual("EC is fun", 10) == {"genome": "EC is fun", "fitness": 10}
assert initialize_individual("Fun is EC", 9) == {"genome": "Fun is EC", "fitness": 9}
