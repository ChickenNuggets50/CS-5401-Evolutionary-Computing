#!/bin/bash
# DO NOT EDIT THIS FILE!

rm results.txt
timeout 1200 python3 top_evolve.py
test -f results.txt && grade=$(cat results.txt) || grade=0
# if test -f results.txt
# then
#     grade=$(cat results.txt)
# else
#     grade=0
# fi
mypy --strict --disallow-any-explicit ./*.py && ((grade = grade + 5))
black --check ./*.py && ((grade = grade + 5))
echo "$grade" >results.txt
echo "Your base grade is: $grade"
if ((grade > 70)); then
    echo "You're passing."
    echo "Use tuning, control, and meta-optimization to find better solutions."
    echo "If you have questions about how grading happens, just look at this script."
    exit 0
else
    exit 1
fi
