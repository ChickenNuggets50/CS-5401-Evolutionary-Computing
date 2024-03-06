#!/bin/bash
# Execute this whenever you want to work on the assignment,
# and run grade.sh
# In Windows, copy paste the below command without sudo.

# sudo docker run --interactive --tty --mount type=bind,source="$(pwd)"/,target=/your_code "$(basename "$(pwd)")" fish
sudo docker run --interactive --tty --mount type=bind,source="$(pwd)"/,target=/your_code --name disposable_container weasel fish
# TODO maybe run fish inside of tmux, include various dotfiles like nice vim config too?
