#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Virtualenv
virtualenv venv
source venv/bin/activate
pip3.6 install -r requirements.txt

# Comment for using GPU
export CUDA_VISIBLE_DEVICES=-1

# Append PYTHONPATH that points to gym-minigrid
export PYTHONPATH=./../gym-minigrid:$PYTHONPATH

# Begin experiment
python3.6 train.py \
--config configs/minigrid/empty.yaml \
--output-folder log-minigrid-empty
