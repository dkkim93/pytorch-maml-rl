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
python3.6 test.py \
--config log-minigrid-empty/config.json \
--policy log-minigrid-empty/policy.th \
--output log-minigrid-empty/results.npz \
--meta-batch-size 1
