#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cs231n
spyder &
nohup jupyter notebook --browser firefox &> /dev/null
