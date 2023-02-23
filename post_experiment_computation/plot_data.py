import os
import numpy as np
from utils import DataComputer

dc = DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/behavior_change",
    sub_exp_folder="behavior_change"
)

eval_res = dc.load_exp()
print(eval_res)