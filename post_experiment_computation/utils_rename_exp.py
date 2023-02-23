from dataclasses import dataclass
from utils import DataComputer, walk_runs
import os

@dataclass
class RunComputer(DataComputer):
    run_offset: int = 20
    test: bool = True

    def load_run(self, run_path, run_name):
        file_path = os.path.join(run_path, run_name)
        run_nr = int(run_name.removeprefix("exp_"))
        new_run_name = f"exp_{run_nr + self.run_offset}"

        if not self.test:
            os.rename(file_path, os.path.join(run_path, new_run_name))

        message = f"renamed '{run_name}' to '{new_run_name}'!"
        return run_path, message


    def comp_sub_exp(self, data):
        return data[1]
    
    def comp_exp(self, loaded_data):
        return loaded_data


dc = RunComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/learn_brown_exp_var_stdsel",
    sub_exp_folder="learn_brown_exp_var_stdsel",
    test=False

)
eval_res = dc.load_exp()
print(eval_res)