import os
import numpy as np
from utils import plot_pred_vs_gt, plot_rmse_over_mean_meas, DataComputer, save, RunRes
from matplotlib import pyplot as plot

for i in np.arange(1, 5, 1):

    class RunComputer(DataComputer):
    
        def comp_run(self, run_res: RunRes):
            run_res.estimation = run_res.estimation / i
            run_res.gt = run_res.gt / i

            plot_pred_vs_gt(run_res)
            save(path=run_res.run_path,name=f"{run_res.run_name}_pred_vs_gt")
            return super().comp_run(run_res)

    dc = RunComputer(
        base_path=f"/home/bela/Cloud/code/Git/alsbts-experiments/eval/wint_brown_var_wint_sdtsel",
        sub_exp_folder=f"wint_brown_var_wint{i}_sdtsel",
    )
    eval_res = dc.load_exp()

    #plot_rmse_over_mean_meas(eval_res, exp_quant_name=f"intervals_{i}")

    #save(name=f"rmse over nr of measurements win{i}",path=dc.base_path)
#plot.legend()
#save(name="combined rmse over nr of measurements",path=dc.base_path)

# class RunComputer(DataComputer):
    
#     def comp_run(self, run_res: RunRes):
#         run_res.estimation = run_res.estimation / 6
#         return super().comp_run(run_res)

# dc =RunComputer(
#     base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/int_brown_exp_var_stdsel",
#     sub_exp_folder="int_brown_exp_var_stdsel",
# )
# eval_res = dc.load_exp()


# plot_rmse_over_mean_meas(eval_res, exp_quant_name=f"int_brown")
# plot.legend()
# save(name="rmse over nr of measurements",path=dc.base_path)