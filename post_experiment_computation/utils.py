from dataclasses import dataclass
from typing import Any
from typing_extensions import Protocol
from matplotlib import pyplot as plot
from nptyping import NDArray, Shape, Number, Float
import numpy as np
import os


sim_time = 1000

@dataclass
class RunRes():
    run_path: str
    run_name: str

    time: NDArray[Shape["Batch, 1"], Number]
    var: NDArray[Shape["Batch, 1"], Number]

    query: NDArray[Shape["Batch, 1"], Number]
    result: NDArray[Shape["Batch, 1"], Number]

    estimation: NDArray[Shape["Batch, 1"], Number]
    gt: NDArray[Shape["Batch, 1"], Number]

    query_time: NDArray[Shape["Batch, 1"], Number]
    query_var: NDArray[Shape["Batch, 1"], Number]   

def plot_pred_vs_gt(sim_res: RunRes):
    plot.plot(sim_res.time, sim_res.estimation)
    plot.plot(sim_res.time, sim_res.gt)

def calc_measurements(sim_res: RunRes)-> int:
    measurements: int = sim_res.result.shape[0]
    return measurements

def calc_nr_meas(sim_res: RunRes)-> NDArray[Shape["Batch, 1"], Number]:
    measurements = np.arange(1,sim_res.result.shape[0]+1)
    return measurements

def calc_meas_time(sim_res: RunRes) -> NDArray[Shape["Batch, 1"], Number]:
    time = sim_res.query[:,0]
    return time

def calc_rmse(sim_res: RunRes)-> float:
    rmse = np.sqrt(np.sum((sim_res.gt - sim_res.estimation)**2)/sim_res.gt.shape[0])
    return rmse

def calc_mean_std(datas):

    mean = np.mean(datas)
    std = np.std(datas)

    return mean, std

class FileWorker(Protocol):
    def __call__(self, file_path: str, file_name: str) -> Any: ...

def walk_files(path: str, worker: FileWorker):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for f in fnames:
            if f.endswith(".npy"):
                yield worker(file_path = dirpath, file_name = f)

class RunWorker(Protocol):
    def __call__(self, run_path: str, run_name: str) -> Any: ...

def walk_runs(path: str, worker: RunWorker):
    for dirpath, dnames, fnames in os.walk(path):
        d: str
        for d in dnames:
            if d.startswith("exp_"):
                yield worker(run_path = dirpath, run_name = d)

class SubExpWorker(Protocol):
    def __call__(self, sub_exp_path: str, sub_exp_name: str) -> Any: ...

def walk_dirs(path: str, worker: SubExpWorker):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for d in dnames:
                yield worker(sub_exp_path = dirpath, sub_exp_name = d)

@dataclass
class EvalRes():
    exp_quantity: NDArray[Shape["Batch, 1"], Float]
    mean_rmse: NDArray[Shape["Batch, 1"], Float]
    std_rmse: NDArray[Shape["Batch, 1"], Float]
    mean_meas: NDArray[Shape["Batch, 1"], Float]
    std_meas: NDArray[Shape["Batch, 1"], Float]

def norm_time(time):
    return time / sim_time

def plot_rmse_over_exp_quant(eval_res: EvalRes, exp_quant_name = "exp_quant"):
    
    plot.plot(eval_res.exp_quantity, eval_res.mean_rmse, label=f"Mean {exp_quant_name}")
    plot.fill_between(
        eval_res.exp_quantity,
        eval_res.mean_rmse - eval_res.std_rmse*1.96,
        eval_res.mean_rmse + eval_res.std_rmse*1.96,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    plot.xlabel(exp_quant_name)
    plot.ylabel("rmse")

def plot_rmse_over_mean_meas(eval_res: EvalRes, exp_quant_name = "exp_quant"):
    
    plot.plot(eval_res.mean_meas, eval_res.mean_rmse, label=f"Mean {exp_quant_name}")
    plot.fill_between(
        eval_res.mean_meas,
        eval_res.mean_rmse - eval_res.std_rmse*1.96,
        eval_res.mean_rmse + eval_res.std_rmse*1.96,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    plot.xlabel("nr of measurements")
    plot.ylabel("rmse")
    plot.ylim(0, 1)
    plot.xlim(0,1000)

def plot_sum_meas_over_time(meas_res, exp_quant_name = "exp_quant"):

    for exp_quantity, times, mean_measures, std_meas in zip(*meas_res):
    
        plot.plot(times, mean_measures, label=f"Mean for {exp_quant_name}={exp_quantity:.3f}")
        plot.fill_between(
            times,
            mean_measures - std_meas*1.96,
            mean_measures + std_meas*1.96,
            alpha=0.1,
            color="black",
            #label=r"$\pm$ 1 std. dev.",
        )
        plot.xlabel("time")
        plot.ylabel("total acquired measurements")

def plot_meas_over_time(meas_res, exp_quant_name = "exp_quant"):

    for exp_quantity, times, mean_measures, std_meas in zip(*meas_res):
    
        plot.plot(times, mean_measures/times, label=f"Mean for {exp_quant_name}={exp_quantity:.3f}")
        plot.fill_between(
            times,
            (mean_measures - std_meas*1.96)/times,
            (mean_measures + std_meas*1.96)/times,
            alpha=0.1,
            color="black",
            #label=r"$\pm$ 1 std. dev.",
        )
        plot.xlabel("time")
        plot.ylabel("acquired measurements")


def plot_meas_per_step_vs_exp_quant(meas_res, exp_quant_name = "exp_quant"):
    exp_quantities, _,_,_ = meas_res

    mean_meass = []
    mean_stds = []
    for exp_quantity, times, mean_measures, std_meas in zip(*meas_res):
        mean_meass.append(np.median(mean_measures/times))
        mean_stds.append(np.median(std_meas/times))

    exp_quantities = np.asarray(exp_quantities)
    mean_meass = np.asarray(mean_meass)
    mean_stds = np.asarray(mean_stds)
    
    plot.plot(exp_quantities, mean_meass, label=f"Mean for {exp_quant_name}")
    plot.fill_between(
        exp_quantities,
        mean_meass - mean_stds*1.96,
        mean_meass + mean_stds*1.96,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    plot.xlabel(exp_quant_name)
    plot.ylabel("acquired measurements per step")


def save(name, path):
    plot.title(name)
    loc = os.path.join(path,f"{name}.svg")
    plot.savefig(loc, format="svg")
    plot.clf()

@dataclass
class DataComputer:
    sort_index: int = 3
    skip_points: int = 0
    sub_exp_folder: str = "exp_sub_folder"
    base_path: str = "./eval/exp_folder"

    def load_file(self, file_path, file_name):
        file_data = np.load(os.path.join(file_path, file_name))
        return self.comp_file(file_data)

    def comp_file(self, file_data):
        return file_data

    def load_run(self, run_path, run_name):
        file_path = os.path.join(run_path, run_name)
        file_process = "all_data_process.npy"
        file_result = "all_data_result.npy"
        file_stream = "all_data_stream.npy"
        file_estimation = "estimation_data.npy"
        file_gt = "gt_data.npy"

        data_process = self.load_file(file_path, file_process)
        data_result = self.load_file(file_path, file_result)
        data_stream = self.load_file(file_path, file_stream)
        data_estimation = self.load_file(file_path, file_estimation)
        data_gt = self.load_file(file_path, file_gt)

        run_res = RunRes(
            run_path=run_path,
            run_name=run_name,
            time = data_stream[self.skip_points:,:1],
            var = data_stream[self.skip_points:,1:],
            query = data_result[self.skip_points:,:-1],
            result = data_result[self.skip_points:,-1:],
            estimation = data_estimation[self.skip_points:],
            gt = data_gt[self.skip_points:,-1:],
            query_time = data_gt[self.skip_points:,:1],
            query_var= data_gt[self.skip_points:,1:2]
        )
        return self.comp_run(run_res)
    
    def check_run_res_ok(self, run_res: RunRes):
        return not np.any(run_res.estimation == np.nan)

    def comp_run_not_ok(self, run_res: RunRes):
        print("Corrupt Sim Data: ", run_res.run_path)
        print("Min = ", np.min(run_res.estimation))
        run_res.estimation[run_res.estimation < 0] = 0
        run_res.estimation[run_res.estimation < 0] = 0

    def comp_run(self, run_res: RunRes):
        if not self.check_run_res_ok(run_res):
            self.comp_run_not_ok(run_res)

        rmse = calc_rmse(run_res)
        nr_measurements = calc_measurements(run_res)
        return rmse, nr_measurements


    def load_sub_exp(self, sub_exp_path: str, sub_exp_name: str):
        if sub_exp_name.startswith(self.sub_exp_folder):
            exp_quantity_str = sub_exp_name.removeprefix(self.sub_exp_folder)
            if exp_quantity_str:
                exp_quantity = float(exp_quantity_str)
            else:
                exp_quantity = None

            run_data = [run_result for run_result in walk_runs(path=os.path.join(sub_exp_path, sub_exp_name), worker=self.load_run)]

            data = exp_quantity, run_data

            return self.comp_sub_exp(data)

    def comp_sub_exp(self, data):
        exp_quantity, rmse_meas = data

        rmse , meas = np.asarray(rmse_meas).T

        mean_rmse, std_rmse = calc_mean_std(rmse)

        mean_meas, std_meas = calc_mean_std(meas)

        quant_rmse_std_meas_std = (exp_quantity, mean_rmse, std_rmse, mean_meas, std_meas)
        
        return quant_rmse_std_meas_std

    def load_exp(self):
        data = [i_r_v for i_r_v in walk_dirs(path=self.base_path, worker=self.load_sub_exp) if i_r_v is not None]
        return self.comp_exp(data)

    def comp_exp(self, loaded_data):
        inter_rmse_std_meas_std = np.asarray(loaded_data)
        inter_rmse_std_meas_std = inter_rmse_std_meas_std[inter_rmse_std_meas_std[:, self.sort_index].argsort()]
        exp_quantity, mean_rmse, std_rmse, mean_meas, std_meas = inter_rmse_std_meas_std.T

        eval_res = EvalRes(exp_quantity=exp_quantity, mean_rmse=mean_rmse, std_rmse=std_rmse, mean_meas=mean_meas, std_meas=std_meas)
        return eval_res


@dataclass
class MeasComputer(DataComputer):
    sort_index: int = 0

    def comp_run(self, run_res: RunRes):
        if not self.check_run_res_ok(run_res):
            self.comp_run_not_ok(run_res)

        meas = calc_nr_meas(run_res)
        time = calc_meas_time(run_res)
        return meas, time


    def comp_sub_exp(self, data):
        exp_quantity, meas_time = data

        meass , times = zip(*meas_time)

        all_times = np.unique(np.concatenate(times))
        sorted_times = np.sort(all_times )

        nr_runs = np.zeros_like(sorted_times)
        added_measurements = np.zeros_like(sorted_times)
        squared_meas = np.zeros_like(sorted_times)
        
        for i, time in enumerate(sorted_times):

            for run_meas, run_time in meas_time:
                index = np.where(run_time >= time)

                try:
                    index = index[0][0]
                    nr_runs[i] = nr_runs[i] + 1
                    added_measurements[i] = added_measurements[i] + run_meas[index]
                    squared_meas[i] = squared_meas[i] + run_meas[index]**2
                except IndexError:
                    ... # skip it



        mean_measures = added_measurements / nr_runs
        std_meas = np.sqrt(squared_meas / nr_runs - (added_measurements / nr_runs)**2)
        return exp_quantity, sorted_times, mean_measures, std_meas


    def comp_exp(self, loaded_data):        
        exp_quantity, times, mean_measures, std_meas = zip(*sorted(loaded_data, key=lambda x: x[self.sort_index]))

        
        return exp_quantity, times, mean_measures, std_meas