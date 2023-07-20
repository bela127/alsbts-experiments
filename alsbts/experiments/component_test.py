
from alts.core.experiment_runner import ExperimentRunner
from alsbts.modules.blueprint import SbBlueprint
from alts.modules.evaluator import PrintTimeSourceEvaluator, PrintNewDataPointsEvaluator, PrintQueryEvaluator
from alts.modules.data_process.time_source import IterationTimeSource
from alts.modules.data_process.process import IntegratingDSProcess, WindowDSProcess
from alts.modules.oracle.data_source import BrownianDriftDataSource

stop_time = 1000
blueprint1 = SbBlueprint(
    repeat=2,
    exp_name=f"component_test_dsp",
    exp_path="./eval/component_test",

    time_source = IterationTimeSource(time_step=1),
    evaluators=(PrintTimeSourceEvaluator(), PrintNewDataPointsEvaluator(), PrintQueryEvaluator(), *SbBlueprint.evaluators),
)

blueprint2 = SbBlueprint(
    repeat=2,
    exp_name=f"component_test_idsp",
    exp_path="./eval/component_test",

    process = IntegratingDSProcess(
        data_source=BrownianDriftDataSource(reinit=True),
    ),

    time_source = IterationTimeSource(time_step=1),
    evaluators=(PrintTimeSourceEvaluator(),*SbBlueprint.evaluators),
)

blueprint3 = SbBlueprint(
    repeat=2,
    exp_name=f"component_test_wdsp",
    exp_path="./eval/component_test",

    process = WindowDSProcess(
        data_source=BrownianDriftDataSource(reinit=True),
    ),

    time_source = IterationTimeSource(time_step=1),
    evaluators=(PrintTimeSourceEvaluator(),*SbBlueprint.evaluators),
)

blueprints = [blueprint1, blueprint2, blueprint3]


if __name__ == '__main__':
    er = ExperimentRunner(blueprints)
    er.run_experiments()#_parallel()

