
from alts.core.experiment_runner import ExperimentRunner
from alsbts.modules.blueprint import SbBlueprint


stop_time = 1000
blueprint = SbBlueprint(
    repeat=3,
    exp_name=f"component_test",
    exp_path="./eval/component_test",
)

if __name__ == '__main__':
    er = ExperimentRunner([blueprint])
    er.run_experiments_parallel()

