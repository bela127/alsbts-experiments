import numpy as np
from alts.core.experiment_runner import ExperimentRunner

from alsbts.modules.blueprint import SbBlueprint

from alsbts.modules.query.query_selector import StreamQuerySelector


from alsbts.core.experiment_modules import StreamExperiment
from alsbts.modules.query.query_sampler import StreamQuerySampler


from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_decider import ThresholdQueryDecider


from alsbts.modules.estimator import IntBrownGPEstimator
from alsbts.modules.selection_criteria import STDSelectionCriteria

from alts.modules.oracle.data_source import BrownianDriftDataSource
from alts.modules.data_process.process import DataSourceProcess, IntegratingDSProcess
from alts.modules.data_process.time_source import IterationTimeSource


blueprints = []
for std in np.logspace(0.2,-2.2, 20):

    bp = SbBlueprint(
        repeat=50,
        time_source = IterationTimeSource(time_step=0.5),
        experiment_modules=StreamExperiment(
            query_selector=StreamQuerySelector(
                query_optimizer=NoQueryOptimizer(
                    selection_criteria= STDSelectionCriteria(std_threshold=std),
                    query_sampler=StreamQuerySampler(),
                ),
                query_decider=ThresholdQueryDecider(threshold=0.0),
                ),
            estimator=IntBrownGPEstimator(),
        ),
        exp_name=f"brown_est_int_brown_var_sel_std{std}",
        exp_path="./eval/brown_est_int_brown_var_sel_std",
    )
    blueprints.append(bp)


if __name__ == '__main__':
    er = ExperimentRunner(blueprints)
    er.run_experiments_parallel()

    
