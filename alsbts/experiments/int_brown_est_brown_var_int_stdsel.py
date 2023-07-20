import numpy as np

from alts.core.experiment_runner import ExperimentRunner

from alsbts.modules.estimator import PassThroughEstimator
from alts.core.query.query_selector import StreamQuerySelector


from alsbts.core.experiment_modules import StreamExperiment
from alsbts.modules.query.query_sampler import StreamQuerySampler
from alsbts.modules.query.query_decider import EmptyQueryDecider


from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_decider import ThresholdQueryDecider

from alsbts.modules.selection_criteria import FixedIntervalSelectionCriteria

from alsbts.modules.blueprint import SbBlueprint
from alts.modules.oracle.data_source import BrownianDriftDataSource
from alts.modules.data_process.process import IntegratingDSProcess

from alsbts.modules.estimator import BrownGPEstimator
from alsbts.modules.selection_criteria import STDSelectionCriteria



blueprints = []
for i in np.arange(0, 5, 1):
    for std in np.logspace(-2.5,-1, 15):

        bp = SbBlueprint(
            repeat=10,

            process=IntegratingDSProcess(data_source=BrownianDriftDataSource(reinit=True, rbf_leng=0.1), integration_time=i),

            experiment_modules=StreamExperiment(
            query_selector=StreamQuerySelector(
                query_optimizer=NoQueryOptimizer(
                    selection_criteria= STDSelectionCriteria(std_threshold=std),
                    query_sampler=StreamQuerySampler(),
                ),
                query_decider=ThresholdQueryDecider(threshold=0.0),
                ),
                estimator=BrownGPEstimator(length_scale = 0.1),
            ),

            exp_name=f"int_brown_est_brown_var_int{i}_stdsel{std}",
            exp_path=f"./eval/int_brown_est_brown_var_int_stdsel",
        )
        blueprints.append(bp)

if __name__ == '__main__':
    er = ExperimentRunner(blueprints)
    er.run_experiments_parallel()
