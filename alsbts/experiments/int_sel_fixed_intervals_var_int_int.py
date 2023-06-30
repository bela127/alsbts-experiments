import numpy as np

from alts.core.experiment_runner import ExperimentRunner

from alsbts.modules.estimator import PassThroughEstimator
from alsbts.modules.query.query_selector import StreamQuerySelector


from alsbts.core.experiment_modules import StreamExperiment
from alsbts.modules.query.query_sampler import StreamQuerySampler
from alsbts.modules.query.query_decider import EmptyQueryDecider


from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_decider import ThresholdQueryDecider

from alsbts.modules.selection_criteria import FixedIntervalSelectionCriteria

from alsbts.modules.blueprint import SbBlueprint
from alts.modules.oracle.data_source import BrownianDriftDataSource

from alts.modules.data_process.process import IntegratingDSProcess



blueprints = []
for i in np.arange(0, 5, 1):
    for t in np.arange(0.1, 20, 0.1):

        bp = SbBlueprint(
            repeat=10,

            process=IntegratingDSProcess(data_source=BrownianDriftDataSource(reinit=True), integration_time=i),

            experiment_modules=StreamExperiment(
                query_selector=StreamQuerySelector(
                    query_optimizer=NoQueryOptimizer(selection_criteria= FixedIntervalSelectionCriteria(time_interval=t), query_sampler=StreamQuerySampler()),
                    query_decider=ThresholdQueryDecider(threshold=0.0),
                    ),
                estimator=PassThroughEstimator(),
            ),

            exp_name=f"int_sel_fixed_intervals_var_int{i}_int{t}",
            exp_path=f"./eval/int_sel_fixed_intervals_var_int_int",
        )
        blueprints.append(bp)

if __name__ == '__main__':
    er = ExperimentRunner(blueprints)
    er.run_experiments_parallel()
