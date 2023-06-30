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


blueprints = []
stop_time = 1000
for t in np.logspace(-1,2.3, 25): #np.arange(0.1, 20, 0.1):

    bp = SbBlueprint(
        repeat=50,

        experiment_modules=StreamExperiment(
            query_selector=StreamQuerySelector(
                query_optimizer=NoQueryOptimizer(selection_criteria= FixedIntervalSelectionCriteria(time_interval=t), query_sampler=StreamQuerySampler()),
                query_decider=ThresholdQueryDecider(threshold=0.0),
                ),
            estimator=PassThroughEstimator(),
        ),

        exp_name=f"fixed_intervals_var_int{t}",
        exp_path="./eval/fixed_intervals_var_int",
    )
    blueprints.append(bp)

if __name__ == '__main__':
    er = ExperimentRunner(blueprints)
    er.run_experiments_parallel()
    #input()
