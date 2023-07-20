
from alts.core.experiment_runner import ExperimentRunner
from alsbts.modules.blueprint import SbBlueprint

from alsbts.modules.estimator import PassThroughEstimator
from alts.core.query.query_selector import StreamQuerySelector


from alsbts.core.experiment_modules import StreamExperiment
from alsbts.modules.query.query_sampler import StreamQuerySampler
from alsbts.modules.query.query_decider import EmptyQueryDecider


from alts.modules.query.query_optimizer import NoQueryOptimizer
from alts.modules.query.query_decider import ThresholdQueryDecider

from alsbts.modules.selection_criteria import ChangeSelectionCriteria


stop_time = 1000
blueprint = SbBlueprint(
    repeat=50,

        experiment_modules=StreamExperiment(
            query_selector=StreamQuerySelector(
                query_optimizer=NoQueryOptimizer(selection_criteria= ChangeSelectionCriteria(), query_sampler=StreamQuerySampler()),
                query_decider=ThresholdQueryDecider(threshold=0.0),
            ),
        estimator=PassThroughEstimator(),
    ),
    exp_name=f"behavior_change",
    exp_path="./eval/behavior_change",
)

if __name__ == '__main__':
    er = ExperimentRunner([blueprint])
    er.run_experiments_parallel()

